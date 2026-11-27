import os
import ssl
import sys
import urllib.request
from dataclasses import replace
from functools import lru_cache, partial
from math import ceil
from typing import Callable, NamedTuple

import certifi
import numpy as np
import torch
from safetensors.torch import load_file
from torch.nn import functional as F
from typing_extensions import override

import folder_paths
import comfy.model_management as mm
from comfy_api.latest import ComfyExtension, InputImpl, Types, io

_VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

from depth_pro_2 import PRESETS, DepthPro2, create_model_and_transforms
from depth_pro_2.model import to_normalized_tensor
from depth_pro_2.network.fov import FOVEstimator

CHECKPOINT_FOLDER = "depth_pro2"
CHECKPOINT_DIR = os.path.join(folder_paths.models_dir, CHECKPOINT_FOLDER)
if CHECKPOINT_FOLDER in folder_paths.folder_names_and_paths:
    # extra_model_paths.yaml already created the folder, keep its search order.
    folder_paths.add_model_folder_path(CHECKPOINT_FOLDER, CHECKPOINT_DIR)
    folder_paths.folder_names_and_paths[CHECKPOINT_FOLDER][1].update(folder_paths.supported_pt_extensions)
else:
    folder_paths.folder_names_and_paths[CHECKPOINT_FOLDER] = (
        [CHECKPOINT_DIR], folder_paths.supported_pt_extensions)

PRECISIONS = {
    "auto": None,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

DepthPro2Model = io.Custom("DEPTH_PRO2_MODEL")


class DepthPro2Bundle(NamedTuple):
    model: "DepthPro2 | CoreMLDepthPro2"
    transform: Callable
    # Core ML packages only run at the size they were exported for; a torch model takes any.
    size: tuple[int, int] | None = None


# far -> near ramp, matching the paper's red near, blue far visualization.
COLORMAP = (
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
    (0.0, 1.0, 0.0),
    (1.0, 1.0, 0.0),
    (1.0, 0.0, 0.0),
)


def _download(uri: str, path: str):
    # certifi's bundle instead of the platform default, which the python.org
    # macOS builds ship empty.
    context = ssl.create_default_context(cafile=certifi.where())
    print(f"Downloading {uri.rsplit('/', 1)[-1]} to {path}")
    with urllib.request.urlopen(uri, context=context) as response, open(path + ".part", "wb") as f:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r{done / 1e6:.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    if total and done != total:
        os.remove(path + ".part")
        raise RuntimeError(f"Incomplete download of {uri}.")
    print()
    os.replace(path + ".part", path)


def _checkpoint(uri: str, name: str | None = None) -> str:
    name = name or uri.rsplit("/", 1)[-1]
    directories = folder_paths.get_folder_paths(CHECKPOINT_FOLDER)
    for directory in directories:
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            return path
    path = os.path.join(directories[0], name)
    os.makedirs(directories[0], exist_ok=True)
    _download(uri, path)
    return path


def _load(name: str, precision: str) -> DepthPro2Bundle:
    config = PRESETS[name]
    config = replace(
        config,
        checkpoint_uri=_checkpoint(config.checkpoint_uri),
        fov_checkpoint_uri=_checkpoint(config.fov_checkpoint_uri),
    )
    model, transform = create_model_and_transforms(
        config, device=mm.get_torch_device(), precision=PRECISIONS[precision]
    )
    return DepthPro2Bundle(model, transform)


FOCAL_CHECKPOINT_URI = "https://huggingface.co/apple/DepthPro2-focal/resolve/main/model.safetensors"
FOCAL_CHECKPOINT_NAME = "depthpro2-focal.safetensors"


@lru_cache(maxsize=4)
def _focal(device: str, precision: str) -> FOVEstimator:
    """The standalone focal estimator, kept per device and precision."""
    target = torch.device(device)
    dtype = PRECISIONS[precision] or (torch.float32 if target.type == "cpu" else torch.bfloat16)
    with torch.device(target):
        fov = FOVEstimator(dtype)
    fov.load_state_dict(load_file(_checkpoint(FOCAL_CHECKPOINT_URI, FOCAL_CHECKPOINT_NAME)))
    return fov.to(target).eval()


COREML_SUFFIX = ".mlpackage"
COREML_MISSING = "no Core ML package found"
COMPUTE_UNITS = ("gpu", "neural_engine", "cpu", "all")

_coreml_modules = None


def _coreml():
    """The Core ML framework and NSURL, with an install hint when the bridge is absent."""
    global _coreml_modules
    if _coreml_modules is None:
        try:
            import CoreML
            from Foundation import NSURL
        except ImportError as missing:
            raise ValueError(
                "Core ML needs the Python bridge for Apple's framework: "
                "pip install pyobjc-framework-CoreML. It runs on macOS 15 and later, "
                "on Apple silicon."
            ) from missing
        _coreml_modules = CoreML, NSURL
    return _coreml_modules


def _objc(result):
    """Split a PyObjC (value, error) pair from a value a binding already hands back bare."""
    return result if isinstance(result, tuple) else (result, None)


def _ml_dtypes(CoreML) -> dict:
    return {
        int(CoreML.MLMultiArrayDataTypeFloat16): np.float16,
        int(CoreML.MLMultiArrayDataTypeFloat32): np.float32,
        int(CoreML.MLMultiArrayDataTypeDouble): np.float64,
    }


def _coreml_packages() -> list[str]:
    """The .mlpackage folders sitting in a registered models/depth_pro2 directory."""
    packages = []
    for directory in folder_paths.get_folder_paths(CHECKPOINT_FOLDER):
        if not os.path.isdir(directory):
            continue
        for entry in sorted(os.listdir(directory)):
            path = os.path.join(directory, entry)
            if entry.endswith(COREML_SUFFIX) and os.path.isdir(path) and entry not in packages:
                packages.append(entry)
    return packages


def _coreml_package(name: str) -> str:
    directories = folder_paths.get_folder_paths(CHECKPOINT_FOLDER)
    for directory in directories:
        path = os.path.join(directory, name)
        if os.path.isdir(path):
            return path
    raise ValueError(f"{name} is not a .mlpackage in {' or '.join(directories)}.")


def _coreml_transform(image: np.ndarray) -> torch.Tensor:
    """RGB uint8 (H, W, 3) to a float (3, H, W) tensor in [0, 1], the range packages expect."""
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).to(torch.float32).div_(255.0)


def _numpy_copy(array) -> np.ndarray:
    """Copy a Core ML output out of its buffer, which the engine strides wider than the shape."""
    CoreML, _ = _coreml()
    dtype = np.dtype(_ml_dtypes(CoreML)[int(array.dataType())])
    block = {}

    def keep(raw, size):
        block["raw"] = bytes(raw)

    array.getBytesWithHandler_(keep)
    values = np.frombuffer(block["raw"], dtype=dtype)
    strides = [int(stride) * dtype.itemsize for stride in array.strides()]
    shape = tuple(int(dimension) for dimension in array.shape())
    return np.array(np.lib.stride_tricks.as_strided(values, shape=shape, strides=strides))


class CoreMLDepthPro2:
    """A Core ML Depth Pro 2 package, run by the system Core ML runtime.

    It answers the one call the depth node makes, ``infer``, but takes raw pixels rather than a
    normalized tensor, and only at the size it was exported for: other inputs are scaled to cover
    that size and center cropped, the way Apple's own demo does it.
    """

    def __init__(self, model, name: str, data_type: int, dtype, height: int, width: int):
        self.model = model
        self.name = name
        self.data_type = data_type
        self.dtype = dtype
        self.height = height
        self.width = width

    @classmethod
    def load(cls, package: str, compute_units: str) -> "CoreMLDepthPro2":
        CoreML, NSURL = _coreml()
        path = _coreml_package(package)
        compiled, error = _objc(CoreML.MLModel.compileModelAtURL_error_(
            NSURL.fileURLWithPath_(path), None))
        if compiled is None:
            raise ValueError(f"Core ML could not compile {package}: {error}")
        config = CoreML.MLModelConfiguration.alloc().init()
        config.setComputeUnits_({
            "gpu": CoreML.MLComputeUnitsCPUAndGPU,
            "neural_engine": CoreML.MLComputeUnitsCPUAndNeuralEngine,
            "cpu": CoreML.MLComputeUnitsCPUOnly,
            "all": CoreML.MLComputeUnitsAll,
        }[compute_units])
        model, error = _objc(CoreML.MLModel.modelWithContentsOfURL_configuration_error_(
            compiled, config, None))
        if model is None:
            raise ValueError(
                f"Core ML could not load {package}: {error}. Apple's published _w4 packages are "
                "not loadable on macOS yet; export a stateless one with "
                "python tools/export_coreml.py --size <W>x<H>."
            )
        description = model.modelDescription()
        if len(description.stateDescriptionsByName() or {}):
            raise ValueError(
                f"{package} keeps earlier frames in its Core ML state, which this node does not "
                "carry between frames. Export a stateless image package instead: "
                "python tools/export_coreml.py --size <W>x<H>."
            )
        inputs = description.inputDescriptionsByName() or {}
        if len(inputs) != 1:
            raise ValueError(f"{package} takes {len(inputs)} inputs, expected one frame input.")
        name, spec = next(iter(inputs.items()))
        shape = [int(dimension) for dimension in spec.multiArrayConstraint().shape()]
        if len(shape) != 4 or shape[1] != 3:
            raise ValueError(f"{package} takes a {shape} input, expected a (1, 3, H, W) frame.")
        data_type = int(spec.multiArrayConstraint().dataType())
        dtype = _ml_dtypes(CoreML).get(data_type)
        if dtype is None:
            raise ValueError(f"{package} takes a {data_type} input, expected floating point pixels.")
        return cls(model, name, data_type, dtype, shape[-2], shape[-1])

    def infer(self, x: torch.Tensor, f_px: float | None = None) -> dict:
        """Inverse depth (H, W) for one RGB frame in [0, 1], cover cropped to the package size.

        ``f_px`` is accepted for the torch model's signature: the network predicts inverse depth
        normalized by the focal length, so it does not use one.
        """
        CoreML, _ = _coreml()
        height, width = x.shape[-2:]
        if (height, width) != (self.height, self.width):
            scale = max(self.width / width, self.height / height)
            x = F.interpolate(
                x.unsqueeze(0),
                size=(ceil(height * scale), ceil(width * scale)),
                mode="bilinear",
                align_corners=False,
                antialias=scale < 1.0,
            )[0]
            top = (x.shape[-2] - self.height) // 2
            left = (x.shape[-1] - self.width) // 2
            x = x[..., top:top + self.height, left:left + self.width]
        frame = np.ascontiguousarray(x.detach().cpu().numpy()[None], dtype=self.dtype)
        strides = [stride // frame.itemsize for stride in frame.strides]
        array = _objc(CoreML.MLMultiArray.alloc().initWithBytesNoCopy_shape_dataType_strides_mutableShapedBufferProvider_(
            frame.ctypes.data, list(frame.shape), self.data_type, strides, None))[0]
        features, error = _objc(CoreML.MLDictionaryFeatureProvider.alloc().initWithDictionary_error_(
            {self.name: CoreML.MLFeatureValue.featureValueWithMultiArray_(array)}, None))
        if features is None:
            raise ValueError(f"Core ML could not bind the frame: {error}")
        prediction, error = self.model.predictionFromFeatures_options_error_(features, None, None)
        if prediction is None:
            raise ValueError(f"Core ML inference failed: {error}")
        inverse = prediction.featureValueForName_("inverse_depth").multiArrayValue()
        if inverse is None:
            raise ValueError("The package returned no inverse_depth multi array.")
        inverse = _numpy_copy(inverse)[0].astype(np.float32)
        return {"inverse_depth": torch.from_numpy(inverse)}


def _load_coreml(package: str, compute_units: str) -> DepthPro2Bundle:
    net = CoreMLDepthPro2.load(package, compute_units)
    return DepthPro2Bundle(net, _coreml_transform, (net.height, net.width))


def _colorize(gray: torch.Tensor) -> torch.Tensor:
    lut = torch.tensor(COLORMAP, device=gray.device, dtype=gray.dtype)
    idx = gray * (len(COLORMAP) - 1)
    lo = idx.floor().clamp(0, len(COLORMAP) - 2).long()
    frac = (idx - lo).unsqueeze(-1)
    return lut[lo] * (1.0 - frac) + lut[lo + 1] * frac


def _normalized(frame: torch.Tensor, transform: Callable) -> torch.Tensor:
    """One IMAGE row to the float (3, H, W) tensor a backend expects."""
    return transform(frame.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu().numpy())


def _fit(x: torch.Tensor, max_side: int) -> tuple[torch.Tensor, float]:
    """Shorten an (3, H, W) frame to max_side, with the factor a focal length scales by."""
    side = max(x.shape[-2:])
    if not 0 < max_side < side:
        return x, 1.0
    scale = max_side / side
    size = [round(dimension * scale) for dimension in x.shape[-2:]]
    return F.interpolate(x.unsqueeze(0), size=size, mode="bilinear", align_corners=False)[0], scale


def _map(inverse: torch.Tensor, output: str) -> torch.Tensor:
    """An inverse depth map as an (H, W, 3) image, normalized over this frame."""
    gray = (inverse - inverse.amin()) / (inverse.amax() - inverse.amin()).clamp_min(1e-12)
    return _colorize(gray) if output == "color" else gray.unsqueeze(-1).repeat(1, 1, 3)


class DepthPro2ModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DepthPro2ModelLoader",
            display_name="Load Depth Pro 2 Model",
            category="model/loaders",
            description=(
                "Load an Apple Depth Pro 2 preset together with its field-of-view network. The "
                "checkpoint is downloaded into models/depth_pro2 the first time it is used."
            ),
            inputs=[
                io.Combo.Input("model", options=list(PRESETS), default="depthpro2-h"),
                io.Combo.Input(
                    "precision",
                    options=list(PRECISIONS),
                    default="auto",
                    tooltip="Autocast dtype: auto is bfloat16 on GPU and float32 on CPU. Use fp16 on GPUs without bfloat16.",
                    advanced=True,
                ),
            ],
            outputs=[DepthPro2Model.Output()],
        )

    @classmethod
    def execute(cls, model, precision) -> io.NodeOutput:
        return io.NodeOutput(_load(model, precision))


class DepthPro2CoreMLModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        packages = _coreml_packages() or [COREML_MISSING]
        return io.Schema(
            node_id="DepthPro2CoreMLModelLoader",
            display_name="Load Depth Pro 2 Core ML",
            category="model/loaders",
            description=(
                "Load a Core ML Depth Pro 2 package from models/depth_pro2 and run it on the Apple "
                "silicon GPU. Export one with python tools/export_coreml.py; refresh the node list "
                "after adding it. Apple's published _w4 packages are not loadable on macOS yet."
            ),
            inputs=[
                io.Combo.Input(
                    "package",
                    options=packages,
                    default=packages[0],
                    tooltip="A .mlpackage folder in models/depth_pro2, at the size you want the depth map.",
                ),
                io.Combo.Input(
                    "compute_units",
                    options=list(COMPUTE_UNITS),
                    default="gpu",
                    tooltip="Where Core ML runs the package. GPU is the fast one for Depth Pro 2; the neural engine measured slower.",
                    advanced=True,
                ),
            ],
            outputs=[DepthPro2Model.Output()],
        )

    @classmethod
    def execute(cls, package, compute_units) -> io.NodeOutput:
        if package == COREML_MISSING:
            raise ValueError(
                "No Core ML package here yet. Create one with "
                "python tools/export_coreml.py --preset depthpro2-s --size 854x480, put it in "
                "models/depth_pro2 and refresh the node list."
            )
        return io.NodeOutput(_load_coreml(package, compute_units))


class DepthPro2Depth(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DepthPro2Depth",
            display_name="Depth Pro 2",
            category="depth",
            description=(
                "Metric monocular depth from a loaded Depth Pro 2 model, returned as an image map "
                "normalized per frame (white near, black far)."
            ),
            inputs=[
                DepthPro2Model.Input("model"),
                io.Image.Input("image"),
                io.Float.Input(
                    "focal_length_px",
                    default=0.0,
                    min=0.0,
                    max=100000.0,
                    step=1.0,
                    round=0.1,
                    tooltip="Focal length in pixels. 0 estimates it with the field-of-view network.",
                ),
                io.Int.Input(
                    "max_side",
                    default=0,
                    min=0,
                    max=16384,
                    tooltip="Downscale the longest side to this many pixels before inference, 0 keeps full resolution. Shorten it only if the image does not fit in memory; a given focal length is scaled the same way. A Core ML package ignores it and runs at the size it was exported for.",
                    advanced=True,
                ),
                io.Combo.Input(
                    "output",
                    options=["grayscale", "color"],
                    default="grayscale",
                    tooltip="grayscale: inverse depth, white near. color: depth ramp, red near, blue far.",
                ),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(cls, model, image, focal_length_px, max_side, output) -> io.NodeOutput:
        net, transform, size = model
        maps = []
        for frame in image:
            x = _normalized(frame, transform)
            scale = 1.0
            if size is None:  # A Core ML package runs at the size it was exported for.
                x, scale = _fit(x, max_side)
            f_px = focal_length_px * scale if focal_length_px > 0 else None
            maps.append(_map(net.infer(x, f_px=f_px)["inverse_depth"], output))
        return io.NodeOutput(torch.stack(maps).to(mm.intermediate_device()))


class DepthPro2Focal(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DepthPro2Focal",
            display_name="Depth Pro 2 Focal",
            category="depth",
            description=(
                "Estimate focal length in pixels from an image with Apple's standalone "
                "DepthPro2-focal network. The checkpoint is downloaded into "
                "models/depth_pro2 on first use; connect the output to a depth node's "
                "focal_length_px input."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="IMAGE batch; frames are estimated separately and the median is returned.",
                ),
                io.Combo.Input(
                    "precision",
                    options=list(PRECISIONS),
                    default="auto",
                    tooltip="Autocast dtype: auto is bfloat16 on GPU and float32 on CPU.",
                    advanced=True,
                ),
            ],
            outputs=[io.Float.Output("focal_length_px")],
        )

    @classmethod
    def execute(cls, image, precision) -> io.NodeOutput:
        device = mm.get_torch_device()
        fov = _focal(str(device), precision)
        transform = partial(to_normalized_tensor, device=device)
        values = []
        with torch.no_grad():
            for frame in image:
                values.append(fov.focal_length_px(_normalized(frame, transform)[None]).item())
        return io.NodeOutput(float(np.median(values)))


class DepthPro2Video(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DepthPro2Video",
            display_name="Depth Pro 2 Video",
            category="depth",
            description=(
                "Depth for a whole video, with neighbouring frames attended together so the map "
                "stops flickering. Frame rate and audio of the source are carried over."
            ),
            inputs=[
                DepthPro2Model.Input("model"),
                io.Video.Input("video", tooltip="The clip to measure, straight from Load Video."),

                io.Float.Input(
                    "focal_length_px",
                    default=0.0,
                    min=0.0,
                    max=100000.0,
                    step=1.0,
                    round=0.1,
                    tooltip="Focal length in pixels, shared by the clip. 0 takes the median of the estimates over the first window.",
                ),
                io.Int.Input(
                    "window",
                    default=16,
                    min=1,
                    max=256,
                    tooltip="Frames attended together. More frames are steadier and cost more memory: a window of them is through the network at once. 1 is frame by frame.",
                ),
                io.Int.Input(
                    "overlap",
                    default=4,
                    min=0,
                    max=255,
                    tooltip="Frames two neighbouring clips share, cross faded at the seam. Less than window. Ignored by the unbuffered mode.",
                    advanced=True,
                ),
                io.Combo.Input(
                    "mode",
                    options=["buffered", "unbuffered"],
                    default="buffered",
                    tooltip="buffered predicts overlapping clips, so a frame also sees the frames after it: steadiest. unbuffered predicts each frame as it arrives, from itself and the ones before it: cheaper, needs no overlap.",
                    advanced=True,
                ),
                io.Int.Input(
                    "max_side",
                    default=1024,
                    min=0,
                    max=16384,
                    tooltip="Downscale the longest side to this many pixels before inference, 0 keeps full resolution. Video costs memory fast: a whole window is inferred at once. A given focal length is scaled the same way, and a Core ML package ignores this.",
                    advanced=True,
                ),
                io.Combo.Input(
                    "output",
                    options=["grayscale", "color"],
                    default="grayscale",
                    tooltip="grayscale: inverse depth, white near. color: depth ramp, red near, blue far.",
                ),
            ],
            outputs=[
                io.Video.Output(display_name="video"),
                io.Image.Output(display_name="images"),
                io.Float.Output(display_name="fps"),
            ],
        )

    @classmethod
    def execute(cls, model, video, focal_length_px, window, overlap, mode, max_side, output) -> io.NodeOutput:
        net, transform, size = model
        components = video.get_components()
        frames = components.images
        if frames.shape[0] == 0:
            raise ValueError("The video holds no frames.")
        # Frames of one video share a size, so the first one settles the scale for all of them.
        limit = 0 if size is not None else max_side  # A Core ML package runs at its own size.
        probe, scale = _fit(_normalized(frames[0], transform), limit)
        f_px = focal_length_px * scale if focal_length_px > 0 else None

        def prepared():
            yield probe
            for frame in frames[1:]:
                yield _fit(_normalized(frame, transform), limit)[0]

        if size is None:
            predictions = net.infer_video(prepared(), window=window, overlap=overlap, f_px=f_px, mode=mode)
        else:  # Core ML packages hold no state between frames, so nothing is attended together.
            predictions = (net.infer(frame, f_px=f_px) for frame in prepared())
        maps = [_map(prediction["inverse_depth"], output) for prediction in predictions]
        images = torch.stack(maps).to(mm.intermediate_device())
        return io.NodeOutput(
            InputImpl.VideoFromComponents(Types.VideoComponents(
                images=images, audio=components.audio, frame_rate=components.frame_rate)),
            images,
            float(components.frame_rate),
        )


class DepthPro2Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            DepthPro2ModelLoader,
            DepthPro2CoreMLModelLoader,
            DepthPro2Depth,
            DepthPro2Focal,
            DepthPro2Video,
        ]


async def comfy_entrypoint() -> DepthPro2Extension:
    return DepthPro2Extension()
