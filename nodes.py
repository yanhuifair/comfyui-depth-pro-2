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
from huggingface_hub import list_repo_files
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
from depth_pro_2.network.fov import FOVEstimator, median_focal_length_px

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
DepthPro2FocalModel = io.Custom("DEPTH_PRO2_FOCAL_MODEL")
DepthPro2FocalCoreMLModel = io.Custom("DEPTH_PRO2_FOCAL_COREML_MODEL")


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

APPLE_COREML_REPO = "apple/DepthPro2-coreml"
APPLE_COREML_PACKAGES = (
    "depthpro2-s_w4.mlpackage",
    "depthpro2-b_w4.mlpackage",
    "depthpro2-l_w4.mlpackage",
    "depthpro2-h_w4.mlpackage",
)
APPLE_RESOLUTIONS = ("854x480", "640x360", "896x512", "1138x640", "1280x720")
APPLE_FOV_PACKAGE = "depthpro2-fov_512x896.mlpackage"
APPLE_FOV_SIZE = (512, 896)  # The package's fixed (H, W), a 16:9 frame.

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


def _apple_coreml_package(name: str) -> str:
    """Download one of Apple's published .mlpackage folders into models/depth_pro2."""
    directories = folder_paths.get_folder_paths(CHECKPOINT_FOLDER)
    for directory in directories:
        path = os.path.join(directory, name)
        if os.path.isfile(os.path.join(path, "Manifest.json")):
            return path
    path = os.path.join(directories[0], name)
    files = [f for f in list_repo_files(APPLE_COREML_REPO) if f.startswith(name + "/")]
    if not files:
        raise ValueError(f"{name} is not in {APPLE_COREML_REPO}.")
    for file in files:
        target = os.path.join(path, *file.split("/")[1:])
        if os.path.isfile(target):
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        _download(f"https://huggingface.co/{APPLE_COREML_REPO}/resolve/main/{file}", target)
    return path


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


def _cover(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Scale a (3, H, W) frame to cover (height, width), then centre crop, as Apple's demo does."""
    source_h, source_w = x.shape[-2:]
    if (source_h, source_w) == (height, width):
        return x
    scale = max(width / source_w, height / source_h)
    x = F.interpolate(
        x.unsqueeze(0),
        size=(ceil(source_h * scale), ceil(source_w * scale)),
        mode="bilinear",
        align_corners=False,
        antialias=scale < 1.0,
    )[0]
    top = (x.shape[-2] - height) // 2
    left = (x.shape[-1] - width) // 2
    return x[..., top:top + height, left:left + width]


def _ml_array(values: np.ndarray, data_type: int):
    """Bind a contiguous numpy array to Core ML as a multi array, without copying."""
    CoreML, _ = _coreml()
    values = np.ascontiguousarray(values)
    strides = [stride // values.itemsize for stride in values.strides]
    return _objc(CoreML.MLMultiArray.alloc().initWithBytesNoCopy_shape_dataType_strides_mutableShapedBufferProvider_(
        values.ctypes.data, list(values.shape), data_type, strides, None))[0]


class CoreMLDepthPro2:
    """A Core ML Depth Pro 2 package, run by the system Core ML runtime.

    It answers the one call the depth node makes, ``infer``, but takes raw pixels rather than a
    normalized tensor, and only at the size its selected function was exported for. Stateful
    packages keep earlier frames in Core ML state: ``infer`` starts a fresh state for a photo, and
    ``keep_state=True`` continues the one ``reset_state`` made at the start of a video.
    """

    def __init__(self, model, name: str, data_type: int, dtype, height: int, width: int,
                 stateful: bool):
        self.model = model
        self.name = name
        self.data_type = data_type
        self.dtype = dtype
        self.height = height
        self.width = width
        self.stateful = stateful
        self.state = None
        self.reset_state()

    @classmethod
    def load(cls, package: str, compute_units: str, function: str | None = None) -> "CoreMLDepthPro2":
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
        if function is not None:
            config.setFunctionName_(function)
        model, error = _objc(CoreML.MLModel.modelWithContentsOfURL_configuration_error_(
            compiled, config, None))
        if model is None:
            raise ValueError(
                f"Core ML could not load {package}: {error}. For a package with several sizes, "
                "pick one with the resolution input."
            )
        description = model.modelDescription()
        stateful = bool(description.stateDescriptionsByName() or {})
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
        return cls(model, name, data_type, dtype, shape[-2], shape[-1], stateful)

    def reset_state(self):
        self.state = self.model.newState() if self.stateful else None

    def infer(self, x: torch.Tensor, f_px: float | None = None,
              keep_state: bool = False) -> dict:
        """Inverse depth (H, W) for one RGB frame in [0, 1], cover cropped to the function's size.

        ``f_px`` is accepted for the torch model's signature: the network predicts inverse depth
        normalized by the focal length, so it does not use one.
        """
        CoreML, _ = _coreml()
        frame = np.ascontiguousarray(
            _cover(x, self.height, self.width).detach().cpu().numpy()[None], dtype=self.dtype)
        array = _ml_array(frame, self.data_type)
        features, error = _objc(CoreML.MLDictionaryFeatureProvider.alloc().initWithDictionary_error_(
            {self.name: CoreML.MLFeatureValue.featureValueWithMultiArray_(array)}, None))
        if features is None:
            raise ValueError(f"Core ML could not bind the frame: {error}")
        if self.stateful:
            if not keep_state or self.state is None:
                self.reset_state()
            prediction, error = _objc(self.model.predictionFromFeatures_usingState_error_(
                features, self.state, None))
        else:
            prediction, error = _objc(self.model.predictionFromFeatures_options_error_(
                features, None, None))
        if prediction is None:
            raise ValueError(f"Core ML inference failed: {error}")
        inverse = prediction.featureValueForName_("inverse_depth").multiArrayValue()
        if inverse is None:
            raise ValueError("The package returned no inverse_depth multi array.")
        inverse = _numpy_copy(inverse)[0].astype(np.float32)
        return {"inverse_depth": torch.from_numpy(inverse)}


def _load_coreml(package: str, compute_units: str, function: str | None = None) -> DepthPro2Bundle:
    net = CoreMLDepthPro2.load(package, compute_units, function)
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
                "silicon GPU. Export one with python tools/export_coreml.py, or use the Apple "
                "Core ML loader to fetch Apple's published packages; refresh the node list after "
                "adding one."
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


class DepthPro2AppleCoreMLLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DepthPro2AppleCoreMLLoader",
            display_name="Load Apple Depth Pro 2 Core ML",
            category="model/loaders",
            description=(
                "Download and load Apple's published Depth Pro 2 Core ML packages. The w4 packages "
                "hold four frames of state: a depth node starts a new state per photo, and a video "
                "node carries one state through the whole clip."
            ),
            inputs=[
                io.Combo.Input(
                    "model",
                    options=list(APPLE_COREML_PACKAGES),
                    default=APPLE_COREML_PACKAGES[0],
                    tooltip="Apple's published package; downloaded into models/depth_pro2 on first use.",
                ),
                io.Combo.Input(
                    "resolution",
                    options=("default",) + APPLE_RESOLUTIONS,
                    default="default",
                    tooltip="A res_<W>x<H> function in the package. default is its first exported size, 854x480.",
                ),
                io.Combo.Input(
                    "compute_units",
                    options=list(COMPUTE_UNITS),
                    default="gpu",
                    tooltip="Where Core ML runs the package. GPU is the fast one; the neural engine measured slower.",
                    advanced=True,
                ),
            ],
            outputs=[DepthPro2Model.Output()],
        )

    @classmethod
    def execute(cls, model, resolution, compute_units) -> io.NodeOutput:
        function = None if resolution == "default" else f"res_{resolution}"
        return io.NodeOutput(_load_coreml(_apple_coreml_package(model), compute_units, function))


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


class DepthPro2FocalModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DepthPro2FocalModelLoader",
            display_name="Load Depth Pro 2 Focal Model",
            category="model/loaders",
            description=(
                "Load Apple's standalone DepthPro2-focal weights with PyTorch. The checkpoint is "
                "downloaded into models/depth_pro2 on first use; connect the output to Depth Pro 2 "
                "Focal, and that node's output to a depth node's focal_length_px input."
            ),
            inputs=[
                io.Combo.Input(
                    "precision",
                    options=list(PRECISIONS),
                    default="auto",
                    tooltip="Autocast dtype: auto is bfloat16 on GPU and float32 on CPU.",
                    advanced=True,
                ),
            ],
            outputs=[DepthPro2FocalModel.Output()],
        )

    @classmethod
    def execute(cls, precision) -> io.NodeOutput:
        return io.NodeOutput(_focal(str(mm.get_torch_device()), precision))


class DepthPro2Focal(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DepthPro2Focal",
            display_name="Depth Pro 2 Focal",
            category="depth",
            description=(
                "Estimate focal length in pixels from an image with a loaded DepthPro2-focal "
                "model. Frames of a batch are estimated separately and the median is returned."
            ),
            inputs=[
                DepthPro2FocalModel.Input("model"),
                io.Image.Input(
                    "image",
                    tooltip="IMAGE batch; frames are estimated separately and the median is returned.",
                ),
            ],
            outputs=[io.Float.Output("focal_length_px")],
        )

    @classmethod
    def execute(cls, model, image) -> io.NodeOutput:
        transform = partial(to_normalized_tensor, device=next(model.parameters()).device)
        values = []
        with torch.no_grad():
            for frame in image:
                values.append(model.focal_length_px(_normalized(frame, transform)[None]).item())
        return io.NodeOutput(float(np.median(values)))


@lru_cache(maxsize=4)
def _coreml_fov(compute_units: str):
    """Apple's Core ML field-of-view package, kept per compute-unit choice."""
    CoreML, NSURL = _coreml()
    path = _apple_coreml_package(APPLE_FOV_PACKAGE)
    compiled, error = _objc(CoreML.MLModel.compileModelAtURL_error_(NSURL.fileURLWithPath_(path), None))
    if compiled is None:
        raise ValueError(f"Core ML could not compile {APPLE_FOV_PACKAGE}: {error}")
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
        raise ValueError(f"Core ML could not load {APPLE_FOV_PACKAGE}: {error}")
    description = model.modelDescription()
    if "fov" not in (description.outputDescriptionsByName() or {}):
        raise ValueError(f"{APPLE_FOV_PACKAGE} has no fov output.")
    inputs = description.inputDescriptionsByName() or {}
    if len(inputs) != 1:
        raise ValueError(f"{APPLE_FOV_PACKAGE} takes {len(inputs)} inputs, expected one frame input.")
    name, spec = next(iter(inputs.items()))
    shape = [int(dimension) for dimension in spec.multiArrayConstraint().shape()]
    if len(shape) != 4 or tuple(shape[-2:]) != APPLE_FOV_SIZE:
        raise ValueError(f"{APPLE_FOV_PACKAGE} takes a {shape} input, expected (1, 3, 512, 896).")
    data_type = int(spec.multiArrayConstraint().dataType())
    dtype = _ml_dtypes(CoreML).get(data_type)
    if dtype is None:
        raise ValueError(f"{APPLE_FOV_PACKAGE} takes a {data_type} input, expected floating point pixels.")
    return model, name, data_type, dtype


class DepthPro2FocalCoreMLLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DepthPro2FocalCoreMLLoader",
            display_name="Load Depth Pro 2 Focal Core ML",
            category="model/loaders",
            description=(
                "Download and load Apple's Core ML field-of-view package from "
                "apple/DepthPro2-coreml. Connect the output to Depth Pro 2 Focal (Core ML)."
            ),
            inputs=[
                io.Combo.Input(
                    "compute_units",
                    options=list(COMPUTE_UNITS),
                    default="gpu",
                    tooltip="Where Core ML runs the package. GPU is the fast one; the neural engine measured slower.",
                    advanced=True,
                ),
            ],
            outputs=[DepthPro2FocalCoreMLModel.Output()],
        )

    @classmethod
    def execute(cls, compute_units) -> io.NodeOutput:
        return io.NodeOutput(_coreml_fov(compute_units))


class DepthPro2FocalCoreML(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DepthPro2FocalCoreML",
            display_name="Depth Pro 2 Focal (Core ML)",
            category="depth",
            description=(
                "Estimate focal length in pixels with a loaded Core ML field-of-view model. "
                "Frames are cover cropped to its fixed 896x512 input; the batch's median is "
                "returned."
            ),
            inputs=[
                DepthPro2FocalCoreMLModel.Input("model"),
                io.Image.Input(
                    "image",
                    tooltip="IMAGE batch; frames are estimated separately and the median is returned.",
                ),
            ],
            outputs=[io.Float.Output("focal_length_px")],
        )

    @classmethod
    def execute(cls, model, image) -> io.NodeOutput:
        CoreML, _ = _coreml()
        fov, name, data_type, dtype = model
        values = []
        for frame in image:
            x = _cover(_normalized(frame, _coreml_transform), *APPLE_FOV_SIZE)
            pixels = np.ascontiguousarray(x.detach().cpu().numpy()[None], dtype=dtype)
            array = _ml_array(pixels, data_type)
            features, error = _objc(CoreML.MLDictionaryFeatureProvider.alloc().initWithDictionary_error_(
                {name: CoreML.MLFeatureValue.featureValueWithMultiArray_(array)}, None))
            if features is None:
                raise ValueError(f"Core ML could not bind the frame: {error}")
            prediction, error = _objc(fov.predictionFromFeatures_options_error_(features, None, None))
            if prediction is None:
                raise ValueError(f"Core ML inference failed: {error}")
            angles = np.ascontiguousarray(
                _numpy_copy(prediction.featureValueForName_("fov").multiArrayValue()), dtype=np.float32)
            values.append(median_focal_length_px(torch.from_numpy(angles), *APPLE_FOV_SIZE).item())
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
                    tooltip="Frames attended together. More frames are steadier and cost more memory: a window of them is through the network at once. 1 is frame by frame. Core ML packages ignore this: their w4 state carries four frames.",
                ),
                io.Int.Input(
                    "overlap",
                    default=4,
                    min=0,
                    max=255,
                    tooltip="Frames two neighbouring clips share, cross faded at the seam. Less than window. Ignored by the unbuffered mode and by Core ML packages.",
                    advanced=True,
                ),
                io.Combo.Input(
                    "mode",
                    options=["buffered", "unbuffered"],
                    default="buffered",
                    tooltip="buffered predicts overlapping clips, so a frame also sees the frames after it: steadiest. unbuffered predicts each frame as it arrives, from itself and the ones before it: cheaper, needs no overlap. Core ML packages use their own four-frame state and ignore this.",
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
        else:  # Core ML packages step one frame at a time, carrying their state through the clip.
            net.reset_state()
            predictions = (net.infer(frame, f_px=f_px, keep_state=True) for frame in prepared())
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
            DepthPro2AppleCoreMLLoader,
            DepthPro2FocalModelLoader,
            DepthPro2FocalCoreMLLoader,
            DepthPro2Depth,
            DepthPro2Focal,
            DepthPro2FocalCoreML,
            DepthPro2Video,
        ]


async def comfy_entrypoint() -> DepthPro2Extension:
    return DepthPro2Extension()
