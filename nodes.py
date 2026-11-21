import os
import ssl
import sys
import urllib.request
from dataclasses import replace

import certifi
import torch
from torch.nn import functional as F
from typing_extensions import override

import folder_paths
import comfy.model_management as mm
from comfy_api.latest import ComfyExtension, io

_VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

from depth_pro_2 import PRESETS, create_model_and_transforms

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

# far -> near ramp, matching the paper's red near, blue far visualization.
COLORMAP = (
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
    (0.0, 1.0, 0.0),
    (1.0, 1.0, 0.0),
    (1.0, 0.0, 0.0),
)

_loaded = {"key": None, "model": None, "transform": None}


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


def _checkpoint(uri: str) -> str:
    name = uri.rsplit("/", 1)[-1]
    directories = folder_paths.get_folder_paths(CHECKPOINT_FOLDER)
    for directory in directories:
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            return path
    path = os.path.join(directories[0], name)
    os.makedirs(directories[0], exist_ok=True)
    _download(uri, path)
    return path


def _load(name: str, precision: str, device: torch.device):
    key = (name, precision, str(device))
    if _loaded["key"] != key:
        _loaded["model"] = None
        _loaded["transform"] = None
        mm.soft_empty_cache()
        config = PRESETS[name]
        config = replace(
            config,
            checkpoint_uri=_checkpoint(config.checkpoint_uri),
            fov_checkpoint_uri=_checkpoint(config.fov_checkpoint_uri),
        )
        model, transform = create_model_and_transforms(
            config, device=device, precision=PRECISIONS[precision]
        )
        _loaded.update(key=key, model=model, transform=transform)
    return _loaded["model"], _loaded["transform"]


def _colorize(gray: torch.Tensor) -> torch.Tensor:
    lut = torch.tensor(COLORMAP, device=gray.device, dtype=gray.dtype)
    idx = gray * (len(COLORMAP) - 1)
    lo = idx.floor().clamp(0, len(COLORMAP) - 2).long()
    frac = (idx - lo).unsqueeze(-1)
    return lut[lo] * (1.0 - frac) + lut[lo + 1] * frac


class DepthPro2Depth(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DepthPro2Depth",
            display_name="Depth Pro 2",
            category="depth",
            description=(
                "Metric monocular depth from Apple Depth Pro 2, returned as an image map "
                "normalized per frame (white near, black far)."
            ),
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("model", options=list(PRESETS), default="depthpro2-h"),
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
                    tooltip="Downscale the longest side to this many pixels before inference, 0 keeps full resolution. Shorten it only if the image does not fit in memory; a given focal length is scaled the same way.",
                    advanced=True,
                ),
                io.Combo.Input(
                    "precision",
                    options=list(PRECISIONS),
                    default="auto",
                    tooltip="Autocast dtype: auto is bfloat16 on GPU and float32 on CPU. Use fp16 on GPUs without bfloat16.",
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
    def execute(cls, image, model, focal_length_px, max_side, precision, output) -> io.NodeOutput:
        net, transform = _load(model, precision, mm.get_torch_device())
        maps = []
        for frame in image:
            x = transform(frame.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu().numpy())
            f_px = focal_length_px if focal_length_px > 0 else None
            height, width = x.shape[-2:]
            if 0 < max_side < max(height, width):
                scale = max_side / max(height, width)
                x = F.interpolate(
                    x.unsqueeze(0),
                    size=(round(height * scale), round(width * scale)),
                    mode="bilinear",
                    align_corners=False,
                )[0]
                if f_px is not None:
                    f_px *= scale
            inverse = net.infer(x, f_px=f_px)["inverse_depth"]
            gray = (inverse - inverse.amin()) / (inverse.amax() - inverse.amin()).clamp_min(1e-12)
            maps.append(_colorize(gray) if output == "color" else gray.unsqueeze(-1).repeat(1, 1, 3))
        return io.NodeOutput(torch.stack(maps).to(mm.intermediate_device()))


class DepthPro2Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            DepthPro2Depth,
        ]


async def comfy_entrypoint() -> DepthPro2Extension:
    return DepthPro2Extension()
