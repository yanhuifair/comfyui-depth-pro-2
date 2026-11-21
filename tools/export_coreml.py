#!/usr/bin/env python3
"""Export a Depth Pro 2 checkpoint to a Core ML package for the Core ML loader node.

coremltools has no wheels past Python 3.13, so run this once per size with an older interpreter
(`uv run --python 3.13 python tools/export_coreml.py ...`); ComfyUI itself keeps running on
whatever Python it uses. The package goes where the loader node looks for it:

    python tools/export_coreml.py --preset depthpro2-s --size 854x480 \
        --output ../../models/depth_pro2/depthpro2-s_854x480.mlpackage
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

PLUGIN_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = PLUGIN_DIR.parents[1] / "models" / "depth_pro2"
sys.path.insert(0, str(PLUGIN_DIR / "vendor"))

from depth_pro_2.model import (  # noqa: E402  needs the path above
    IMAGENET_MEAN,
    IMAGENET_STD,
    PRESETS,
    create_model_and_transforms,
)


class FrameToInverseDepth(nn.Module):
    """The Core ML interface: an RGB frame in [0, 1] in, focal-normalized inverse depth out.

    The torch model wants an ImageNet-normalized input; Core ML callers hand it raw pixels, so the
    normalization travels with the graph.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        x = (frame - self.mean) / self.std
        return self.model.predict_inverse_depth(x)


def parse_size(text: str) -> tuple[int, int]:
    """Read a ``<W>x<H>`` size, the same spelling Core ML names its functions with."""
    try:
        width, height = (int(part) for part in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"size must look like 854x480, got {text!r}.") from None
    if min(width, height) < 32:
        raise argparse.ArgumentTypeError(f"{text}: both sides must be at least 32 pixels.")
    if width % 32 or height % 32:
        print(f"Note: {text} is not a multiple of 32, so the graph pads it before the backbone.")
    return width, height


def export(preset: str, size: tuple[int, int], output: Path, precision: str) -> None:
    import coremltools as ct

    width, height = size
    config = replace(PRESETS[preset], fov_checkpoint_uri=None)
    checkpoint = MODELS_DIR / Path(config.checkpoint_uri).name
    if checkpoint.exists():
        config = replace(config, checkpoint_uri=str(checkpoint))
    print(f"Loading {preset} from {config.checkpoint_uri}.")
    model, _ = create_model_and_transforms(config, device="cpu", precision=torch.float32)

    wrapper = FrameToInverseDepth(model).eval()
    example = torch.zeros(1, 3, height, width)
    program = torch.export.export(wrapper, (example,)).run_decompositions({})

    print(f"Converting to Core ML at {width}x{height}.")
    mlmodel = ct.convert(
        program,
        inputs=[ct.TensorType(name="frame", shape=(1, 3, height, width), dtype=np.float16)],
        outputs=[ct.TensorType(name="inverse_depth")],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.macOS15,
        skip_model_load=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output))

    check(mlmodel)
    print(f"Wrote {output} ({sum(f.stat().st_size for f in output.rglob('*') if f.is_file()) / 1e6:.0f} MB).")


def check(mlmodel) -> None:
    """Print what the package exposes, so a bad export fails here rather than in ComfyUI."""
    spec = mlmodel.get_spec()
    for kind, features in (("in", spec.description.input), ("out", spec.description.output)):
        for feature in features:
            array = feature.type.multiArrayType
            print(f"Interface {kind}: {feature.name} {array.dataType} {list(array.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="depthpro2-s")
    parser.add_argument("--size", type=parse_size, default=parse_size("854x480"),
                        metavar="WxH", help="input size; a multiple of 32 avoids padding (default 854x480).")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--output", type=Path, default=None,
                        help="where to write the .mlpackage (default ./<preset>_<W>x<H>.mlpackage).")
    args = parser.parse_args()

    width, height = args.size
    output = args.output or PLUGIN_DIR / f"{args.preset}_{width}x{height}.mlpackage"
    export(args.preset, args.size, output, args.precision)


if __name__ == "__main__":
    main()
