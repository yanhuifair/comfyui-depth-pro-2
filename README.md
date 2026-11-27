# ComfyUI Depth Pro 2

English | [简体中文](README_zh.md)

ComfyUI nodes for [Apple Depth Pro 2](https://github.com/apple-aiml-research/ml-depth-pro-2):
sharp metric monocular depth from a single photo, at full resolution. The depth node takes an
`IMAGE` and returns a depth map as `IMAGE` (inverse depth normalized per frame, white near /
black far).

Project page, with the paper, the comparison videos and the benchmarks:
[apple-aiml-research.github.io/ml-depth-pro-2](https://apple-aiml-research.github.io/ml-depth-pro-2/).
Upstream: [paper](https://arxiv.org/abs/2611.26603) ·
[code](https://github.com/apple-aiml-research/ml-depth-pro-2) ·
[weights](https://huggingface.co/apple/DepthPro2) ·
[Core ML weights](https://huggingface.co/apple/DepthPro2-coreml).

![The workflow in ComfyUI: Load Image and Load Depth Pro 2 Model wired into Depth Pro 2, previewing a grayscale depth map of a 1440×1080 photo](examples/example.png)

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yanhuifair/comfyui-depth-pro-2
pip install -r comfyui-depth-pro-2/requirements.txt
```

Run the `pip` line in the Python environment that runs ComfyUI.
`timm`, `huggingface-hub`, `safetensors` and `certifi` are the only additions, plus
`pyobjc-framework-CoreML` on macOS for the optional [Core ML backend](#core-ml-apple-silicon);
`torch`, `numpy`, `Pillow` and `av` already come with ComfyUI. `timm` is held below 1.1 because the
vendored model code uses the timm 1.0 vision-transformer API. Restart ComfyUI: **Load Depth Pro 2 Model** shows up under
`model/loaders` and **Depth Pro 2** under `depth`. Any ComfyUI release with V3 node support
(`comfy_entrypoint`) works; tested on 0.45.4.

## Usage

Connect **Load Depth Pro 2 Model** to the `model` socket of **Depth Pro 2**, feed your `IMAGE` into
`image`, and take the depth map to any image consumer (Save Image, ControlNet, masks, ...). One
loader can serve several depth nodes, and the weights stay loaded until ComfyUI drops them from the
node cache.

## Focal length

Focal estimation has its own loader nodes, just like depth:

- **Load Depth Pro 2 Focal Model** → **Depth Pro 2 Focal** runs Apple's standalone
  [DepthPro2-focal](https://huggingface.co/apple/DepthPro2-focal) weights with PyTorch. The 89 MB
  checkpoint downloads on first use into `models/depth_pro2/`.
- **Load Depth Pro 2 Focal Core ML** → **Depth Pro 2 Focal (Core ML)** runs Apple's Core ML
  field-of-view package from the same repository as the Core ML depth packages.

Both estimator nodes take an `IMAGE`, estimate each frame separately, and return the batch's median
as two `FLOAT`s: `focal_length_px` and `focal_length_mm`. The pixel value feeds a depth node's
`focal_length_px` socket; the millimetre value is a full-frame-equivalent lens focal length, computed
as `f_px × sensor_width_mm / image_width_px` with a default 36 mm sensor (change `sensor_width_mm`
for another camera). The Core ML FOV model estimates pixels on its fixed 896x512 input, so its
millimetre value uses 896 px as the width; the millimetre result is independent of how the source
frame was cropped. The loaded model stays cached between runs, like the depth loader's weights.

## Video

**Depth Pro 2 Video** takes a `VIDEO` and returns a `VIDEO`, so ComfyUI's own nodes are all you
need: **Load Video** → **Depth Pro 2 Video** → **Save Video**. The frame rate and the audio come from
the source, and the `images` socket is there if you want the frames on their own.

The difference from pushing frames through the photo node one by one is that a clip of `window`
frames is predicted together, so neighbouring frames attend to each other and the map stops
crawling. Measured on a 12-frame clip at 512×512 with `depthpro2-s`, as the mean absolute difference
between consecutive normalized maps:

| How the clip is predicted | frame-to-frame change |
| --- | --- |
| frame by frame | 0.056 |
| unbuffered, window 4 | 0.052 |
| buffered, window 4, overlap 2 | 0.044 |
| buffered, window 16, overlap 4 (default) | 0.030 |

Memory grows with `window`: that many frames go through the network at once. That is why `max_side`
starts at 1024 on this node instead of at full resolution. For scale, `depthpro2-h` took 6 s and
3.7 GB on Apple silicon for 16 frames at 1024×768 with the default window.

Compute follows the clip stride rather than the frame count: window 16 with overlap 4 advances 12
frames per pass, so a frame is computed about 1.3 times, while window 4 with overlap 3 advances one
frame per pass and computes each frame four times — 6 s against 23 s on that clip.

## Models

| Preset | File | Size | Notes |
| --- | --- | --- | --- |
| `depthpro2-s` | `depthpro2-s-bgcc6f8bg3.pt` | 97 MB | fastest |
| `depthpro2-b` | `depthpro2-b-xb7m9egktw.pt` | 386 MB | |
| `depthpro2-l` | `depthpro2-l-tegstu7fim.pt` | 1.3 GB | |
| `depthpro2-h` | `depthpro2-h-eivtm5w7m5.pt` | 3.5 GB | default, sharpest |
| FOV head | `depthpro2-fov-ew55zg4kq7.pt` | 94 MB | shared by every preset |
| Focal head | `depthpro2-focal.safetensors` | 89 MB | standalone, used by Depth Pro 2 Focal |

Checkpoints are downloaded on first use of the loader node, into `models/depth_pro2/`. That folder
is registered as `depth_pro2` in `folder_paths`, so extra search paths from `extra_model_paths.yaml`
are used for both lookup and download.

## Core ML (Apple silicon)

Two loaders feed the same `DEPTH_PRO2_MODEL` socket, so every downstream node stays the same.
Apple silicon and macOS 15 or later are required; the Python bridge is in `requirements.txt`.
Apple's published packages are verified here on macOS 27 through the public Core ML API.

**Load Apple Depth Pro 2 Core ML** downloads one of Apple's
[published packages](https://huggingface.co/apple/DepthPro2-coreml/) into `models/depth_pro2/` on
first use and loads a `res_<W>x<H>` function from it.

| Package | Size | Notes |
| --- | --- | --- |
| `depthpro2-s_w4.mlpackage` | 51 MB | fastest |
| `depthpro2-b_w4.mlpackage` | 188 MB | |
| `depthpro2-l_w4.mlpackage` | 891 MB | |
| `depthpro2-h_w4.mlpackage` | 1.6 GB | sharpest |
| `depthpro2-fov_512x896.mlpackage` | 91 MB | focal length, used by the Core ML focal node |

The depth packages expose `res_854x480` (default), `res_640x360`, `res_896x512`, `res_1138x640`
and `res_1280x720`. Pick one with `resolution`; the depth map comes out at that size. Their `w4`
state holds four frames: the photo node starts a fresh state for every image, while **Depth Pro 2
Video** carries one state through the whole clip. `window`, `overlap` and `mode` do not apply to
Core ML. Memory grows quickly with the size: S needs about 1.5 GB at 854x480 and 5 GB at 1280x720;
L about 4 GB and 18 GB. On a 16 GB Mac, stay with S or B at the default size.

**Load Depth Pro 2 Focal Core ML** loads the field-of-view package from the same repository into a
`DEPTH_PRO2_FOCAL_COREML_MODEL` socket; **Depth Pro 2 Focal (Core ML)** then cover-crops each frame to 896x512
and returns the batch's median focal length in both pixels (of that input) and full-frame-equivalent mm.

**Load Depth Pro 2 Core ML** lists the `.mlpackage` folders already in `models/depth_pro2/`,
including Apple's downloaded packages and your own exports. To export your own at another size,
coremltools has no wheels past Python 3.13, so run the exporter with an older interpreter — `uv`
fetches one on the spot, and ComfyUI keeps running on its own Python:

```bash
cd ComfyUI/custom_nodes/comfyui-depth-pro-2
uv run --python 3.13 --with coremltools --with timm --with av \
    python tools/export_coreml.py --preset depthpro2-s --size 854x480 \
    --output ../../models/depth_pro2/depthpro2-s_854x480.mlpackage
```

Refresh the node list (`r` on the canvas) and the package shows up in the loader. Repeat with
another `--size` for another resolution; the export takes a couple of minutes and the package about
half the size of the `.pt`. An exported package holds one size and no temporal state; `max_side` is
ignored for all Core ML packages.

This is a way to run Depth Pro 2 outside PyTorch, not a speed-up. Measured with an exported
`depthpro2-s` package at 854×480, one frame at a time, on Apple silicon:

| Backend | ms per frame |
| --- | --- |
| PyTorch, MPS, fp16 | 18 |
| Core ML, GPU | 24 |
| Core ML, CPU + neural engine | 48 |
| Core ML, CPU only | 495 |

## Inputs

### Load Depth Pro 2 Model

| Input | Default | Description |
| --- | --- | --- |
| `model` | `depthpro2-h` | Preset, see above. |
| `precision` | `auto` | Autocast dtype: `auto` is bfloat16 on GPU and float32 on CPU. Use `fp16` on GPUs without bfloat16. Advanced. |

### Load Depth Pro 2 Core ML

| Input | Default | Description |
| --- | --- | --- |
| `package` | first found | A `.mlpackage` folder in `models/depth_pro2`, which sets the output size. Refresh the node list after adding one. |
| `compute_units` | `gpu` | Where Core ML runs the package: `gpu`, `neural_engine`, `cpu` or `all`. The neural engine is usually slower here. Advanced. |

### Load Apple Depth Pro 2 Core ML

| Input | Default | Description |
| --- | --- | --- |
| `model` | `depthpro2-s_w4.mlpackage` | Apple's published package, downloaded on first use. |
| `resolution` | `default` | `res_<W>x<H>` function to run: `default` (854x480), `640x360`, `896x512`, `1138x640` or `1280x720`. The output has that size. |
| `compute_units` | `gpu` | Where Core ML runs the package: `gpu`, `neural_engine`, `cpu` or `all`. Advanced. |

### Load Depth Pro 2 Focal Model

| Input | Default | Description |
| --- | --- | --- |
| `precision` | `auto` | Autocast dtype: `auto` is bfloat16 on GPU and float32 on CPU. Advanced. |

Output: `DEPTH_PRO2_FOCAL_MODEL`, for **Depth Pro 2 Focal**.

### Load Depth Pro 2 Focal Core ML

| Input | Default | Description |
| --- | --- | --- |
| `compute_units` | `gpu` | Where Core ML runs the field-of-view package: `gpu`, `neural_engine`, `cpu` or `all`. Advanced. |

Output: `DEPTH_PRO2_FOCAL_COREML_MODEL`, for **Depth Pro 2 Focal (Core ML)**.

### Depth Pro 2

| Input | Default | Description |
| --- | --- | --- |
| `model` | — | `DEPTH_PRO2_MODEL` from the loader node. |
| `image` | — | `IMAGE` batch, predicted one frame at a time. |
| `focal_length_px` | `0` | Focal length in pixels. `0` estimates it with the field-of-view network; a Core ML package has none and does not use one. |
| `max_side` | `0` | Downscale the longest side to this many pixels before inference; `0` keeps full resolution. A Core ML package ignores it and runs at its own size. Advanced. |
| `output` | `grayscale` | `grayscale`: inverse depth, white near. `color`: depth ramp, red near, blue far. |

### Depth Pro 2 Focal

| Input | Default | Description |
| --- | --- | --- |
| `model` | — | `DEPTH_PRO2_FOCAL_MODEL` from **Load Depth Pro 2 Focal Model**. |
| `image` | — | `IMAGE` batch, estimated one frame at a time. |
| `sensor_width_mm` | `36.0` | Sensor width in mm for `focal_length_mm`; 36 mm is full-frame / 35mm equivalent. Advanced. |

Outputs: `focal_length_px` and `focal_length_mm` (`FLOAT`), both the median over the batch.

### Depth Pro 2 Focal (Core ML)

| Input | Default | Description |
| --- | --- | --- |
| `model` | — | `DEPTH_PRO2_FOCAL_COREML_MODEL` from **Load Depth Pro 2 Focal Core ML**. |
| `image` | — | `IMAGE` batch; frames are cover cropped to 896x512 and estimated one at a time. |
| `sensor_width_mm` | `36.0` | Sensor width in mm for `focal_length_mm`; 36 mm is full-frame / 35mm equivalent. Advanced. |

Outputs: `focal_length_px` (of the 896x512 input) and `focal_length_mm` (`FLOAT`), both the median over the batch.

### Depth Pro 2 Video

| Input | Default | Description |
| --- | --- | --- |
| `model` | — | `DEPTH_PRO2_MODEL` from either loader. |
| `video` | — | `VIDEO`, typically from **Load Video**. |
| `focal_length_px` | `0` | Focal length in pixels for the whole clip; `0` takes the median of the estimates over the first window. |
| `window` | `16` | Frames attended together. Larger is steadier and costs more memory; `1` is frame by frame. |
| `overlap` | `4` | Frames two neighbouring clips share, cross-faded at the seam. Must stay below `window`; ignored by `unbuffered` and by Core ML. Advanced. |
| `mode` | `buffered` | `buffered` predicts overlapping clips, so a frame also sees later frames; `unbuffered` predicts each frame as it arrives, from itself and the earlier ones. Core ML uses its own four-frame state instead. Advanced. |
| `max_side` | `1024` | Downscale the longest side to this many pixels before inference; `0` keeps full resolution. A Core ML package ignores it. Advanced. |
| `output` | `grayscale` | `grayscale`: inverse depth, white near. `color`: depth ramp, red near, blue far. |

Outputs: `video` (`VIDEO`, with the source frame rate and audio), `images` (`IMAGE` batch), `fps`.

## Notes

- The output is a normalized visualization map, not metric depth in meters. See the upstream repo
  if you need raw metric depth.
- Output resolution is the inference resolution: the input size, or the `max_side`-reduced size. A
  Core ML package returns the size it was exported for.
- The photo node predicts each frame of a batch on its own, each with its own focal length, so a
  batch of video frames has no consistency between them — even when it comes from
  **Get Video Components**. **Depth Pro 2 Video** is the temporal path, attending a clip together
  the way the video results on the project page do.
- Apple's `w4` Core ML packages keep four frames of state. The photo node resets it per image; the
  video node keeps one state for the whole clip, so neighbouring frames still inform each other.
  Exported packages have no state and always run frame by frame.
- The field-of-view head is loaded for every preset, even when `focal_length_px` is supplied.
- **Depth Pro 2 Focal** uses Apple's newer standalone focal weights, not the `.pt` FOV head shared by
  the depth presets; both estimate pixels for the frame they are given.
- Out of memory: lower `max_side`; on the video node lower `window` first, it is the multiplier. A
  supplied focal length is scaled by the same factor.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `SSLCertVerificationError` while downloading | The loader downloads through `certifi`, because python.org macOS builds ship an empty CA bundle. If it still fails, run `Install Certificates.command` in that Python's folder. |
| Hugging Face download fails with a proxy 502 | The focal and Apple Core ML nodes download from Hugging Face. Set `HTTP_PROXY`/`HTTPS_PROXY` to a working proxy before starting ComfyUI, or place the files under `models/depth_pro2/` yourself. |
| Depth scale or geometry looks wrong | `focal_length_px` is a guess when left at `0`; pass the real focal length in pixels. |

## Files

```
nodes.py             loader nodes, depth and focal nodes, checkpoint lookup and download
tools/               export_coreml.py, torch checkpoint to .mlpackage
vendor/depth_pro_2/  Apple's inference code, see vendor/LICENSE
examples/            screenshot above
```

## License

The node code is MIT, see [LICENSE](LICENSE). The vendored `depth_pro_2` sources are Apple's, under
`vendor/LICENSE`. The model weights are
[research-use only](https://github.com/apple-aiml-research/ml-depth-pro-2/blob/main/LICENSE_MODEL).
Check both before shipping anything made with them.

## Support

WeChat tip code, if this node saved you some time:

![Fair's WeChat tip code](examples/tip.jpg)
