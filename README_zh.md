# ComfyUI Depth Pro 2

[English](README.md) | 简体中文

[Apple Depth Pro 2](https://github.com/apple-aiml-research/ml-depth-pro-2) 的 ComfyUI 节点：
单张照片估出边缘锐利的度量深度，全分辨率推理。深度节点输入 `IMAGE`，输出同样为 `IMAGE` 的深度图
（逆深度，按帧归一化，白近黑远）。

项目主页有论文、对比视频和评测结果：
[apple-aiml-research.github.io/ml-depth-pro-2](https://apple-aiml-research.github.io/ml-depth-pro-2/)。
上游链接：[论文](https://arxiv.org/abs/2611.26603) ·
[代码](https://github.com/apple-aiml-research/ml-depth-pro-2) ·
[权重](https://huggingface.co/apple/DepthPro2)。

![1440×1080 照片的彩色深度图，红色为近、蓝色为远](examples/example.png)

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yanhuifair/comfyui-depth-pro-2
pip install -r comfyui-depth-pro-2/requirements.txt
```

`pip` 那行要在运行 ComfyUI 的那个 Python 环境里执行。
只新增 `timm`、`huggingface-hub`、`certifi`，以及在 macOS 上为可选的
[Core ML 后端](#core-mlapple-芯片)准备的 `pyobjc-framework-CoreML`；`torch`、`numpy`、`Pillow`、`av` 由 ComfyUI 本体
提供。`timm` 上限锁在 1.1 以下，因为 vendored 模型代码用的是 timm 1.0 的 vision-transformer 接口。
重启 ComfyUI 后，`model/loaders` 分类下是 **Load Depth Pro 2 Model**，`depth` 分类下是
**Depth Pro 2**。需要支持 V3 节点（`comfy_entrypoint`）的 ComfyUI 版本，实测 0.45.4。

## 用法

把 **Load Depth Pro 2 Model** 的输出连到 **Depth Pro 2** 的 `model` 口，图片接入 `image`，深度图
再接给任意图片节点（保存、ControlNet、转遮罩……）。一个加载器可以同时喂多个深度节点；权重会留在
ComfyUI 的节点缓存里，直到被缓存回收。

## 视频

**Depth Pro 2 Video** 输入 `VIDEO`、输出 `VIDEO`，配 ComfyUI 自带节点就够：**Load Video** →
**Depth Pro 2 Video** → **Save Video**。帧率和音轨沿用源视频，另外还输出 `images`，方便只要帧序列的场合。

它和把帧一帧丢给拍照节点的区别在于：一次联合推理 `window` 帧，相邻帧互相注意到，深度图不再乱爬。
用 `depthpro2-s` 跑一段 12 帧、512×512 的测试片段，按相邻归一化深度图的平均绝对差衡量：

| 推理方式 | 帧间变化 |
| --- | --- |
| 逐帧独立 | 0.056 |
| unbuffered，window 4 | 0.052 |
| buffered，window 4，overlap 2 | 0.044 |
| buffered，window 16，overlap 4（默认） | 0.030 |

显存随 `window` 增长：一次要过 `window` 帧的网络，所以这个节点的 `max_side` 默认是 1024，而不是全分辨率。
做个参考：Apple 芯片上跑 `depthpro2-h`、16 帧 1024×768、默认窗口，耗时 6 s、占 3.7 GB。

算力取决于片段步长而不是帧数：window 16 / overlap 4 每趟前进 12 帧，每帧大约算 1.3 遍；window 4 /
overlap 3 每趟只前进 1 帧，每帧要算 4 遍——同一段素材上前者 6 s、后者 23 s。

## 模型

| 预设 | 文件 | 体积 | 说明 |
| --- | --- | --- | --- |
| `depthpro2-s` | `depthpro2-s-bgcc6f8bg3.pt` | 97 MB | 最快 |
| `depthpro2-b` | `depthpro2-b-xb7m9egktw.pt` | 386 MB | |
| `depthpro2-l` | `depthpro2-l-tegstu7fim.pt` | 1.3 GB | |
| `depthpro2-h` | `depthpro2-h-eivtm5w7m5.pt` | 3.5 GB | 默认，最锐利 |
| FOV 头 | `depthpro2-fov-ew55zg4kq7.pt` | 94 MB | 各预设共用 |

权重在加载器节点首次运行时下载到 `models/depth_pro2/`。该目录已按 `depth_pro2` 注册进 `folder_paths`，
因此 `extra_model_paths.yaml` 里的额外搜索路径对查找和下载都生效。

## Core ML（Apple 芯片）

**Load Depth Pro 2 Core ML** 让模型跑在 Apple 系统的 Core ML 运行时上，而不是 PyTorch。它的输出口仍是
`DEPTH_PRO2_MODEL`，可以直接连进同一个 **Depth Pro 2** 节点。要求 Apple 芯片 + macOS 15 及以上，
Python 侧的桥接层已写进 `requirements.txt`。

Core ML 包把输入尺寸写死在包里，所以要用你手上已有的 `.pt` 自己导一个。coremltools 在 Python 3.13
之后没有 wheel，所以导出得用旧解释器——`uv` 可以现拉一个，ComfyUI 自己该用哪个 Python 还用哪个：

```bash
cd ComfyUI/custom_nodes/comfyui-depth-pro-2
uv run --python 3.13 --with coremltools --with timm --with av \
    python tools/export_coreml.py --preset depthpro2-s --size 854x480 \
    --output ../../models/depth_pro2/depthpro2-s_854x480.mlpackage
```

导出约两分钟，包体积约为 `.pt` 的一半。放进目录后按 `r` 刷新节点列表，加载器里就能选到它。想要别的
分辨率就再换一个 `--size` 导一次。

包只按自己的尺寸跑：输入会被放大到覆盖该尺寸再居中裁剪，深度图输出的也是这个尺寸，而不是输入尺寸。
Core ML 包里没有视场角头，也没有时序状态，`max_side` 对它无效。

它是「在 PyTorch 之外跑 Depth Pro 2」的一条路，不是加速方案。同一台 Apple 芯片机器上，用
`depthpro2-s` 跑 854×480、逐帧推理的实测：

| 后端 | 每帧耗时 |
| --- | --- |
| PyTorch，MPS，fp16 | 18 ms |
| Core ML，GPU | 24 ms |
| Core ML，CPU + 神经引擎 | 48 ms |
| Core ML，仅 CPU | 495 ms |

Apple 自己发布的 Core ML 包（[`apple/DepthPro2-coreml`](https://huggingface.co/apple/DepthPro2-coreml/)）
目前在 macOS 上用公开 API 加载不了：不给 `functionName` 报
`This MLModel doesn't support the multi-function description syntax`，给了又报
`.functionName must be nil unless the model type is ML Program`。Swift 里也一样，所以眼下只能用上面
那条命令自己导。

## 输入项

### Load Depth Pro 2 Model

| 输入 | 默认 | 说明 |
| --- | --- | --- |
| `model` | `depthpro2-h` | 预设，见上表。 |
| `precision` | `auto` | autocast 精度：`auto` 在 GPU 上为 bfloat16、CPU 上为 float32；无 bfloat16 的显卡选 `fp16`。高级选项。 |

### Load Depth Pro 2 Core ML

| 输入 | 默认 | 说明 |
| --- | --- | --- |
| `package` | 目录里第一个 | `models/depth_pro2/` 下的 `.mlpackage` 目录，它同时决定输出尺寸；新增后需刷新节点列表。 |
| `compute_units` | `gpu` | Core ML 的运行设备：`gpu`、`neural_engine`、`cpu` 或 `all`；神经引擎在这里通常更慢。高级选项。 |

### Depth Pro 2

| 输入 | 默认 | 说明 |
| --- | --- | --- |
| `model` | — | 加载器节点输出的 `DEPTH_PRO2_MODEL`。 |
| `image` | — | `IMAGE` 批次，逐帧推理。 |
| `focal_length_px` | `0` | 以像素为单位的焦距；`0` 表示用视场角网络估算。Core ML 包没有视场角头，也不使用焦距。 |
| `max_side` | `0` | 推理前把最长边缩到该像素数，`0` 表示保持原分辨率；Core ML 包忽略它，按包自身的尺寸推理。高级选项。 |
| `output` | `grayscale` | `grayscale`：逆深度，白色为近；`color`：深度色带，红近蓝远。 |

### Depth Pro 2 Video

| 输入 | 默认 | 说明 |
| --- | --- | --- |
| `model` | — | 任一加载器输出的 `DEPTH_PRO2_MODEL`。 |
| `video` | — | `VIDEO`，一般来自 **Load Video**。 |
| `focal_length_px` | `0` | 整段视频共用的焦距（像素）；`0` 取第一个窗口估算值的中位数。 |
| `window` | `16` | 联合推理的帧数。越大越稳、越吃显存；`1` 就是逐帧独立。 |
| `overlap` | `4` | 相邻两个片段共享的帧数，接缝处做交叉淡入淡出；必须小于 `window`，`unbuffered` 忽略它。高级选项。 |
| `mode` | `buffered` | `buffered` 推重叠片段，因此每帧还能看到其后的帧；`unbuffered` 靠 KV cache，每帧只用自己的和之前的帧。高级选项。 |
| `max_side` | `1024` | 推理前把最长边缩到该像素数，`0` 表示保持原分辨率；Core ML 包忽略它。高级选项。 |
| `output` | `grayscale` | `grayscale`：逆深度，白色为近；`color`：深度色带，红近蓝远。 |

输出：`video`（`VIDEO`，带源视频的帧率与音轨）、`images`（`IMAGE` 批次）、`fps`。

## 说明

- 输出是按帧归一化的可视化深度图，不是以米为单位的度量深度；需要原始度量深度请看上游仓库。
- 输出分辨率等于推理分辨率，即原图尺寸，或按 `max_side` 缩小后的尺寸；Core ML 包输出的是它导出时的尺寸。
- 拍照节点对批次里的每一帧都独立推理、各自估焦距，所以视频的帧序列从这里过是没有时序一致性的，
  哪怕它来自 **Get Video Components**；时序通路是 **Depth Pro 2 Video**，做法和项目主页上的视频结果一致。
- Core ML 包在帧之间不保存状态，视频节点只能逐帧跑它：没有片段联合注意力，而且整段都按包导出时的尺寸输出。
- 即使填了 `focal_length_px`，视场角头也会一并加载。
- 显存不足：先调小 `window`（它是乘数），再调小 `max_side`；填入的焦距会按同样比例缩放。

## 常见问题

| 现象 | 原因 |
| --- | --- |
| 下载时报 `SSLCertVerificationError` | 加载器改用 `certifi` 的 CA 证书下载，因为 python.org 的 macOS 安装包自带空证书。若仍失败，在该 Python 目录下运行 `Install Certificates.command`。 |
| 深度尺度或结构不对 | `focal_length_px` 为 `0` 时只是估算值，请填入真实焦距（像素）。 |

## 目录结构

```
nodes.py             加载器节点、深度节点、权重查找与下载
tools/               export_coreml.py，把 .pt 权重导成 .mlpackage
vendor/depth_pro_2/  Apple 的推理代码，许可证见 vendor/LICENSE
examples/            上文截图
```

## 许可

本仓库的节点代码为 MIT，见 [LICENSE](LICENSE)；Vendored 的 `depth_pro_2` 源码版权归 Apple，许可证见
`vendor/LICENSE`；模型权重[仅限研究用途](https://github.com/apple-aiml-research/ml-depth-pro-2/blob/main/LICENSE_MODEL)。
商用或发布成果前请确认这两份条款。

## 赞赏

如果这个节点帮到了你，可以请我喝杯咖啡：

![Fair 的赞赏码](examples/tip.jpg)
