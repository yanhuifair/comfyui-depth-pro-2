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
只新增 `timm`、`huggingface-hub`、`certifi` 三项；`torch`、`numpy`、`Pillow`、`av` 由 ComfyUI 本体
提供。`timm` 上限锁在 1.1 以下，因为 vendored 模型代码用的是 timm 1.0 的 vision-transformer 接口。
重启 ComfyUI 后，`model/loaders` 分类下是 **Load Depth Pro 2 Model**，`depth` 分类下是
**Depth Pro 2**。需要支持 V3 节点（`comfy_entrypoint`）的 ComfyUI 版本，实测 0.45.4。

## 用法

把 **Load Depth Pro 2 Model** 的输出连到 **Depth Pro 2** 的 `model` 口，图片接入 `image`，深度图
再接给任意图片节点（保存、ControlNet、转遮罩……）。一个加载器可以同时喂多个深度节点；权重会留在
ComfyUI 的节点缓存里，直到被缓存回收。

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

## 输入项

### Load Depth Pro 2 Model

| 输入 | 默认 | 说明 |
| --- | --- | --- |
| `model` | `depthpro2-h` | 预设，见上表。 |
| `precision` | `auto` | autocast 精度：`auto` 在 GPU 上为 bfloat16、CPU 上为 float32；无 bfloat16 的显卡选 `fp16`。高级选项。 |

### Depth Pro 2

| 输入 | 默认 | 说明 |
| --- | --- | --- |
| `model` | — | 加载器节点输出的 `DEPTH_PRO2_MODEL`。 |
| `image` | — | `IMAGE` 批次，逐帧推理。 |
| `focal_length_px` | `0` | 以像素为单位的焦距；`0` 表示用视场角网络估算。 |
| `max_side` | `0` | 推理前把最长边缩到该像素数，`0` 表示保持原分辨率。高级选项。 |
| `output` | `grayscale` | `grayscale`：逆深度，白色为近；`color`：深度色带，红近蓝远。 |

## 说明

- 输出是按帧归一化的可视化深度图，不是以米为单位的度量深度；需要原始度量深度请看上游仓库。
- 输出分辨率等于推理分辨率，即原图尺寸，或按 `max_side` 缩小后的尺寸。
- 节点逐帧独立推理，各帧使用自己的焦距；项目主页上那种时序一致的视频效果来自上游的多帧联合推理，
  因此 ComfyUI 里的批次帧与帧之间没有时序一致性。
- 即使填了 `focal_length_px`，视场角头也会一并加载。
- 显存不足：调小 `max_side`，填入的焦距会按同样比例缩放。

## 常见问题

| 现象 | 原因 |
| --- | --- |
| 下载时报 `SSLCertVerificationError` | 加载器改用 `certifi` 的 CA 证书下载，因为 python.org 的 macOS 安装包自带空证书。若仍失败，在该 Python 目录下运行 `Install Certificates.command`。 |
| 深度尺度或结构不对 | `focal_length_px` 为 `0` 时只是估算值，请填入真实焦距（像素）。 |

## 目录结构

```
nodes.py             加载器节点、深度节点、权重查找与下载
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
