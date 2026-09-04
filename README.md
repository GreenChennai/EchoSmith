# EchoSmith

> 声匠 —— GPT-SoVITS 声音克隆工作台（训练侧）。纯合成批量生产见姊妹项目 [EchoRunner](https://github.com/GreenChennai/EchoRunner)。

流水线：**上传视频/音频 → 提音轨 → 静音切分（<10s）→ FunASR 自动打标 → 训练清单（.list）**，产出的数据集直接喂给 GPT-SoVITS 微调（训练管理在 M3 迭代），产出的权重在 EchoRunner 中消费。

## 当前版本（v0.1.0）已实现

- **引擎首运行向导**：manifest 驱动下载官方整合包（v2pro/v4/v2 三档，HF 主源 + hf-mirror 镜像回退、断点续传、7z 解压、目录校验）；也支持导入本地 .7z 或直接指向已解压目录
- **素材流水线**：任意视频/音频 → ffmpeg 提 32k 单声道音轨 → 静音检测切分（阈值/最短静音/片段长度可调）→ 逐片转写 → 生成 GPT-SoVITS 格式 `.list`
- **打标**：优先接本地 FunASR 服务（MomentShift 服务模式 `http://127.0.0.1:8000/v1`，OpenAI 兼容）
- **遇错即停**：任一步失败立即停止并给出原因，不自动重试
- **干净落盘**：`models/datasets/<名称>/`（clips/ + <名称>.list），中间长音频自动清理

## 路线图

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M1 | 骨架 + 引擎下载器 | ✔ v0.1.0 |
| M2 | 素材流水线（切分+打标） | ✔ v0.1.0 |
| M3 | UVR5 人声分离 + 训练管理（发起微调/进度/日志） | 规划中 |
| M4 | 声线卡一键生成（直接产出 EchoRunner 可消费的 card.json） | 规划中 |

## 快速开始

### 0. 引擎（一次性，三选一）

「引擎」页：
1. **自动下载**：选版本 → 开始下载（支持断点续传；HF 不通会自动切 hf-mirror）
2. **本地导入**：已下载好的官方 .7z → 「从本地导入」
3. **已有目录**：指向已解压的整合包根目录（需含 `api_v2.py` 与 `runtime/`）

### 1. 跑数据集

「素材流水线」页 → 选素材 → 填数据集名称 → 开始处理。
前置条件：本地 FunASR 服务在线（MomentShift → FunASR 服务模式）。

产出：

```
models/datasets/<名称>/
├── clips/clip_001.wav ...   # 训练片段
└── <名称>.list              # path|speaker|lang|text 标注清单
```

### 2. 训练与合成

M3 提供训练编排；现阶段可在 GPT-SoVITS 自带 webui 中用该 .list 微调，
产出的 `GPT_weights_v2/*.ckpt + SoVITS_weights_v2/*.pth` 按
[EchoRunner README](https://github.com/GreenChennai/EchoRunner#1-放置声线卡) 写成声线卡即可批量合成。

## 开发

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python main.py
```

打包（PyInstaller onefile，引擎外置）：`.venv\Scripts\python -m PyInstaller build.spec`。
推送 `v*` tag 触发 GitHub Actions 构建并挂 Release。

## License

GPL-3.0。GPT-SoVITS 为 MIT，本工具仅引导下载官方整合包并通过其脚本/进程交互，不内嵌再分发。
