<div align="center">
  <img src="./assets/readme/hero.svg" width="100%" alt="EchoSmith — 声音克隆工作台:一段素材变成一个可批量合成的声线模型">
</div>

# EchoSmith

**声匠 · GPT-SoVITS 声音克隆工作台。** 在官方 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 整合包之上,一条龙完成:**引擎部署 → 素材切分打标 → 六阶段微调训练 → 声线卡 → 批量语音合成**。

两种使用方式:

- **Skill / CLI(主路径)**:`echosmith` 技能 + `python -m echosmith.cli`,面向 AI 助手与自动化,一条命令一个环节,遇错即停
- **GUI(辅助模式)**:Dear PyGui 深色工作台,人工操作与观察用,功能与 CLI 完全对等

<div align="center">
  <img src="./assets/readme/shot_train.png" width="49.5%" alt="训练页:数据集选择、轮数与批大小、六阶段进度条">
  <img src="./assets/readme/shot_card.png" width="49.5%" alt="声线卡页:权重扫描、多情感参考音频行、生成 card.json">
</div>

## 它能做什么

| 模块 | 能力 |
|---|---|
| **引擎部署** | manifest 驱动下载官方整合包(HF + hf-mirror 回退、断点续传、7-Zip 解压、目录校验);支持本地 `.7z` / 已有目录导入 |
| **素材流水线** | 视频/音频 → 32k 单声道 → 可选 UVR5 人声分离 → 静音切分(阈值/片段长度可调)→ 自动打标 → GPT-SoVITS 格式 `.list` |
| **打标** | 本地 FunASR 服务优先(OpenAI 兼容),不可达时自动回退整合包内置 funasr 离线批量 |
| **六阶段训练** | `1-text → 2-hubert → 2-sv → 3-semantic → s2(SoVITS) → s1(GPT)`;版本自动探测(v2ProPlus/v2Pro/v2/v1);日志实时、遇错即停 |
| **声线卡** | 扫描训练产物 → 绑定多情感参考音频(3~10 秒硬校验)→ 产出 `card.json` |
| **批量合成** | 引擎托管(自动起停、端口就绪)、每行一条的合成队列、`【情感】`前缀单行覆盖、按声线/日期落盘、可选 mp3 |
| **设备** | CPU 训练(推荐)+ NVIDIA CUDA 原生 |

> 训练编排与 UVR5 走整合包内部接口,**契约已逐行对照 GPT-SoVITS 官方源码核验**;`2-sv` 阶段官方脚本逐行吞错会静默产出废数据,本工具按产物数量守卫拦下。

## 全流程

<div align="center">
  <img src="./assets/readme/pipeline.svg" width="100%" alt="素材流水线 → 静音切分 → 自动打标 → 训练清单 → 六阶段训练 → 声线卡 → 批量合成">
</div>

真实记录:197 条切片(21.4 分钟 UVR5 净音),prepare 四阶段 + s2×8 + s1×15 全程 CPU 约 2.5 小时,GPT 音素准确率 top_3_acc 0.795,合成验证通过。

## 快速开始(CLI / Agent)

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
# 以下命令统一形式:PYTHONPATH=src .venv\Scripts\python.exe -m echosmith.cli <子命令>

# 1. 部署引擎(HF 不通加 --mirror;或 --local 指向已有 .7z/目录)
PYTHONPATH=src .venv\Scripts\python.exe -m echosmith.cli deploy-engine

# 2. 素材 → 数据集
PYTHONPATH=src .venv\Scripts\python.exe -m echosmith.cli dataset --src 素材.mp4 --name myvoice

# 3. 检查修正 models/datasets/myvoice/myvoice.list 文本(转写校对,文本=音频逐字对应)

# 4. 六阶段训练(CPU,时长与素材量成正比)
PYTHONPATH=src .venv\Scripts\python.exe -m echosmith.cli train --list models/datasets/myvoice/myvoice.list --exp myvoice --s2 8 --s1 15

# 5. 声线卡
PYTHONPATH=src .venv\Scripts\python.exe -m echosmith.cli card --exp myvoice --voice myvoice --ref models/datasets/myvoice/clips/clip_001.wav --prompt "参考音频里说的原话"

# 6. 批量合成(自动起停引擎)
PYTHONPATH=src .venv\Scripts\python.exe -m echosmith.cli synth --voice myvoice --text-file lines.txt
```

GUI 方式:`.venv\Scripts\python main.py` 或下载 Release exe——功能与 CLI 一一对应,页面即流程。

## Agent Skill

本仓库附带 `skills/echosmith/SKILL.md`,拷贝(或链接)到 `~/.agents/skills/echosmith/` 即可作为 AI 助手技能使用:AI 会按「部署 → 数据集 → 校对 → 训练 → 声线卡 → 合成」的标准流程自动调用上述 CLI,并把进度与结果汇报给你。

## 设备支持

**CPU**(训练推荐)与 **NVIDIA CUDA 原生**(推理加速)开箱即用;引擎 torch 为 CUDA 构建,N 卡推理开箱即用。

> AMD 用户:曾支持 ZLUDA 转译(推理可用、训练为负收益),现已归档为实验路径,完整踩坑笔记与重部署步骤见 [`docs/zluda-notes.md`](docs/zluda-notes.md)。

## 开发

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt pytest
.venv\Scripts\python -m pytest tests/ -q      # 23 项契约测试
.venv\Scripts\python main.py                   # GUI
```

打包(PyInstaller onefile,引擎外置):`.venv\Scripts\python -m PyInstaller build.spec`。
推送 `v*` tag 触发 GitHub Actions 构建并挂 Release。

设计决策见 [`docs/adr/`](docs/adr/),领域术语见 [`CONTEXT.md`](CONTEXT.md)。

## License

GPL-3.0。GPT-SoVITS 为 MIT,本工具仅引导下载官方整合包并通过其脚本/进程交互,不内嵌再分发。
