# EchoSmith

> 声匠 —— GPT-SoVITS 声音克隆工作台（训练侧）。纯合成批量生产见姊妹项目 [EchoRunner](https://github.com/GreenChennai/EchoRunner)。

流水线：**上传视频/音频 → 提音轨 → 静音切分（<10s）→ FunASR 自动打标 → 训练清单（.list）**，产出的数据集直接喂给 GPT-SoVITS 微调（训练管理在 M3 迭代），产出的权重在 EchoRunner 中消费。

## 当前版本（v0.2.0）已实现

- **引擎首运行向导**：manifest 驱动下载官方整合包（v2pro/v4/v2 三档，HF 主源 + hf-mirror 镜像回退、断点续传、7z 解压、目录校验）；也支持导入本地 .7z 或直接指向已解压目录
- **素材流水线**：任意视频/音频 → ffmpeg 提 32k 单声道音轨 → **可选 UVR5 人声分离**（引擎内驱动，CPU）→ 静音检测切分（阈值/最短静音/片段长度可调）→ 逐片转写 → 生成 GPT-SoVITS 格式 `.list`
- **训练编排（M3）**：`prepare_datasets×3 → s2_train(SoVITS) → s1_train(GPT)` 五阶段子进程链，预训练权重按引擎版本自动探测（v2Pro/v2/v1），日志实时滚屏、遇错即停、可随时停止
- **声线卡一键生成（M3）**：扫描训练产物 → 绑定多套带情感标签的参考音频 → 产出 EchoRunner 兼容 `card.json`，与 [EchoRunner](https://github.com/GreenChennai/EchoRunner) 无缝衔接
- **打标**：优先接本地 FunASR 服务（MomentShift 服务模式 `http://127.0.0.1:8000/v1`，OpenAI 兼容）
- **设备后端兼容（v0.2.7）**：NVIDIA CUDA 原生 / **AMD 经 ZLUDA**（torch 仍走 CUDA 契约，引擎脚本零改动）/ CPU 自动兜底；设置页一键探测设备后端
- **遇错即停**：任一步失败立即停止并给出原因，不自动重试
- **干净落盘**：`models/datasets/<名称>/`（clips/ + <名称>.list）、声线卡落 `models/voices/<名>/card.json`，中间文件自动清理

> ⚠️ 训练编排与 UVR5 走的是整合包内部接口（webui 环境变量契约 / uvr5 lib）。**v0.2.1 起已逐行对照 GPT-SoVITS 官方源码核验**（prepare_datasets 环境变量契约含 i_part/all_parts 分片与合并、s2_train 的 JSON 配置契约、s1_train 的 YAML 配置契约、uvr5 `vr.py` 的 `_path_audio_(inp, ins_root, vocal_root, format, is_hp3)` 签名与 `vocal_<name>_<agg>.wav` 产物命名、v2Pro 的 sv 模型阶段）；引擎版本再演进时错误信息会原样透出便于适配。

## AMD 显卡 GPU 加速（ZLUDA）

N 卡开箱即用（整合包 torch 为 CUDA 构建）；AMD 卡经 [ZLUDA](https://github.com/lshqqytiger/ZLUDA) 转译跑同一条 CUDA 契约，**引擎脚本无需任何手动修改**——EchoSmith 会在引擎下载/导入后与每次训练前自动应用兼容补丁（幂等，`ZLUDA_MODE` 环境变量守卫，不开 GPU 时零副作用）：

- g2pw 文本前端强制走 CPU（ONNX Runtime 的 CUDA EP 在 ZLUDA 下必崩）
- 全部含卷积的引擎脚本（2-hubert / 2-sv / 3-semantic / s2 / s1 / TTS / UVR5）按 **MIOpen 可用性守卫 cuDNN**：官方 HIP SDK 不带 MIOpen.dll → 自动禁用 cuDNN 走 torch 原生卷积；换用 [TheRock nightly](https://therock-nightly-tarball.s3.amazonaws.com/index.html)（`therock-dist-windows-gfx120X-all-*.tar.gz`，RDNA4 对应 gfx120X）等带 MIOpen 的 ML 构建则自动恢复 cuDNN 加速
- `2-sv` 阶段官方脚本逐行 try/except 吞错，特征缺失会**静默产出废数据**——训练器在阶段后清点产物数量，缺了直接报错拦下

设置步骤：「设置」页 → 勾选「启用 ZLUDA GPU 加速」→ 填 ZLUDA 目录（含 `nvcuda.dll`）与 HIP SDK 目录（含 `bin\amdhip64_7.dll`）→ 保存 → 「探测设备后端」应显示 `✔ zluda：AMD Radeon ...[ZLUDA]`。

**本机实测边界（RX 9070 GRE / ZLUDA rel.854c58，诚实数据）**：GPU 对 prepare 特征提取与 TTS 推理收益明确；但 **s2/s1 训练仍建议走 CPU**——GPT-SoVITS 训练是小算子密集型负载，ZLUDA 的逐内核派发开销会吃满一核（GPU 利用率仅 ~20%），实测同数据集 CPU 训练 s2+s1 共 9 分钟、GPU 跑 10 小时未完成一个 epoch。cuDNN/MIOpen 加速路径（TheRock nightly）已验证 MIOpen 本体可用，但 torch cuDNN→垫片→MIOpen 转发链在当前 ZLUDA 构建上不可用（v8 API NOT_SUPPORTED，v7 陷入马拉松 JIT）。CPU 路径全部守卫零副作用，配置里关掉 `zluda_mode` 即回到该路径。

## 路线图

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M1 | 骨架 + 引擎下载器 | ✔ v0.1.0 |
| M2 | 素材流水线（切分+打标） | ✔ v0.1.0 |
| M3 | UVR5 人声分离 + 训练管理 + 声线卡生成 | ✔ v0.2.0（需真机联调） |
| M4 | 训练完成后引导至 EchoRunner 批量合成 | ✔ 走 card.json 衔接 |

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

「训练」页 → 刷新清单 → 选 .list → 实验名/轮数/批大小 → 开始训练。
五阶段自动跑完，产物在引擎 `logs/<实验名>/{SoVITS_weights_v2,GPT_weights_v2}/`。

「声线卡」页 → 填实验名 → 扫描权重 → 添加情感参考音频行 → 生成 card.json。
把声线卡放进 EchoRunner 的 `models/voices/`（或直接指向 EchoSmith 的 models/voices），
即可在 EchoRunner 中批量合成。

## 开发

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt pytest
.venv\Scripts\python -m pytest tests/ -q
.venv\Scripts\python main.py
```

打包（PyInstaller onefile，引擎外置）：`.venv\Scripts\python -m PyInstaller build.spec`。
推送 `v*` tag 触发 GitHub Actions 构建并挂 Release。

## License

GPL-3.0。GPT-SoVITS 为 MIT，本工具仅引导下载官方整合包并通过其脚本/进程交互，不内嵌再分发。
