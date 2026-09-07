---
name: echosmith
description: 声匠 EchoSmith 自动化——GPT-SoVITS 声音克隆全链路。当用户想要:部署 GPT-SoVITS 整合包环境、用音频素材训练声音克隆模型(六阶段微调)、生成声线卡、批量合成语音/配音、检查训练或引擎状态时使用。CPU/CUDA 原生,无需 GUI。
---

# EchoSmith — GPT-SoVITS 声音克隆自动化

把"一段音频素材"变成"一个可批量合成的声线模型"。所有操作通过本仓库的 CLI 完成,
遇错即停(退出码非 0),日志实时走 stdout。

## 先决条件(一次性,通常已就绪)

- 仓库:`E:\平日资料\GitHub\EchoSmith`;依赖已装于 `.venv`(requirements.txt)
- 引擎:GPT-SoVITS 整合包(未部署时用 `deploy-engine`,约 2~3GB 下载)
- 命令统一形式(在仓库根目录执行):

```bash
cd "E:\平日资料\GitHub\EchoSmith"
PYTHONPATH=src .venv/Scripts/python.exe -m echosmith.cli <子命令> [参数]
```

`--engine-dir <路径>` 可放在子命令前覆盖引擎目录(默认读 `config/echosmith.json`)。

## 命令速查

| 目的 | 命令 |
|---|---|
| 部署引擎(下载官方整合包+校验+兼容补丁) | `cli deploy-engine`(加 `--mirror` 走镜像;`--local <.7z/目录>` 本地导入) |
| 素材 → 训练数据集(切分+打标) | `cli dataset --src <视频/音频> --name <数据集名> [--separate]` |
| 六阶段训练 | `cli train --list models/datasets/<名>/<名>.list --exp <实验名> --s2 8 --s1 15 --device cpu` |
| 只跑部分阶段 | `cli train ... --stages 4-sovits,5-gpt` |
| 生成声线卡 | `cli card --exp <实验名> --voice <声线名> --ref <wav> --prompt "<参考音频原话>"` |
| 批量合成 | `cli synth --voice <声线名> --text-file <txt>(每行一条)` |
| 探测引擎 torch 可用性 | `cli probe` |

## 标准流程(新声线从零到出声)

1. `deploy-engine`(一次性;已部署会校验并跳过下载吗——不会,已部署请跳过本步)
2. `dataset --src <素材> --name <名>`:产出 `models/datasets/<名>/clips/*.wav` + `<名>.list`
3. **检查/修正 `<名>.list` 文本**(ASR 转写可能有错;文本=音频逐字对应才训练得好)
4. `train --list ... --exp <名> --device cpu`:六阶段;CPU 上 20 分钟素材约 3 小时
   (s2 ~10 分钟/epoch、s1 ~5 分钟/epoch;权重落引擎 `SoVITS/GPT_weights_v2ProPlus`)
5. `card --exp <名> --voice <名> --ref <clip> --prompt "<那句原话>"`(参考音频 3~10 秒)
6. `synth --voice <名> --text-file <txt>`:自动起停引擎,产物落 `output/<声线>/<日期>/`

## 关键事实与注意

- **设备**:默认 CPU 与 NVIDIA CUDA 原生。训练**用 CPU**(实测远快于 GPU 转译方案);
  `--device cuda` 交由 torch 探测 N 卡。AMD/ZLUDA 属已归档实验路径,见 `docs/zluda-notes.md`
- **合成需引擎在线**:`synth` 会自动启动/停止引擎(端口默认 9885,被占用用 `--port` 换)
- **GPU 首句慢**:CUDA 首次合成要编译内核,首句可能 10 分钟+;`tts_read_timeout` 已配 1800
- **遇错即停**:任何失败立即退出并打印原因,把 stderr/stdout 原样转述给用户即可
- **训练时长预估**:先按 `prepare 20min + s2 8×10min + s1 15×5min` 报给用户,再等其确认
- 打标依赖:本地 FunASR 服务(`config: funasr_url`)或引擎内置 funasr 自动回退,均离线可用
- 训练日志:stdout 实时;引擎日志在 GUI「日志」页或引擎进程输出
- 实验性:句子级数据集管线与 ZLUDA UVR5 驱动在 `scripts/experimental/`(未产品化)
