# CONTEXT.md — 术语表

声匠家族(GPT-SoVITS 声音克隆工具链)的领域术语。只放词汇,不放实现细节。

## 术语

- **引擎(Engine)**:GPT-SoVITS 官方 Windows 整合包解压后的目录(含 `api_v2.py` 与 `runtime\python.exe`)。一切训练与合成都是引擎子进程;本工具只编排,不内嵌。
- **声线卡(Voice Card)**:`card.json`,一条虚拟声线的完整描述——GPT/SoVITS 权重路径 + 多套带情感标签的参考音频 + 参考文本。训练产物经它才能被合成消费。
- **声线卡生成(Card Build)**:扫描指定实验名的训练产物,绑定参考音频(3~10 秒硬校验),产出声线卡。
- **六阶段训练**:引擎的微调流程——`1-text → 2-hubert → 2-sv → 3-semantic → s2(SoVITS) → s1(GPT)`。前四阶段统称 **prepare**。
- **实验名(Experiment)**:一次训练的标识,决定权重文件名前缀与 `logs/<实验名>/` 目录。
- **打标(Transcription)**:为切片生成转写文本。路径有二:本地 FunASR 服务(OpenAI 兼容 `/v1/audio/transcriptions`)、引擎内置 funasr 离线批量。
- **合成队列(Synthesis Queue)**:多行文本按行顺序合成的任务模型;行首 `【情感】` 前缀覆盖单行情感;遇错即停,不自动重试。
- **设备后端(Device Backend)**:训练/推理所在算力,现为 **cpu** 与 **cuda**(NVIDIA 原生)两种。AMD 经 ZLUDA 属已归档的实验路径(见 docs/zluda-notes.md)。
- **辅助模式(Auxiliary Mode)**:GUI 的定位——人工操作与观察用;自动化主路径是 `echosmith` Skill 与 `python -m echosmith.cli`。
