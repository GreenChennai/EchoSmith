# ADR-0001 — 合并 EchoRunner 进 EchoSmith:软件转辅助,Skill 为主路径

日期:2026-09-07 · 状态:已接受

## 背景

声匠家族原有两个仓库:**EchoSmith**(训练侧:引擎向导/素材流水线/六阶段训练/声线卡生成)与
**EchoRunner**(合成侧:引擎托管/合成队列/声线卡消费)。两者共享同一个 GPT-SoVITS 引擎与
声线卡格式,功能互补但分仓导致:格式定义两份、兼容补丁两份、发布流程两份、用户认知两份。

同时,实际使用方式已经迁移:Jimi 模型的完整生产链(素材→打标→训练→声线卡→合成)全部由
AI 助手以脚本驱动完成,GUI 只在人工抽查时打开。

## 决策

1. **EchoRunner 的全部能力并入 EchoSmith**:引擎托管(`engine.py`)、TTS 客户端
   (`tts_client.py`)、合成队列(`queue_worker.py`)、声线卡消费(`voice_card.py`),
   GUI 增加合成台与队列两个页。
2. **软件定位降为辅助模式**:GUI 保留全功能,服务人工操作与观察;自动化主路径是
   **`echosmith` Skill**(Agent 调用)→ `python -m echosmith.cli`
   (deploy-engine / dataset / train / card / synth / probe)。
3. **删除 EchoRunner**:远程仓库与本地目录一并删除;构建目录中的
   `EchoRunner-v0.1.7` 便携包为独立交付物,保留(exe 自包含,不受仓库删除影响)。
4. CLI 不得依赖 GUI 库(dearpygui 仅允许在 `ui/` 内导入)。

## 理由

- 单仓库消除双份格式/补丁/发布;声线卡与兼容补丁的定义只剩一处(见 ADR-0002 的设备决策同理)。
- 真实使用数据(Jimi 全链路)证明 AI+脚本路径的完成度;GUI 的差异化价值在人工验收,
  即"辅助模式"。
- Skill 以仓库 `skills/echosmith/` 为源、安装到用户级 `~/.agents/skills/`,任意会话可调用。

## 后果

- EchoRunner 的 GitHub 发布/历史随仓库删除消失(其 exe 已有便携交付物)。
- EchoSmith README、CI、测试合并后统一维护;版本升到 v0.3.0。
