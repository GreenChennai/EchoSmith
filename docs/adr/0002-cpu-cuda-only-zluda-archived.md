# ADR-0002 — 设备支持收缩为 CPU + CUDA,ZLUDA 转入归档

日期:2026-09-07 · 状态:已接受

## 背景

为在 AMD 显卡(RX 9070 GRE / RDNA4)上复用 CUDA 版整合包,曾深度接入 ZLUDA:
兼容补丁(g2pw CPU EP、cuDNN 按 MIOpen 守卫、fft/复数 abs CPU 往返、2-sv 产物守卫)、
torch/lib DLL 垫片部署/还原、GUI 设置项与 CLI 支持。真机结论(2026-09-06):

- **训练**:ZLUDA 逐内核派发开销吃满单核(GPU 利用率 ~20%),同数据集 CPU 9 分钟 vs
  GPU 10+ 小时未完成一个 epoch——训练场景 ZLUDA 为负收益。
- **推理**:可用(19~29 秒/句),但并不优于 CPU 的工程简单性,且依赖脆弱
  (cuFFT 转译缺失、垫片 DLL 版本配对、MIOpen 链不可用,详见 docs/zluda-notes.md)。

## 决策

1. 设备支持收缩为 **cpu** 与 **cuda**(NVIDIA 原生)两种;CLI `--device` 仅接受这两者,
   `cpu` 通过 `CUDA_VISIBLE_DEVICES=""` 强制,`cuda` 交由 torch 自行探测。
2. GUI 与 Skill 文档移除 ZLUDA 入口;`zluda_patch.py` 模块**保留在仓库**
   (全部补丁由 `ZLUDA_MODE` 环境变量守卫,不启用时零副作用,常驻无成本)。
3. 本地 `E:\zluda_test\`(ZLUDA 运行时、HIP SDK、TheRock 13GB、测试脚本与日志)删除;
   有复用价值的两个脚本(句子级数据集管线 `_jimi_rebuild.py`、UVR5 驱动模板
   `_uvr5_gpu_driver.py`)归档至 `scripts/experimental/`。
4. 全部踩坑知识沉淀至 `docs/zluda-notes.md`,含未来重新部署的完整步骤。

## 理由

默认路径的维护面 = 用户实际使用的路径。ZLUDA 的三条价值(AMD 训练/AMD 推理/知识)
中,前两条被实测否定,第三条由文档满足;而保留默认支持要持续支付双份补丁、双份测试、
双份用户引导的成本。

## 后果

- AMD 用户短期失去开箱 GPU 路径;重新启用 = 按 zluda-notes.md 手动部署 + 环境变量
  `ZLUDA_MODE=1`,代码路径仍在。
- 依赖 ZLUDA 的既有配置文件字段(`zluda_mode` 等)从 UI/CLI 移除,配置文件中的
  残留键被忽略,不迁移。
