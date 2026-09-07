# ZLUDA 部署笔记(RX 9070 GRE / RDNA4 实测全集)

> 状态:**已归档,默认不支持**(ADR-0002)。本文档保留完整知识,未来需要 AMD GPU
> 加速时按此重部署。实测结论先行:**训练场景 ZLUDA 为负收益**(派发开销吃满单核,
> GPU 利用率 ~20%,同任务 CPU 快 60 倍以上);**推理可用**(19~29 秒/句)但不优于
> CPU 的工程简单性。

## 环境组成(缺一不可)

1. **ZLUDA 运行时**(rel.854c58e 及以后,github.com/lshqqytiger/ZLUDA):
   `nvcuda.dll` `zluda.exe` 等。**必须确认包里有 `cufft.dll`**——torch 的
   `cufft64_10.dll` 垫片是个转发器,运行时要加载它;缺失/过旧则所有 FFT 调用报
   `CUFFT_NOT_SUPPORTED`。
2. **HIP SDK 便携提取**(如 ROCm 7.2,目录含 `bin\amdhip64_7.dll`、rocblas 等):
   AMD 官方 HIP SDK 页面下载,**官方 SDK 不带 MIOpen**。
3. 驱动:Adrenalin 最新版即可,无需 PRO 版。

## 部署步骤

1. `set HIP_PATH=<HIP SDK 目录>`;`PATH` 前置 `<ZLUDA 目录>;<HIP SDK>\bin`
2. `set ZLUDA_MODE=1`(EchoSmith 兼容补丁的总开关)
3. EchoSmith v0.2.x 的 `zluda_patch.py` 会自动:部署 torch/lib DLL 垫片(原版备份进
   `_zluda_nvidia_backup\`,状态在 `_zluda_shims_state.json`)、g2pw 走 CPU EP、
   按 MIOpen 守卫 cuDNN、fft/stft/复数 abs CPU 往返。还原:`restore_runtime_shims()`。

## 踩坑全集(按踩到的顺序)

| 坑 | 现象 | 原因 / 解法 |
|---|---|---|
| 官方 HIP SDK 无 MIOpen | `cuDNN error: MIOpen.dll could not be found` | cuDNN 转发依赖 MIOpen.dll;无则必须 `torch.backends.cudnn.enabled=False` 走原生卷积。带 MIOpen 的是 TheRock nightly(`therock-dist-windows-gfx120X-all-*.tar.gz`,RDNA4=gfx120X) |
| MIOpen 调通了也没用 | v8 API `CUDNN_STATUS_NOT_SUPPORTED`;v7 API 马拉松 JIT | torch cuDNN→ZLUDA 垫片→MIOpen 链路不通;且瓶颈本就不在卷积 |
| cuFFT 全类型 NOT_SUPPORTED | `torch.fft.rfft` 即挂 | ZLUDA 的 cuFFT 转译是空壳;须把 `cufft64_10.dll` 换成 ZLUDA 垫片(同名同 ABI,原版备份) |
| 垫片转发器缺实现 | 垫片在但报 NOT_SUPPORTED | `cufft64_10.dll` 依赖 `cufft.dll`(实现),ZLUDA 包曾缺此文件,从 release zip 补 |
| cuFFT 报错后进程不可杀 | taskkill 说"没有实例"但进程仍占 1 核 | CUDA 上下文损坏后清理死锁成僵尸,只能重启机器;文件被锁可用"重命名绕过" |
| 复数 abs 崩 NVRTC | `invalid value for --gpu-architecture` | CUDA 复数 abs 走 jiterator 现场 NVRTC 编内核,旧 NVRTC 不认 ZLUDA 架构——复数 abs 也要 CPU 往返 |
| 训练奇慢 | 33 步跑了 10 小时 | 根因:小算子密集负载的**逐内核派发开销**(CPU 100%,GPU 20%),fp32+无 cuDNN 放大。训练请用 CPU |
| 排除目录后引擎起不来 | `No module named 'datasets1'` | `api_v2.py` 无条件 import `tools/audio_sr.py` → 依赖 `tools/AP_BWE_main\datasets1`;裁剪整合包不能排除 AP_BWE_main |
| VAD/ASR 在 UVR5 人声轨上质量崩 | 转写满屏错字 | 人声分离处理伤 ASR;应"原始音轨跑 VAD/转写,时间戳映射到人声轨切分" |

## UVR5 on ZLUDA(已验证可用)

27.6 分钟音轨仅需 8 分钟(GPU)vs CPU 约 1 小时。驱动模板见
`scripts/experimental/_uvr5_gpu_driver.py`(device=cuda,ZLUDA 环境变量注入后运行)。

## 性能结论备忘

| 场景 | ZLUDA GPU | CPU | 结论 |
|---|---|---|---|
| prepare 特征提取 | 快(BERT/hubert 大算子) | 慢一些 | GPU 有收益 |
| s2/s1 训练 | **10h+/epoch 组** | **9 分钟全程** | 用 CPU |
| TTS 推理 | 19~29 秒/句(热) | 20~60 秒/句 | GPU 可用 |
