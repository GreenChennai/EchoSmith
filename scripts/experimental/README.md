# experimental(未产品化脚本)

来自 zluda_test 的实机验证脚本,归档备查:

- `_jimi_rebuild.py` — 句子级数据集管线 v2:fsmn-vad 语音区间(毫秒)→ >1.2s 静音断组、≤10s 合并 → RAW 转写/VOCAL 切分。比产品 pipeline(静音切分+服务打标)质量更好,待产品化。
- `_uvr5_gpu_driver.py` — UVR5 人声分离驱动(device 可选 cuda),仅 ZLUDA 环境可用,见 docs/zluda-notes.md。
