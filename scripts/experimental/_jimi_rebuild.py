# -*- coding: utf-8 -*-
"""Jimi 数据集重建 v5(纯离线,单位已校准):
VAD(原始音轨 16k)→ 完整句分组(>1.2s 静音断组,跨度≤10s)→
按组时间戳:RAW 切片转写(文本)+ VOCAL 切片(训练音频)→ punc → Jimi.list
"""
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import soundfile as sf

ENG = Path(r"E:\平日资料\GitHub\EchoSmith\models\GPT-SoVITS")
ASR_MODELS = ENG / "tools" / "asr" / "models"
FFDIR = Path(r"E:\平日资料\构建\MomentShift-v0.9.2-ui-redesign")
FFMPEG = FFDIR / "ffmpeg.exe"
FFPROBE = FFDIR / "ffprobe.exe"
VOCAL = sorted(Path(r"E:\zluda_test\uvr5_v").glob("*vocal*.wav"))[-1]
RAW = Path(r"E:\平日资料\声音训练-基米\_jimi_source.wav")  # ASR/VAD 用原始音轨,切分用 VOCAL
DATASET = Path(r"E:\平日资料\GitHub\EchoSmith\models\datasets\Jimi")
SPEAKER, LANG = "speaker0", "ZH"

MAX_CLIP = 10.0
GAP_BREAK = 1.2
PAD = 0.15

logf = open(r"E:\zluda_test\jimi_rebuild.log", "w", encoding="utf-8")


def log(msg: str) -> None:
    print(msg, flush=True)
    logf.write(msg + "\n")
    logf.flush()


def run(cmd, timeout=600):
    return subprocess.run(cmd, capture_output=True, timeout=timeout)


def cut(src: Path, cs: float, ce: float, out: Path, ar: int) -> bool:
    r = run([str(FFMPEG), "-y", "-loglevel", "error",
             "-ss", f"{cs:.3f}", "-to", f"{ce:.3f}", "-i", str(src),
             "-ac", "1", "-ar", str(ar), "-c:a", "pcm_s16le", str(out)])
    return r.returncode == 0


from funasr import AutoModel  # noqa: E402

t0 = time.time()
log("加载 funasr 模型…")
vad = AutoModel(model=str(ASR_MODELS / "speech_fsmn_vad_zh-cn-16k-common-pytorch"),
                disable_update=True, disable_pbar=True)
asr = AutoModel(model=str(ASR_MODELS / "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"),
                disable_update=True, disable_pbar=True)
punc = AutoModel(model=str(ASR_MODELS / "punc_ct-transformer_zh-cn-common-vocab272727-pytorch"),
                 disable_update=True, disable_pbar=True)
log(f"模型就绪 {time.time()-t0:.0f}s")

# ---- 1. VAD(原始音轨 16k),单位:毫秒 ----
proxy16 = Path(tempfile.gettempdir()) / "_jimi_raw16.wav"
cut(RAW, 0, 9999, proxy16, 16000)  # 全量重采样(-ss 0 -to 9999 覆盖全文件)
res = vad.generate(input=str(proxy16), vad_kwargs={"max_single_segment_time": 9000})
regions_ms = res[0]["value"]
total_s = len(sf.read(str(proxy16), dtype="float32")[0]) / 16000
log(f"VAD {len(regions_ms)} 段,语音合计 {sum(b-a for a, b in regions_ms)/1000/60:.1f} min / 总长 {total_s/60:.1f} min")

# ---- 2. 分组:毫秒内直接判跨度/间隙(阈值换算为 ms) ----
MAX_MS = MAX_CLIP * 1000
GAP_MS = GAP_BREAK * 1000
groups = []
cur = [regions_ms[0]]
for rg in regions_ms[1:]:
    span = rg[1] - cur[0][0]
    gap = rg[0] - cur[-1][1]
    if span > MAX_MS or gap > GAP_MS:
        groups.append(cur)
        cur = [rg]
    else:
        cur.append(rg)
groups.append(cur)
log(f"分组 {len(groups)} 组")

# ---- 3. 逐组:RAW 切片转写 + VOCAL 切片为训练音频 ----
clips_dir = DATASET / "clips"
for old in clips_dir.glob("clip_*.wav"):
    old.unlink()

tmp = Path(tempfile.gettempdir()) / "_jimi_groups"
tmp.mkdir(exist_ok=True)
entries = []
idx = 0
t0 = time.time()
for gi, grp in enumerate(groups):
    cs = max(0.0, grp[0][0] / 1000 - PAD)
    ce = min(total_s, grp[-1][1] / 1000 + PAD)
    if ce - cs < 1.2:
        continue
    # RAW 16k 切片 → 转写
    raw_seg = tmp / f"g{gi:04d}.wav"
    if not cut(RAW, cs, ce, raw_seg, 16000):
        continue
    rs = asr.generate(input=str(raw_seg))
    text = str(rs[0].get("text", "")).strip()
    raw_seg.unlink(missing_ok=True)
    if len(text) < 2:
        continue
    try:
        pr = punc.generate(input=text)
        text = str(pr[0].get("text", text)).strip()
    except Exception:
        pass
    text = text.replace("|", "/").replace("\n", " ").strip()
    # VOCAL 32k 切片 → 训练音频
    idx += 1
    out = clips_dir / f"clip_{idx:03d}.wav"
    if not cut(VOCAL, cs, ce, out, 32000):
        log(f"VOCAL 切片失败 {out.name}")
        idx -= 1
        continue
    entries.append((out.resolve(), text))
    if (gi + 1) % 20 == 0:
        log(f"  {gi+1}/{len(groups)} 组,有效 {len(entries)},{time.time()-t0:.0f}s")

# ---- 4. 写清单 ----
lines = [f"{w.as_posix()}|{SPEAKER}|{LANG}|{t}" for w, t in entries]
(DATASET / "Jimi.list").write_text("\n".join(lines) + "\n", encoding="utf-8")
dur_total = 0.0
for w, _t in entries:
    r = run([str(FFPROBE), "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(w)])
    try:
        dur_total += float(r.stdout.decode().strip())
    except ValueError:
        pass
log(f"完成:{len(entries)} 条片段,合计 {dur_total/60:.1f} 分钟 → Jimi.list")
for w, t in entries[:3]:
    log(f"  样例 {w.name}: {t[:60]}")
logf.close()
