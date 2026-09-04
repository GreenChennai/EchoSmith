"""静音检测 + 训练片段切分（目标 <10s，适配 GPT-SoVITS 训练集）。"""
from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass

from .config import Config
from .ffmpeg_util import WIN_SILENT, FFmpegError, find_ffmpeg

_RE_START = re.compile(r"silence_start:\s*([0-9.]+)")
_RE_END = re.compile(r"silence_end:\s*([0-9.]+)")


@dataclass
class Segment:
    start: float
    end: float

    @property
    def dur(self) -> float:
        return self.end - self.start


def detect_silences(cfg: Config, wav) -> list[Segment]:
    """返回静音区间列表 [(start, end)]。"""
    cmd = [
        find_ffmpeg(cfg), "-i", str(wav), "-af",
        f"silencedetect=noise={float(cfg['silence_db'])}dB:d={float(cfg['silence_min_dur'])}",
        "-f", "null", "-",
    ]
    r = subprocess.run(cmd, capture_output=True, creationflags=WIN_SILENT, timeout=3600)
    text = r.stderr.decode("utf-8", "replace")
    silences: list[Segment] = []
    cur_start: float | None = None
    for line in text.splitlines():
        m = _RE_START.search(line)
        if m:
            cur_start = float(m.group(1))
            continue
        m = _RE_END.search(line)
        if m and cur_start is not None:
            silences.append(Segment(cur_start, float(m.group(1))))
            cur_start = None
    if cur_start is not None:
        silences.append(Segment(cur_start, cur_start))  # 末尾未闭合静音
    return silences


def build_segments(silences: list[Segment], total_dur: float, min_ms: int, max_ms: int) -> list[Segment]:
    """静音之间 = 语音段；超长段按 max_ms 均分；过短段丢弃。"""
    speech: list[Segment] = []
    cursor = 0.0
    for s in silences:
        if s.start > cursor:
            speech.append(Segment(cursor, s.start))
        cursor = max(cursor, s.end)
    if cursor < total_dur:
        speech.append(Segment(cursor, total_dur))

    max_s = max_ms / 1000.0
    min_s = min_ms / 1000.0
    out: list[Segment] = []
    for seg in speech:
        if seg.dur < min_s:
            continue
        if seg.dur <= max_s:
            out.append(seg)
            continue
        n = int(seg.dur / max_s) + (1 if seg.dur % max_s > 0 else 0)
        step = seg.dur / n
        for i in range(n):
            out.append(Segment(seg.start + i * step, seg.start + (i + 1) * step))
    return out
