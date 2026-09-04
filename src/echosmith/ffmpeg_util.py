"""ffmpeg / ffprobe 探测与音频提取。"""
from __future__ import annotations

import subprocess
import sys
import shutil
from pathlib import Path

from .config import Config

WIN_SILENT = 0x08000000 if sys.platform == "win32" else 0


class FFmpegError(RuntimeError):
    pass


_MOMENTSHIFT_HINTS = [
    r"E:\平日资料\GitHub\MomentShift\tools\ffmpeg_bin",
]


def find_ffmpeg(cfg: Config) -> str:
    """配置 → PATH → MomentShift 内置。返回可执行文件路径，找不到抛错。"""
    p = str(cfg["ffmpeg_path"] or "").strip()
    if p and Path(p).is_file():
        return p
    which = shutil.which("ffmpeg")
    if which:
        return which
    for hint in _MOMENTSHIFT_HINTS:
        cand = Path(hint) / "ffmpeg.exe"
        if cand.is_file():
            return str(cand)
    raise FFmpegError("找不到 ffmpeg：请在设置页指定路径（MomentShift 内置构建也可）")


def find_ffprobe(cfg: Config) -> str:
    p = str(cfg["ffprobe_path"] or "").strip()
    if p and Path(p).is_file():
        return p
    ffmpeg = Path(find_ffmpeg(cfg))
    cand = ffmpeg.with_name("ffprobe.exe")
    if cand.is_file():
        return str(cand)
    which = shutil.which("ffprobe")
    if which:
        return which
    cand2 = ffmpeg.with_name("ffprobe") if (ffmpeg.with_name("ffprobe")).is_file() else None
    if cand2:
        return str(cand2)
    raise FFmpegError("找不到 ffprobe（应与 ffmpeg 同目录）")


def probe_duration(cfg: Config, media: Path) -> float:
    ffprobe = find_ffprobe(cfg)
    r = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(media)],
        capture_output=True, creationflags=WIN_SILENT, timeout=60,
    )
    try:
        return float(r.stdout.decode("utf-8", "replace").strip())
    except ValueError as e:
        raise FFmpegError(f"读取时长失败：{r.stderr.decode('utf-8', 'replace')[:200]}") from e


def extract_audio(cfg: Config, src: Path, out_wav: Path) -> Path:
    """任意视频/音频 → 32k 单声道 wav（GPT-SoVITS 训练友好格式）。"""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        find_ffmpeg(cfg), "-y", "-loglevel", "error", "-i", str(src),
        "-vn", "-ac", "1", "-ar", "32000", str(out_wav),
    ]
    r = subprocess.run(cmd, capture_output=True, creationflags=WIN_SILENT, timeout=3600)
    if r.returncode != 0:
        raise FFmpegError(f"提取音轨失败：{r.stderr.decode('utf-8', 'replace')[:300]}")
    return out_wav


def cut_clip(cfg: Config, src: Path, start_s: float, end_s: float, out: Path) -> Path:
    """按起止时间切出一段（重编码 pcm 保证精度）。"""
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        find_ffmpeg(cfg), "-y", "-loglevel", "error",
        "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}", "-i", str(src),
        "-c:a", "pcm_s16le", str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, creationflags=WIN_SILENT, timeout=300)
    if r.returncode != 0:
        raise FFmpegError(f"切片失败 {out.name}：{r.stderr.decode('utf-8', 'replace')[:200]}")
    return out
