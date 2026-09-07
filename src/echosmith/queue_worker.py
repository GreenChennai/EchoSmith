"""批量合成队列：单线程顺序消费，遇错即停（符合流水线策略：不自动重试）。"""
from __future__ import annotations

import datetime as _dt
import re
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path

from .config import Config
from .engine import Engine, EngineError
from .state import Shared
from .tts_client import TTSClient, TTSClientError
from .voice_card import VoiceCard

WIN_SILENT = 0x08000000 if sys.platform == "win32" else 0

# 行首情感覆盖语法：【开心】今天天气不错。
_EMOTION_RE = re.compile(r"^【\s*(.+?)\s*】\s*")


class Job:
    def __init__(
        self,
        card: VoiceCard,
        emotion: str,
        lines: list[str],
        text_lang: str,
        speed_factor: float,
        split_method: str,
    ) -> None:
        self.card = card
        self.emotion = emotion
        self.lines = [ln.strip() for ln in lines if ln.strip()]
        self.text_lang = text_lang
        self.speed_factor = speed_factor
        self.split_method = split_method

    @property
    def name(self) -> str:
        return f"{self.card.name}·{self.emotion}×{len(self.lines)}"


class Worker(threading.Thread):
    def __init__(self, cfg: Config, engine: Engine, shared: Shared) -> None:
        super().__init__(name="echorunner-worker", daemon=True)
        self.cfg = cfg
        self.engine = engine
        self.shared = shared
        self._jobs: deque[Job] = deque()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._resume = threading.Event()
        self._resume.set()  # 默认不暂停
        self.stopped = threading.Event()

    # -- 队列操作（UI 线程调用） ------------------------------------------------
    def submit(self, job: Job) -> int:
        with self._lock:
            self._jobs.append(job)
            n = len(self._jobs)
        self.shared.set_queued(n)
        self._wake.set()
        return n

    def clear(self) -> int:
        with self._lock:
            n = len(self._jobs)
            self._jobs.clear()
        self.shared.set_queued(0)
        self.shared.set_status("空闲")
        return n

    def pause(self) -> None:
        self._resume.clear()
        self.shared.set_status("已暂停")

    def resume(self) -> None:
        self._resume.set()

    def is_paused(self) -> bool:
        return not self._resume.is_set()

    # -- 主循环 ------------------------------------------------------------------
    def run(self) -> None:
        while not self.stopped.is_set():
            self._wake.wait()
            if self.stopped.is_set():
                break
            with self._lock:
                job = self._jobs[0] if self._jobs else None
            if job is None:
                self._wake.clear()
                continue
            self._resume.wait()
            if self.stopped.is_set():
                break
            try:
                self._run_job(job)
            except (TTSClientError, EngineError, OSError) as e:
                # 遇错停止：清空剩余队列，等待用户处理后再继续
                self.clear()
                self.shared.set_status("因错误停止", str(e))
                self.shared.log(f"[队列] 错误：{e}")
                self._pause_for_error()
                continue
            except Exception as e:  # 兜底：意外异常不得静默杀死队列线程
                self.clear()
                self.shared.set_status("因错误停止", f"{type(e).__name__}: {e}")
                self.shared.log(f"[队列] 未预期异常：{type(e).__name__}: {e}")
                self._pause_for_error()
                continue
            with self._lock:
                # 运行中用户可能已「清空队列」——当前 job 已不在队列时不得强弹
                if self._jobs and self._jobs[0] is job:
                    self._jobs.popleft()
                n = len(self._jobs)
            self.shared.set_queued(n)
            if n == 0:
                self._wake.clear()
                self.shared.set_status("空闲")
            else:
                self.shared.set_status("等待中…")

    def _pause_for_error(self) -> None:
        self._resume.clear()
        self._resume.wait()  # 用户点「继续」后恢复

    # -- 单个任务 -------------------------------------------------------------------
    def _run_job(self, job: Job) -> None:
        if not job.lines:
            return
        self.shared.set_status("准备引擎…")
        if not self.engine.port_open():
            self.engine.start()
        client = TTSClient(self.cfg.engine_url, read_timeout=int(self.cfg["tts_read_timeout"] or 600))
        try:
            self._synthesize(job, client)
        finally:
            client.session.close()

    def _synthesize(self, job: Job, client: TTSClient) -> None:
        # 预热：切权重（首次调用时）
        self.shared.set_status("切换声线权重…")
        client.set_voice(job.card, force=False)

        day = _dt.date.today().isoformat()
        out_root = self.cfg.output_path / job.card.name / day
        out_root.mkdir(parents=True, exist_ok=True)
        self.shared.set_last_output_dir(str(out_root))

        total = len(job.lines)
        self.shared.set_progress(job.name, 0, total)
        self.shared.log(f"[队列] 开始任务 {job.name}（{total} 条）→ {out_root}")

        n = _next_free_index(out_root, job.card.name)
        for i, line in enumerate(job.lines, 1):
            self._resume.wait()
            if self.stopped.is_set():
                return
            emo_label, text = _split_emotion(line, job.emotion, job.card)
            emo = job.card.pick_emotion(emo_label)
            stem = f"{_safe(job.card.name)}_{_safe(emo.label)}_{n:03d}"
            out_wav = out_root / f"{stem}.wav"
            self.shared.set_progress(job.name, i, total)
            self.shared.set_status(f"合成中 {i}/{total}")
            client.tts_to_file(
                out_wav,
                text=text,
                text_lang=job.text_lang,
                ref_audio_path=emo.ref,
                prompt_text=emo.prompt_text,
                prompt_lang="zh",
                text_split_method=job.split_method,
                speed_factor=job.speed_factor,
                aux_ref_audio_paths=emo.aux_refs,
            )
            n += 1
            if self.cfg.data["save_mp3"]:
                _to_mp3(out_wav, self.cfg.data["ffmpeg_path"])
            self.shared.log(f"[队列] ✔ {out_wav.name}")
        self.shared.set_progress(job.name, total, total)
        self.shared.set_status("任务完成")
        self.shared.log(f"[队列] 任务完成：{job.name}")


# -- 工具 ------------------------------------------------------------------------

def _split_emotion(line: str, default: str, card: VoiceCard) -> tuple[str, str]:
    m = _EMOTION_RE.match(line)
    if m and m.group(1) in card.emotions:
        return m.group(1), line[m.end():]
    return default, line


def _safe(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", name).strip() or "voice"


def _next_free_index(out_root: Path, voice: str) -> int:
    n = 1
    while any(out_root.glob(f"{_safe(voice)}_*_{n:03d}.*")):
        n += 1
    return n


def _to_mp3(wav: Path, ffmpeg_path: str) -> None:
    ffmpeg = ffmpeg_path.strip()
    if not ffmpeg:
        return  # 未配置则静默跳过
    mp3 = wav.with_suffix(".mp3")
    try:
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(wav),
             "-codec:a", "libmp3lame", "-q:a", "2", str(mp3)],
            creationflags=WIN_SILENT, timeout=300, check=True,
        )
    except (subprocess.SubprocessError, OSError) as e:
        raise TTSClientError(f"mp3 转码失败：{e}") from e
