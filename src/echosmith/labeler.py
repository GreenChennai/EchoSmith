"""打标：本地 FunASR 服务（MomentShift 服务模式，OpenAI 兼容 /v1/audio/transcriptions）。

策略：优先本地 FunASR 服务（逐条、实时进度）；服务不可达时回退
引擎整合包自带的 funasr（tools/asr/funasr_asr.py，一次子进程批量转写，
模型只加载一次，离线可用——模型随整合包预置）。
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import requests

from .config import Config


class LabelerError(RuntimeError):
    pass


def check_service(cfg: Config, timeout: float = 3.0) -> bool:
    """探活 FunASR 服务（GET /health 挂在根路径）。"""
    base = str(cfg["funasr_url"]).rstrip("/")
    root = base[: -len("/v1")] if base.endswith("/v1") else base
    try:
        r = requests.get(f"{root}/health", timeout=timeout)
        return r.status_code == 200
    except requests.RequestException:
        return False


def transcribe(cfg: Config, wav: Path) -> str:
    base = str(cfg["funasr_url"]).rstrip("/")
    url = f"{base}/audio/transcriptions"
    try:
        with wav.open("rb") as f:
            r = requests.post(
                url,
                files={"file": (wav.name, f, "audio/wav")},
                data={"model": str(cfg["funasr_model"])},
                timeout=(5, 300),
            )
    except requests.RequestException as e:
        raise LabelerError(f"FunASR 请求失败：{e}") from e
    if r.status_code != 200:
        raise LabelerError(f"FunASR 返回 {r.status_code}: {r.text[:200]}")
    data = r.json()
    text = str(data.get("text", "")).strip()
    if not text:
        raise LabelerError(f"{wav.name} 转写结果为空")
    return text


def transcribe_engine_batch(cfg: Config, clips: list[Path], lang: str = "zh") -> dict[Path, str]:
    """回退：引擎整合包自带 funasr 批量转写（一次子进程，模型只加载一次）。

    clips 须在同一目录（流水线切片本就如此）。返回 {clip绝对路径: 文本}；
    子进程失败、无产出或个别切片缺文本由调用方校验。
    """
    if not clips:
        return {}
    engine = cfg.engine_path
    script = engine / "tools" / "asr" / "funasr_asr.py"
    python = engine / "runtime" / "python.exe"
    if not script.is_file() or not python.is_file():
        raise LabelerError(f"引擎缺少内置 funasr 组件（{script} / {python}）")
    clips_dir = clips[0].resolve().parent
    if any(c.resolve().parent != clips_dir for c in clips):
        raise LabelerError("引擎批量转写要求所有切片位于同一目录")
    # 引擎 funasr_asr 的 -l 只认 zh/yue/auto；EchoSmith 配置里是大写 ZH
    lang = lang.lower() if lang else "zh"
    if lang not in ("zh", "yue", "auto"):
        lang = "zh"
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    with tempfile.TemporaryDirectory(prefix="echosmith_asr_") as td:
        cmd = [
            str(python), "tools/asr/funasr_asr.py",
            "-i", str(clips_dir), "-o", td, "-l", lang,
        ]
        try:
            proc = subprocess.run(
                cmd, cwd=str(engine), capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                env=env, timeout=3600,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            raise LabelerError(f"引擎 funasr 子进程失败：{e}") from e
        list_file = Path(td) / f"{clips_dir.name}.list"
        if not list_file.is_file():
            tail = ((proc.stderr or "") + "\n" + (proc.stdout or ""))[-400:]
            raise LabelerError(f"引擎 funasr 未产出标注文件（返回码 {proc.returncode}）\n{tail}")
        texts: dict[Path, str] = {}
        for line in list_file.read_text("utf-8").splitlines():
            parts = line.split("|", 3)
            if len(parts) != 4:
                continue
            wav_path, _speaker, _lang, text = parts
            texts[Path(wav_path).resolve()] = text.strip()
        if not texts:
            raise LabelerError("引擎 funasr 转写结果为空")
        return texts


def write_list(entries: list[tuple[Path, str]], out_list: Path, speaker: str, lang: str) -> Path:
    """写 GPT-SoVITS .list：vocal_path|speaker_name|language|text（绝对路径）。"""
    out_list.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for wav, text in entries:
        text = text.replace("|", "/").replace("\n", " ").strip()
        lines.append(f"{wav.resolve().as_posix()}|{speaker}|{lang}|{text}")
    out_list.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_list
