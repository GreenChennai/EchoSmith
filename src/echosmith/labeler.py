"""打标：本地 FunASR 服务（MomentShift 服务模式，OpenAI 兼容 /v1/audio/transcriptions）。

策略：优先本地 FunASR 服务；不可用时提示用户（回退引擎内置 funasr 属 M3 内容）。
"""
from __future__ import annotations

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


def write_list(entries: list[tuple[Path, str]], out_list: Path, speaker: str, lang: str) -> Path:
    """写 GPT-SoVITS .list：vocal_path|speaker_name|language|text（绝对路径）。"""
    out_list.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for wav, text in entries:
        text = text.replace("|", "/").replace("\n", " ").strip()
        lines.append(f"{wav.resolve().as_posix()}|{speaker}|{lang}|{text}")
    out_list.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_list
