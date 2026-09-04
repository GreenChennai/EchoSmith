"""声线卡生成：把训练产物 + 参考音频打包成 EchoRunner 兼容的 card.json。"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config


class CardError(RuntimeError):
    pass


@dataclass
class RefRow:
    label: str
    ref: str          # 参考音频路径（相对引擎目录或绝对）
    prompt_text: str = ""
    aux_refs: list[str] = field(default_factory=list)


def scan_weights(engine: Path, exp: str) -> dict[str, list[Path]]:
    """扫描 logs/<exp> 下的训练产物（兼容 v2/v2Pro 与 v1 目录名）。"""
    out: dict[str, list[Path]] = {"sovits": [], "gpt": []}
    for sub in ("SoVITS_weights_v2", "SoVITS_weights"):
        d = engine / "logs" / exp / sub
        if d.is_dir():
            out["sovits"].extend(sorted(d.glob("*.pth"), key=lambda p: p.stat().st_mtime))
    for sub in ("GPT_weights_v2", "GPT_weights"):
        d = engine / "logs" / exp / sub
        if d.is_dir():
            out["gpt"].extend(sorted(d.glob("*.ckpt"), key=lambda p: p.stat().st_mtime))
    return out


def _rel_to_engine(engine: Path, p: Path) -> str:
    try:
        return p.resolve().relative_to(engine.resolve()).as_posix()
    except ValueError:
        return p.resolve().as_posix()


def build_card(cfg: Config, engine: Path, exp: str, voice_name: str,
               default_emotion: str, rows: list[RefRow]) -> Path:
    if not voice_name.strip():
        raise CardError("声线名称为空")
    weights = scan_weights(engine, exp)
    if not weights["sovits"]:
        raise CardError(f"logs/{exp} 下没有 SoVITS 权重（先完成训练）")
    if not weights["gpt"]:
        raise CardError(f"logs/{exp} 下没有 GPT 权重（先完成训练）")
    good_rows = [r for r in rows if r.label.strip() and r.ref.strip()]
    if not good_rows:
        raise CardError("至少需要一条参考音频（情感标签 + 文件）")
    labels = [r.label.strip() for r in good_rows]
    if len(labels) != len(set(labels)):
        raise CardError("情感标签有重复")
    if default_emotion.strip() and default_emotion.strip() not in labels:
        raise CardError(f"默认情感「{default_emotion.strip()}」不在标签列表中")

    card = {
        "name": voice_name.strip(),
        "gpt": _rel_to_engine(engine, weights["gpt"][-1]),
        "sovits": _rel_to_engine(engine, weights["sovits"][-1]),
        "default_emotion": default_emotion.strip() or labels[0],
        "emotions": {
            r.label.strip(): {
                "ref": _rel_to_engine(engine, Path(r.ref)),
                "prompt_text": r.prompt_text.strip(),
                "aux_refs": [],
            }
            for r in good_rows
        },
        "_generated_by": "EchoSmith",
        "_created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    from .config import resolve  # noqa: PLC0415
    out_dir = resolve(cfg.data.get("voices_dir", "models/voices")) / voice_name.strip()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "card.json"
    out.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
