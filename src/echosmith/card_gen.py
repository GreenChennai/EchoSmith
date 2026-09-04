"""声线卡生成：把训练产物 + 参考音频打包成 EchoRunner 兼容的 card.json。"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .ffmpeg_util import probe_duration


class CardError(RuntimeError):
    pass


# GPT-SoVITS api_v2 对参考音频的硬性要求：3~10 秒
REF_MIN_S, REF_MAX_S = 3.0, 10.0


@dataclass
class RefRow:
    label: str
    ref: str          # 参考音频路径（相对引擎目录或绝对）
    prompt_text: str = ""
    aux_refs: list[str] = field(default_factory=list)


_SOVITS_ROOTS = ("SoVITS_weights_v2ProPlus", "SoVITS_weights_v2Pro", "SoVITS_weights_v4",
                 "SoVITS_weights_v3", "SoVITS_weights_v2", "SoVITS_weights")
_GPT_ROOTS = ("GPT_weights_v2ProPlus", "GPT_weights_v2Pro", "GPT_weights_v4",
              "GPT_weights_v3", "GPT_weights_v2", "GPT_weights")


def scan_weights(engine: Path, exp: str) -> dict[str, list[Path]]:
    """扫描训练产物：logs/<exp>/ 下的训练目录 + 引擎根版本目录（if_save_every_weights 拷贝，
    文件名以实验名开头）。"""
    out: dict[str, list[Path]] = {"sovits": [], "gpt": []}
    for sub in _SOVITS_ROOTS:
        d = engine / "logs" / exp / sub
        if d.is_dir():
            out["sovits"].extend(sorted(d.glob("*.pth"), key=lambda p: p.stat().st_mtime))
        d2 = engine / sub
        if d2.is_dir():
            out["sovits"].extend(sorted(
                (p for p in d2.glob("*.pth") if p.name.startswith(exp)),
                key=lambda p: p.stat().st_mtime))
    for sub in _GPT_ROOTS:
        d = engine / "logs" / exp / sub
        if d.is_dir():
            out["gpt"].extend(sorted(d.glob("*.ckpt"), key=lambda p: p.stat().st_mtime))
        d2 = engine / sub
        if d2.is_dir():
            out["gpt"].extend(sorted(
                (p for p in d2.glob("*.ckpt") if p.name.startswith(exp)),
                key=lambda p: p.stat().st_mtime))
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
    # api_v2 要求参考音频 3~10 秒，越界的参考音频合成时必失败——生成期就拦下
    usable: list[RefRow] = []
    for r in good_rows:
        p = Path(r.ref)
        if not p.is_absolute():
            p = engine / p
        try:
            dur = probe_duration(cfg, p)
        except Exception:
            raise CardError(f"参考音频不可读：{r.ref}")
        if dur < REF_MIN_S or dur > REF_MAX_S:
            continue
        usable.append(r)
    if not usable:
        raise CardError(
            f"所有参考音频都不在 {REF_MIN_S:.0f}~{REF_MAX_S:.0f} 秒范围内"
            "（api_v2 硬性要求），请换用训练集里时长合规的切片"
        )
    good_rows = usable
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
