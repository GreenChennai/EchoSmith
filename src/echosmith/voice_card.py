"""声线卡：models/voices/<名称>/card.json

card.json 格式（相对路径一律相对 GPT-SoVITS 整合包根目录，与 api_v2 行为一致）：
{
    "name": "小雨",
    "gpt": "GPT_weights_v2/xiaoyu-e10.ckpt",
    "sovits": "SoVITS_weights_v2/xiaoyu-e10.pth",
    "default_emotion": "平静",
    "emotions": {
        "平静": {"ref": "refs/xiaoyu/calm.wav", "prompt_text": "参考音频里说的那句话。"},
        "开心": {"ref": "refs/xiaoyu/happy.wav", "prompt_text": "……", "aux_refs": []}
    }
}
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

CARD_FILE = "card.json"


@dataclass
class Emotion:
    label: str
    ref: str                      # 参考音频路径（相对引擎目录或绝对）
    prompt_text: str = ""
    aux_refs: list[str] = field(default_factory=list)


@dataclass
class VoiceCard:
    name: str
    dir: Path
    gpt: str = ""
    sovits: str = ""
    emotions: dict[str, Emotion] = field(default_factory=dict)
    default_emotion: str = ""

    def emotion_labels(self) -> list[str]:
        return list(self.emotions.keys())

    def pick_emotion(self, label: str | None) -> Emotion:
        if label and label in self.emotions:
            return self.emotions[label]
        if self.default_emotion in self.emotions:
            return self.emotions[self.default_emotion]
        return next(iter(self.emotions.values()))


def scan_voice_cards(voices_dir: Path) -> list[VoiceCard]:
    """扫描 voices_dir 下所有含 card.json 的一级子目录，按名称排序。"""
    cards: list[VoiceCard] = []
    if not voices_dir.is_dir():
        return cards
    for sub in sorted(voices_dir.iterdir()):
        card_path = sub / CARD_FILE
        if not sub.is_dir() or not card_path.is_file():
            continue
        card = load_card(card_path)
        if card:
            cards.append(card)
    return cards


def load_card(card_path: Path) -> VoiceCard | None:
    try:
        data = json.loads(card_path.read_text("utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    emotions: dict[str, Emotion] = {}
    for label, e in (data.get("emotions") or {}).items():
        if not isinstance(e, dict) or not e.get("ref"):
            continue
        emotions[label] = Emotion(
            label=label,
            ref=str(e["ref"]),
            prompt_text=str(e.get("prompt_text", "")),
            aux_refs=[str(x) for x in (e.get("aux_refs") or [])],
        )
    if not emotions:
        return None
    default = data.get("default_emotion") or next(iter(emotions))
    return VoiceCard(
        name=str(data.get("name") or card_path.parent.name),
        dir=card_path.parent,
        gpt=str(data.get("gpt", "")),
        sovits=str(data.get("sovits", "")),
        emotions=emotions,
        default_emotion=default,
    )
