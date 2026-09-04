"""DPG 中文字体加载（内置 HarmonyOS Sans SC，回退系统字体）。"""
from __future__ import annotations

import sys
from pathlib import Path

import dearpygui.dearpygui as dpg

_FONT_SIZE = 15
_BUNDLED = Path(__file__).resolve().parents[2] / "assets" / "fonts" / "HarmonyOS_Sans_SC_Regular.ttf"
_FALLBACKS = [
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
    Path("C:/Windows/Fonts/simsun.ttc"),
]


def _frozen_bundled() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "assets" / "fonts" / "HarmonyOS_Sans_SC_Regular.ttf"
    return _BUNDLED


def load_cjk_font() -> bool:
    """在 create_context 之后、setup_dearpygui 之前调用。dearpygui>=2.0 字符范围全自动。"""
    for path in [_frozen_bundled(), *_FALLBACKS]:
        if not path.is_file():
            continue
        try:
            with dpg.font_registry():
                font = dpg.add_font(str(path), _FONT_SIZE)
            dpg.bind_font(font)
            return True
        except SystemError:
            continue
    return False
