"""DPG 中文字体加载（内置 HarmonyOS Sans SC，回退系统字体）。"""
from __future__ import annotations

import sys
from pathlib import Path

import dearpygui.dearpygui as dpg

_DEFAULT_SIZE = 15
_SMALL_SIZE = 13    # 说明/提示（theme.caption / theme.note）
_HEADER_SIZE = 17   # 分组标题（theme.header）
_BUNDLED = Path(__file__).resolve().parents[2] / "assets" / "fonts" / "HarmonyOS_Sans_SC_Regular.ttf"
_FALLBACKS = [
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
    Path("C:/Windows/Fonts/simsun.ttc"),
]

_fonts: dict[str, int] = {}


def _frozen_bundled() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "assets" / "fonts" / "HarmonyOS_Sans_SC_Regular.ttf"
    return _BUNDLED


def load_cjk_font() -> bool:
    """在 create_context 之后、setup_dearpygui 之前调用。dearpygui>=2.0 字符范围全自动。

    顺带注册小号/标题字号供 theme 层级使用；失败时只退回单一字号。
    """
    for path in [_frozen_bundled(), *_FALLBACKS]:
        if not path.is_file():
            continue
        try:
            with dpg.font_registry():
                _fonts["default"] = dpg.add_font(str(path), _DEFAULT_SIZE)
                for kind, size in (("small", _SMALL_SIZE), ("header", _HEADER_SIZE)):
                    try:
                        _fonts[kind] = dpg.add_font(str(path), size)
                    except SystemError:
                        pass  # 个别系统字体缺字形表时忽略次级字号
            dpg.bind_font(_fonts["default"])
            return True
        except SystemError:
            continue
    return False


def font(kind: str = "default") -> int | None:
    """已注册的字体句柄；未加载返回 None。"""
    return _fonts.get(kind)
