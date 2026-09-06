# -*- coding: utf-8 -*-
"""ui.theme 契约单测：离屏构建主题并渲染构件，不需真视口。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

dpg = pytest.importorskip("dearpygui.dearpygui")

from echosmith.ui import theme  # noqa: E402


def test_theme_apply_and_helpers():
    dpg.create_context()
    try:
        theme.apply()
        assert theme._theme is not None
        assert theme._theme_primary is not None
        assert theme._theme_danger is not None
        theme.apply()  # 幂等：二次调用不重建不报错

        # 语义色判定
        assert theme.status_color("训练完成") == theme.ACCENT
        assert theme.status_color("引擎就绪 ✔") == theme.ACCENT
        assert theme.status_color("失败：阶段 2-sv 退出码 1") == theme.DANGER
        assert theme.status_color("端口不可达 ✘") == theme.DANGER
        assert theme.status_color("空闲") == theme.TEXT_FAINT
        assert theme.status_color("阶段 2/6") == theme.TEXT

        # 构件助手在窗口上下文内可用
        with dpg.window():
            theme.header("分组标题")
            theme.caption("说明文字")
            item = theme.note("被动信息", wrap=300)
            theme.field_label("字段")
            btn = dpg.add_button(label="主按钮")
            theme.primary(btn)
            theme.danger(dpg.add_button(label="危险按钮"))
            assert dpg.get_item_theme(btn) is not None
        dpg.destroy_context()
    finally:
        try:
            dpg.destroy_context()
        except SystemError:
            pass


def test_contrast_minimums():
    """正文字色对背景的 WCAG 对比度 >= 4.5（背景按亮度取最深/最浅两种）。"""

    def lum(c):
        def ch(v):
            v = v / 255
            return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
        r, g, b = c[:3]
        return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)

    def ratio(a, b):
        la, lb = sorted((lum(a), lum(b)), reverse=True)
        return (la + 0.05) / (lb + 0.05)

    for text in (theme.TEXT, theme.TEXT_MUTED, theme.TEXT_FAINT,
                 theme.ACCENT, theme.DANGER, theme.WARNING):
        assert ratio(text, theme.BG) >= 4.5, f"{text} 对背景对比度不足"
    # 压在 accent 上的主按钮文字
    assert ratio(theme.ACCENT, theme.ON_ACCENT) >= 4.5
