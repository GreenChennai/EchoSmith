"""EchoUI 设计令牌与 Dear PyGui 主题（与 EchoRunner 同源，改动需两边同步）。

设计读场：桌面工具型应用（音频生产工作台），深色专业语言。
拨盘：variance 3（工具可预测性优先）、motion 1（即时模式无动效）、density 6。
色彩纪律：slate 深中性非纯黑；**单一 accent = 绿**（"开始/运行"走带语义），
警示琥珀、危险红各司其职；正文/次级/提示三档文字全部过 4.5:1 对比。
形状锁：圆角一律 4px（滚动条 6px）；内边距统一；无动效即无分心动效。
"""
from __future__ import annotations

import sys

import dearpygui.dearpygui as dpg

RGB = tuple[int, int, int]

# ---- 色板（WCAG 对 bg 均 >= 4.5:1） -----------------------------------------
BG = (15, 23, 42)            # 0F172A slate-900 窗口底
BG_CHILD = (19, 26, 41)      # 日志等子区（比窗口更深一档）
BG_CARD = (23, 30, 48)       # 弹层/浮出
BG_FRAME = (30, 41, 59)      # 1E293B 输入框/按钮
BG_FRAME_HOVER = (37, 50, 70)
BG_FRAME_ACTIVE = (43, 58, 80)
BORDER = (51, 65, 85)        # 334155 slate-700

TEXT = (241, 245, 249)       # F1F5F9 正文（~15:1）
TEXT_MUTED = (148, 163, 184)  # 94A3B8 次级/标签（~7:1）
TEXT_FAINT = (122, 136, 154)  # 被动信息（~4.7:1）

ACCENT = (34, 197, 94)       # 22C55E 主操作/选中/进度
ACCENT_HOVER = (74, 222, 128)
ACCENT_ACTIVE = (22, 163, 74)
ON_ACCENT = (15, 23, 42)     # 压在 accent 上的深字

DANGER = (239, 68, 68)       # EF4444 报错文本/危险语义
DANGER_BG = (58, 26, 29)     # 危险按钮底（subtle，不抢主操作）
DANGER_HOVER = (76, 32, 36)
DANGER_TEXT = (252, 165, 165)

WARNING = (234, 179, 8)      # EAB308 警示提示

# ---- 全局主题句柄（apply() 后有效） -----------------------------------------
_theme = None
_theme_primary = None
_theme_danger = None


def apply() -> None:
    """构建并绑定全局主题 + 主/危险按钮变体。在 create_context 之后调用一次。"""
    global _theme, _theme_primary, _theme_danger
    if _theme is not None:
        return

    with dpg.theme() as _theme:
        with dpg.theme_component(dpg.mvAll):
            # —— 颜色 ——
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, BG, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, BG, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, BG_CARD, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, BG_CHILD, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg, (*BG_CARD, 252), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Border, BORDER, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Text, TEXT, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, TEXT_FAINT, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, BG_FRAME, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, BG_FRAME_HOVER, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, BG_FRAME_ACTIVE, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Button, BG_FRAME, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, BG_FRAME_HOVER, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, BG_FRAME_ACTIVE, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark, ACCENT, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, ACCENT, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, ACCENT_ACTIVE, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, ACCENT, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_PlotHistogramHovered, ACCENT_HOVER, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Separator, BORDER, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, BG, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, BORDER, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered, (71, 85, 105), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabActive, (100, 116, 139), category=dpg.mvThemeCat_Core)
            # Tab：选中项与窗口融为一体 + accent 顶线（现代下划线 tab）
            dpg.add_theme_color(dpg.mvThemeCol_Tab, BG, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, BG_FRAME_HOVER, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TabSelected, BG, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TabSelectedOverline, ACCENT, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TabDimmed, BG, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TabDimmedSelected, BG_FRAME, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Header, BG_FRAME, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, BG_FRAME_HOVER, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, BG_FRAME_ACTIVE, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ModalWindowDimBg, (0, 0, 0, 120), category=dpg.mvThemeCat_Core)
            # —— 形状/密度（形状锁：一律 4px 圆角） ——
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 14, 14, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 10, 6, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 10, 8, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_ItemInnerSpacing, 8, 6, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_IndentSpacing, 18, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 8, 5, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_PopupRounding, 4, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 4, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 4, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 4, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_ScrollbarRounding, 6, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize, 0, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_TabBarBorderSize, 1, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_TabBarOverlineSize, 2, category=dpg.mvThemeCat_Core)
    dpg.bind_theme(_theme)

    # 主按钮变体：accent 底 + 深字（页面上每个 tab 至多一个）
    with dpg.theme() as _theme_primary:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, ACCENT, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, ACCENT_HOVER, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, ACCENT_ACTIVE, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Text, ON_ACCENT, category=dpg.mvThemeCat_Core)

    # 危险按钮变体：暗红底 subtle（停止/取消/清空类）
    with dpg.theme() as _theme_danger:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, DANGER_BG, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, DANGER_HOVER, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, DANGER_HOVER, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Text, DANGER_TEXT, category=dpg.mvThemeCat_Core)


def primary(tag: str) -> None:
    """把主按钮变体绑到指定按钮（页面的唯一主动作）。"""
    if _theme_primary is not None:
        dpg.bind_item_theme(tag, _theme_primary)


def danger(tag: str) -> None:
    """把危险按钮变体绑到指定按钮（停止/取消/清空）。"""
    if _theme_danger is not None:
        dpg.bind_item_theme(tag, _theme_danger)


# ---- 文本构件助手（层级 = 字号 × 颜色） --------------------------------------
def header(text: str) -> None:
    """分组标题：较大字号 + 正文色。"""
    item = dpg.add_text(text)
    from ..fontutil import font  # noqa: PLC0415
    f = font("header")
    if f is not None:
        dpg.bind_item_font(item, f)


def caption(text: str) -> None:
    """分组说明行：小字号 + 次级色。"""
    item = dpg.add_text(text, color=TEXT_MUTED)
    from ..fontutil import font  # noqa: PLC0415
    f = font("small")
    if f is not None:
        dpg.bind_item_font(item, f)


def note(text: str = "", **kwargs) -> int:
    """被动信息行（最近输出等）：小字号 + 弱色。"""
    item = dpg.add_text(text, color=TEXT_FAINT, **kwargs)
    from ..fontutil import font  # noqa: PLC0415
    f = font("small")
    if f is not None:
        dpg.bind_item_font(item, f)
    return item


def field_label(text: str) -> None:
    """输入框前的字段标签：次级色。"""
    dpg.add_text(text, color=TEXT_MUTED)


# ---- 语义状态色 --------------------------------------------------------------
_OK_MARKS = ("完成", "就绪", "✔", "成功", "通过")
_BAD_MARKS = ("失败", "错误", "✘", "不可达")
_IDLE_MARKS = ("空闲", "停止", "已取消")


def status_color(text: str) -> RGB:
    """按状态文本语义取色：成功=accent、失败=红、空闲=弱、进行中=正文色。"""
    if any(k in text for k in _BAD_MARKS):
        return DANGER
    if any(k in text for k in _OK_MARKS):
        return ACCENT
    if any(k in text for k in _IDLE_MARKS) or not text.strip():
        return TEXT_FAINT
    return TEXT


def set_status(tag: str, text: str) -> None:
    """写状态文本并同步语义色。"""
    dpg.set_value(tag, text)
    dpg.configure_item(tag, color=status_color(text))


def polish_viewport(title: str) -> None:
    """show_viewport 之后调用（Windows）：
    1. 修正视口标题 —— DPG 的 create_viewport/set_viewport_title 对中文标题会
       产生 GBK 乱码（UTF-8 字节被当宽字符直传 GLFW），故用 Win32
       SetWindowTextW 以 UTF-16 直写，绕开 DPG 的标题通道；
    2. 标题栏切暗色（DWMWA_USE_IMMERSIVE_DARK_MODE，Win10 1809+，旧系统静默跳过）。
    """
    if sys.platform != "win32":
        return
    import ctypes
    import os

    u32 = ctypes.WinDLL("user32")
    u32.SetWindowTextW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    u32.GetWindowThreadProcessId.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.wintypes.DWORD)]
    u32.IsWindowVisible.argtypes = [ctypes.c_void_p]
    hits: list[int] = []

    @ctypes.WINFUNCTYPE(ctypes.c_ssize_t, ctypes.c_ssize_t, ctypes.c_ssize_t)
    def _cb(hwnd, _lparam):
        pid = ctypes.wintypes.DWORD()
        u32.GetWindowThreadProcessId(ctypes.c_void_p(hwnd), ctypes.byref(pid))
        if pid.value == os.getpid() and u32.IsWindowVisible(ctypes.c_void_p(hwnd)):
            hits.append(hwnd)
            return 0
        return 1

    u32.EnumWindows(_cb, 0)
    if not hits:
        return
    hwnd = ctypes.c_void_p(hits[0])
    u32.SetWindowTextW(hwnd, title)
    dwm = ctypes.WinDLL("dwmapi")
    val = ctypes.c_int(1)
    dwm.DwmSetWindowAttribute(hwnd, 20,  # IMMERSIVE_DARK_MODE
                              ctypes.byref(val), ctypes.sizeof(val))
