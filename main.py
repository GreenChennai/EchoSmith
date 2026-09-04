"""EchoSmith 入口：Windows DPI 感知 + 启动 UI。"""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _enable_windows_dpi_awareness() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except (AttributeError, OSError):
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            pass


def main() -> None:
    _enable_windows_dpi_awareness()
    from echosmith.ui.app import run  # noqa: PLC0415
    run()


if __name__ == "__main__":
    main()
