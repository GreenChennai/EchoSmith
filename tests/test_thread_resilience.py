# -*- coding: utf-8 -*-
"""后台线程韧性回归测试：意外异常必须落到 UI，不得静默杀死线程。"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from echosmith.state import Shared  # noqa: E402


def _wait_error(shared: Shared, needle: str, timeout: float = 3.0) -> bool:
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        if needle in shared.snapshot()["error"]:
            return True
        time.sleep(0.02)
    return False


def test_trainer_survives_unexpected_exception(monkeypatch):
    """_run 抛出窄异常集之外的异常（如 JSONDecodeError）时必须落到 UI。"""
    from echosmith.trainer import Trainer

    shared = Shared()
    tr = Trainer.__new__(Trainer)  # 跳过 __init__（不需要真配置）
    threading.Thread.__init__(tr, name="t", daemon=True)
    tr.shared = shared
    tr.cancelled = threading.Event()
    tr.proc = None

    def boom():
        raise ValueError("Expecting value: line 1 column 1 (char 0)")

    monkeypatch.setattr(tr, "_run", boom)
    tr.start()
    tr.join(timeout=5)
    assert not tr.is_alive()
    assert _wait_error(shared, "ValueError"), "意外异常未落到 shared.error"
    assert shared.snapshot()["tr"][1] != "", "失败状态未写"


def test_pipeline_survives_unexpected_exception(monkeypatch):
    from echosmith.pipeline import Pipeline

    shared = Shared()
    p = Pipeline.__new__(Pipeline)
    threading.Thread.__init__(p, name="p", daemon=True)
    p.shared = shared
    p.cancelled = threading.Event()

    def boom():
        raise KeyError("subprocess.TimeoutExpired 模拟")

    monkeypatch.setattr(p, "_run", boom)
    p.start()
    p.join(timeout=5)
    assert not p.is_alive()
    assert _wait_error(shared, "KeyError"), "意外异常未落到 shared.error"
    assert shared.snapshot()["pl"][1] != "", "失败状态未写"


def test_downloader_survives_unexpected_exception(monkeypatch):
    from echosmith.downloader import DownloadTask

    shared = Shared()
    t = DownloadTask.__new__(DownloadTask)
    threading.Thread.__init__(t, name="t", daemon=True)
    t.shared = shared
    t.cancelled = threading.Event()
    t.running = threading.Event()

    def boom():
        raise RuntimeError("py7zr 内部异常模拟")

    monkeypatch.setattr(t, "_run", boom)
    t.start()
    t.join(timeout=5)
    assert not t.is_alive()
    assert _wait_error(shared, "RuntimeError"), "意外异常未落到 shared.error"
