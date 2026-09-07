# -*- coding: utf-8 -*-
"""worker 线程韧性回归测试：清空竞态 + 意外异常不得杀死队列线程。"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from echosmith.queue_worker import Job, Worker  # noqa: E402
from echosmith.state import Shared  # noqa: E402
from echosmith.voice_card import VoiceCard  # noqa: E402


class _Cfg:
    engine_url = "http://127.0.0.1:9"

    def __getitem__(self, k):
        return None


class _Engine:
    pass


def _mk_worker() -> Worker:
    w = Worker(_Cfg(), _Engine(), Shared())
    card = VoiceCard(name="t", dir=Path("."), gpt="g", sovits="s",
                     default_emotion="平静")
    card.emotions = {"平静": None}  # 仅占位,stub 的 _run_job 不会用到
    w._card = card
    return w


def _mk_job() -> Job:
    card = VoiceCard(name="t", dir=Path("."), gpt="g", sovits="s",
                     default_emotion="平静")
    return Job(card=card, emotion="平静", lines=["a", "b"], text_lang="zh",
               speed_factor=1.0, split_method="cut5")


def test_worker_survives_clear_during_run(monkeypatch):
    """任务运行中点「清空队列」：任务结束后不得因 popleft 空队列而杀死线程。"""
    w = _mk_worker()
    started = threading.Event()
    gate = threading.Event()

    def fake_run_job(job):
        started.set()
        gate.wait(5)

    monkeypatch.setattr(w, "_run_job", fake_run_job)
    w.start()
    try:
        w.submit(_mk_job())
        assert started.wait(5), "任务未开始"
        w.clear()          # 用户在任务运行中点「清空队列」
        gate.set()         # 放行,让任务"完成"
        deadline = time.time() + 3
        while time.time() < deadline and len(w._jobs) == 0 and threading.active_count() > 1 \
                and not getattr(w, "_loop_done", False):
            time.sleep(0.05)
        time.sleep(0.3)    # 给 run 循环时间踩到 popleft
        assert w.is_alive(), "worker 线程在清空竞态下死亡（IndexError 未捕获）"
        assert w.shared.snapshot()["queue_status"] == "空闲"
    finally:
        w.stopped.set()
        w._resume.set()
        w._wake.set()
        w.join(timeout=3)


def test_worker_survives_unexpected_exception(monkeypatch):
    """_run_job 抛非预期异常（不在窄异常集内）时,线程必须存活并把错误落到 UI。"""
    w = _mk_worker()

    def boom(job):
        raise IndexError("模拟意外异常")

    monkeypatch.setattr(w, "_run_job", boom)
    w.start()
    try:
        w.submit(_mk_job())
        deadline = time.time() + 3
        while time.time() < deadline and "IndexError" not in w.shared.snapshot()["error"]:
            time.sleep(0.05)
        assert "IndexError" in (w.shared.snapshot()["queue_error"] + w.shared.snapshot()["error"]), "意外异常未落到 UI"
        assert w.is_alive(), "worker 线程因意外异常死亡"
        assert len(w._jobs) == 0, "出错后应清空剩余队列（遇错停止）"
    finally:
        w.stopped.set()
        w._resume.set()
        w._wake.set()
        w.join(timeout=3)
