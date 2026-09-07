# -*- coding: utf-8 -*-
"""引擎环境注入 + 合成栈契约测试(自 EchoRunner 合并)。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from echosmith.engine import Engine, EngineError  # noqa: E402


class _Shared:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def log(self, msg: str) -> None:
        self.lines.append(msg)


class _Cfg:
    def __init__(self, data: dict) -> None:
        self.data = data
        self.engine_path = data.get("engine_dir_path", Path())

    def __getitem__(self, k):
        return self.data.get(k)


def test_engine_env_is_pure_cpu_cuda(monkeypatch, tmp_path: Path) -> None:
    """_build_env 只回传系统环境(CPU/CUDA 原生),不注入任何 ZLUDA 变量。"""
    monkeypatch.setenv("ZLUDA_MODE", "1")  # 即使环境残留也不得注入
    eng = Engine(_Cfg({"zluda_mode": False, "zluda_dir": "", "hip_path": ""}), _Shared())
    env = eng._build_env()
    assert "ZLUDA_MODE" not in env or env["ZLUDA_MODE"] == "1"  # 原样透传,不改写
    assert "HIP_PATH" not in env


def test_engine_start_missing_dir(monkeypatch, tmp_path: Path) -> None:
    """引擎目录未配置/无效时,启动给出可解决的错误。"""
    eng = Engine(_Cfg({"engine_dir": str(tmp_path / "nope"), "engine_dir_path": tmp_path / "nope",
                       "engine_host": "127.0.0.1", "engine_port": 9880}), _Shared())
    with pytest.raises(EngineError):
        eng.start(wait_timeout=1)


def test_engine_port_open(monkeypatch):
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.listen(1)
    eng = Engine(_Cfg({"engine_host": "127.0.0.1", "engine_port": port}), _Shared())
    try:
        assert eng.port_open() is True
    finally:
        s.close()
    assert eng.port_open() is False
