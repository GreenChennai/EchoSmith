"""GPT-SoVITS 引擎进程管理：启动整合包内 api_v2.py，等待端口就绪，收集日志。"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from .config import Config
from .state import Shared

WIN_SILENT = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW


class EngineError(RuntimeError):
    pass


def _find_engine_python(engine_dir: Path) -> Path:
    candidates = [
        engine_dir / "runtime" / "python.exe",
        engine_dir / "runtime" / "python3.exe",
        engine_dir / "python" / "python.exe",
        engine_dir / "venv" / "Scripts" / "python.exe",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise EngineError(f"整合包内未找到 Python 运行时（{engine_dir}）")


class Engine:
    """负责 api_v2.py 子进程生命周期。所有方法可在任意线程调用。"""

    def __init__(self, cfg: Config, shared: Shared) -> None:
        self.cfg = cfg
        self.shared = shared
        self.proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    # -- 探测 ---------------------------------------------------------------
    def configured(self) -> bool:
        d = self.cfg.engine_path
        return bool(self.cfg.data["engine_dir"]) and (d / "api_v2.py").is_file()

    def port_open(self, timeout: float = 0.4) -> bool:
        try:
            with socket.create_connection(
                (self.cfg.data["engine_host"], int(self.cfg.data["engine_port"])), timeout
            ):
                return True
        except OSError:
            return False

    def running(self) -> bool:
        with self._lock:
            return self.proc is not None and self.proc.poll() is None

    # -- 启停 ----------------------------------------------------------------
    def _build_env(self) -> dict:
        """构造引擎子进程环境(设备后端:CPU / NVIDIA CUDA 原生)。"""
        return os.environ.copy()

    def start(self, wait_timeout: float = 180.0) -> None:
        with self._lock:
            if self.proc is not None and self.proc.poll() is not None:
                self.proc = None
            if self.proc is not None:
                return
            if not self.configured():
                raise EngineError("未配置引擎目录，或目录内没有 api_v2.py")
            engine_dir = self.cfg.engine_path
            python = _find_engine_python(engine_dir)
            env = self._build_env()
            tag = ""
            cmd = [
                str(python), "api_v2.py",
                "-a", str(self.cfg.data["engine_host"]),
                "-p", str(int(self.cfg.data["engine_port"])),
            ]
            self.shared.log(f"[引擎] 启动{tag}: {' '.join(cmd)}（工作目录 {engine_dir}）")
            try:
                self.proc = subprocess.Popen(
                    cmd,
                    cwd=str(engine_dir),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    creationflags=WIN_SILENT,
                )
            except OSError as e:
                raise EngineError(f"引擎进程启动失败: {e}") from e
            threading.Thread(target=self._pump_log, daemon=True).start()

        if not self._wait_ready(wait_timeout):
            raise EngineError(
                f"引擎在 {wait_timeout:.0f}s 内未就绪（端口 "
                f"{self.cfg.data['engine_port']} 未监听），请查看引擎日志页"
            )
        self.shared.log("[引擎] 端口就绪，可以开始合成")

    def stop(self) -> None:
        with self._lock:
            proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
        except OSError:
            pass
        self.shared.log("[引擎] 已停止")

    # -- 内部 ------------------------------------------------------------------
    def _wait_ready(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                return False  # 进程提前退出
            if self.port_open():
                return True
            time.sleep(0.5)
        return False

    def _pump_log(self) -> None:
        proc = self.proc
        if proc is None or proc.stdout is None:
            return
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", errors="replace")
            self.shared.engine_log_line(line)
        if self.proc is proc and proc.poll() is not None:
            self.shared.log(f"[引擎] 进程已退出，返回码 {proc.returncode}")
