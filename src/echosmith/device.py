"""设备后端兼容层：CUDA 原生 / AMD via ZLUDA / CPU 的探测与环境注入。

三类后端的判定与契约：

- **cuda（NVIDIA 原生）**：``torch.cuda.is_available()`` 为真且未开 ZLUDA；
  整合包 torch 是 cu118 构建，开箱即用。
- **zluda（AMD 经 CUDA 转译）**：``zluda_mode`` 开启后，子进程环境注入
  ``HIP_PATH`` / ``ZLUDA_MODE`` / PATH 前置 ZLUDA 目录，torch 依旧认为自己在
  用 CUDA——引擎脚本无需区分。要求整合包引擎已应用 zluda_patch 补丁
  （下载后与训练/引擎启动前自动应用）。
- **cpu**：以上皆不可用时兜底；训练配置 batch 减半、fp32（与 webui 一致）。

``probe()`` 用整合包自带 runtime 实测 ``torch.cuda.is_available()``，供设置页
显示与故障定位；它不改变任何配置。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from .config import Config

WIN_SILENT = 0x08000000 if sys.platform == "win32" else 0


class DeviceError(RuntimeError):
    pass


def zluda_env(cfg: Config) -> dict:
    """按配置返回 ZLUDA 子进程环境；未启用返回空 dict（零副作用）。

    启用但配置不完整时抛 DeviceError，信息给到能在设置页解决的程度。
    """
    if not cfg["zluda_mode"]:
        return {}
    zluda_dir = str(cfg["zluda_dir"] or "").strip()
    hip_path = str(cfg["hip_path"] or "").strip()
    if not zluda_dir or not hip_path:
        raise DeviceError(
            "已启用 ZLUDA GPU 加速，但未配置 ZLUDA 目录 / HIP SDK 路径（设置页）"
        )
    if not (Path(zluda_dir) / "nvcuda.dll").is_file():
        raise DeviceError(f"ZLUDA 目录无效（缺 nvcuda.dll）：{zluda_dir}")
    if not (Path(hip_path) / "bin" / "amdhip64_7.dll").is_file():
        raise DeviceError(f"HIP SDK 目录无效（缺 bin/amdhip64_7.dll）：{hip_path}")
    return {
        "HIP_PATH": hip_path,
        "ZLUDA_MODE": "1",  # 引擎补丁开关：g2pw 走 CPU EP、cuDNN 按 MIOpen 守卫
        "PATH": f"{zluda_dir};{Path(hip_path) / 'bin'};"
                + os.environ.get("PATH", ""),
    }


_PROBE_CODE = '''
import json
try:
    import torch
    info = {"ok": True, "torch": torch.__version__, "cuda_build": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available())}
    if info["cuda_available"]:
        info["device_name"] = torch.cuda.get_device_name(0)
except Exception as e:
    info = {"ok": False, "error": f"{type(e).__name__}: {e}"}
print("##ECHOSMITH_PROBE##" + json.dumps(info))
'''


def probe(engine_dir: Path, cfg: Config, timeout: float = 180.0) -> dict:
    """用引擎自带 runtime 实测 torch CUDA 可用性。

    返回 dict：``backend``（cuda / zluda / cpu / unknown）、``device_name``、
    ``torch``、``detail``。ZLUDA 模式下会注入环境再测；首次探测在 ZLUDA 下
    可能需要编译/枚举，超时给足。
    """
    engine_dir = Path(engine_dir)
    py = engine_dir / "runtime" / "python.exe"
    if not py.is_file():
        return {"ok": False, "backend": "unknown",
                "detail": f"引擎 runtime 不存在：{py}"}
    env = {**os.environ, **zluda_env(cfg)}
    backend = "zluda" if cfg["zluda_mode"] else "cuda"
    try:
        p = subprocess.run(
            [str(py), "-c", _PROBE_CODE], cwd=str(engine_dir), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=timeout, creationflags=WIN_SILENT,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "backend": "unknown",
                "detail": f"探测超时（{timeout:.0f}s）"}
    out = p.stdout.decode("utf-8", "replace")
    line = next((l for l in out.splitlines() if "##ECHOSMITH_PROBE##" in l), None)
    if line is None:
        return {"ok": False, "backend": "unknown",
                "detail": out.strip()[-400:] or "无输出"}
    info = json.loads(line.split("##ECHOSMITH_PROBE##", 1)[1])
    if not info.get("ok"):
        return {"ok": False, "backend": "unknown",
                "detail": info.get("error", "未知错误")}
    if not info["cuda_available"]:
        return {"ok": True, "backend": "cpu", "torch": info["torch"],
                "detail": "torch 报告 CUDA 不可用（CPU 模式）"}
    return {
        "ok": True, "backend": backend, "device_name": info["device_name"],
        "torch": info["torch"], "cuda_build": info.get("cuda_build"),
        "detail": info["device_name"],
    }
