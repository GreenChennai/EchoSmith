"""UVR5 人声/背景音分离：api_v2 不暴露此能力，改为向整合包注入驱动脚本，
用整合包自带 Python 调 tools/uvr5/lib 内部实现（CPU 模式），多策略回退。

注意：该模块依赖引擎实际版本内的 lib_5_0 接口，属于「需真机联调」路径；
失败时错误信息会原样透出，便于按引擎版本适配。
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from .config import Config
from .downloader import valid_engine
from .state import Shared

WIN_SILENT = 0x08000000 if sys.platform == "win32" else 0

DRIVER_NAME = "_echosmith_uvr5.py"

DRIVER_SRC = '''# -*- coding: utf-8 -*-
# EchoSmith 注入的 UVR5 驱动（CPU）。用法:
#   runtime/python.exe _echosmith_uvr5.py <input_wav> <vocal_dir> <ins_dir>
# 契约对照 RVC-Boss/GPT-SoVITS main：tools/uvr5/vr.py
#   AudioPre.__init__(agg, model_path, device, is_half, tta=False)
#   _path_audio_(music_file, ins_root, vocal_root, format="flac", is_hp3=False)
#   产物命名：vocal_root/vocal_<basename>_<agg>.wav
import os, sys, traceback
root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(root, "tools", "uvr5"))
inp, vocal_dir, ins_dir = sys.argv[1], sys.argv[2], sys.argv[3]
W = os.path.join(root, "tools", "uvr5", "uvr5_weights")

def run(model_file, cls_name, is_hp3):
    import vr
    cls = getattr(vr, cls_name)
    m = cls(agg=10, model_path=os.path.join(W, model_file),
            device="cpu", is_half=False)
    m._path_audio_(inp, ins_dir, vocal_dir, "wav", is_hp3)

errs = []
for model_file, cls_name in [
    ("HP5_only_main_vocal.pth", "AudioPre"),
    ("HP2_all_vocals.pth", "AudioPre"),
    ("VR-DeEchoAggressive.pth", "AudioPreDeEcho"),
]:
    if not os.path.isfile(os.path.join(W, model_file)):
        errs.append(f"{model_file}: 权重缺失")
        continue
    try:
        run(model_file, cls_name, "HP3" in model_file)
        print("UVR5_OK")
        sys.exit(0)
    except Exception:
        errs.append(f"{cls_name}({model_file}):\\n{traceback.format_exc()}")
print("UVR5_ALL_FAILED")
for e in errs:
    print(e)
sys.exit(3)
'''


class Uvr5Error(RuntimeError):
    pass


def separate(cfg: Config, shared: Shared, wav_path: Path) -> Path:
    """返回人声 wav 路径（引擎输出的 *_vocal*.wav 中最新者）。"""
    engine = cfg.engine_path
    py = valid_engine(engine)
    if py is None:
        raise Uvr5Error("引擎未就绪：请先在「引擎」页完成整合包下载/导入")
    driver = engine / DRIVER_NAME
    driver.write_text(DRIVER_SRC, encoding="utf-8")
    vocal_dir = wav_path.parent / "uvr5_vocal"
    ins_dir = wav_path.parent / "uvr5_instrument"
    vocal_dir.mkdir(parents=True, exist_ok=True)
    ins_dir.mkdir(parents=True, exist_ok=True)
    shared.log(f"[分离] 启动 UVR5（CPU）：{wav_path.name}")

    proc = subprocess.Popen(
        [str(py), DRIVER_NAME, str(wav_path), str(vocal_dir), str(ins_dir)],
        cwd=str(engine), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        creationflags=WIN_SILENT,
    )
    tail: list[str] = []
    for raw in proc.stdout:  # type: ignore[union-attr]
        line = raw.decode("utf-8", "replace").rstrip()
        tail.append(line)
        shared.log(f"[分离] {line}")
    proc.wait(timeout=3600 * 6)
    if proc.returncode != 0:
        raise Uvr5Error(
            "UVR5 分离失败（引擎版本接口差异？日志见上）。"
            "若 uvr5_weights 缺失，可在引擎 webui 中打开一次 UVR5 页触发模型下载。"
        )
    vocals = sorted(vocal_dir.glob("*vocal*.wav"), key=lambda p: p.stat().st_mtime)
    if not vocals:
        raise Uvr5Error(f"UVR5 未产出人声文件（{vocal_dir}）")
    out = vocals[-1]
    shared.log(f"[分离] 人声就绪：{out.name}")
    return out
