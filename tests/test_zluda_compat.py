# -*- coding: utf-8 -*-
"""zluda_patch / device 契约单测：假引擎目录，不依赖真整合包。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from echosmith.device import DeviceError, zluda_env  # noqa: E402
from echosmith.zluda_patch import (  # noqa: E402
    apply,
    check,
    deploy_runtime_shims,
    restore_runtime_shims,
)

# 与官方整合包一致的锚点样本 --------------------------------------------------
G2PW_SINGLE = '''import os
import onnxruntime

class OnnxProvider:
    def __init__(self):
        if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            self.provider = ["CUDAExecutionProvider", "CPUExecutionProvider"]
'''

G2PW_MULTI = '''import os
import onnxruntime

class OnnxProvider:
    def __init__(self):
        if (
            "CUDAExecutionProvider" in onnxruntime.get_available_providers()
        ):
            self.provider = ["CUDAExecutionProvider", "CPUExecutionProvider"]
'''

TRAIN_SCRIPT = '''import os
import sys
import torch
import torch.nn as nn

def main():
    print(torch.cuda.is_available())
'''

TTS_SCRIPT = '''import os
import torchaudio

import torch
import torch.nn as nn

class TTS:
    pass
'''


@pytest.fixture()
def fake_engine(tmp_path: Path) -> Path:
    """最小整合包骨架：api_v2.py + 全部补丁目标脚本。"""
    eng = tmp_path / "GPT-SoVITS"
    for rel, body in [
        ("api_v2.py", "# api_v2 entry\n"),
        ("GPT_SoVITS/text/g2pw/onnx_api.py", G2PW_SINGLE),
        ("GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py", TRAIN_SCRIPT),
        ("GPT_SoVITS/prepare_datasets/2-get-sv.py", TRAIN_SCRIPT),
        ("GPT_SoVITS/prepare_datasets/3-get-semantic.py", TRAIN_SCRIPT),
        ("GPT_SoVITS/s2_train.py", TRAIN_SCRIPT),
        ("GPT_SoVITS/s2_train_v3_lora.py", TRAIN_SCRIPT),
        ("GPT_SoVITS/s1_train.py", TRAIN_SCRIPT),
        ("GPT_SoVITS/TTS_infer_pack/TTS.py", TTS_SCRIPT),
        ("tools/uvr5/vr.py", TRAIN_SCRIPT),
    ]:
        f = eng / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(body, encoding="utf-8")
    return eng


# 补丁应用 / 幂等 / 语法 -------------------------------------------------------
def test_apply_and_idempotent(fake_engine: Path) -> None:
    done = apply(fake_engine)
    assert done, "首次应用应有产出"
    applied, missing = check(fake_engine)
    assert not missing, f"应用后不应有缺失：{[p.rel for p in missing]}"
    assert len(applied) == len(done)
    assert apply(fake_engine) == [], "二次应用必须幂等（零改动）"


def test_patched_scripts_compile(fake_engine: Path) -> None:
    apply(fake_engine)
    for f in fake_engine.rglob("*.py"):
        compile(f.read_text(encoding="utf-8"), str(f), "exec"), f


def test_guard_content(fake_engine: Path) -> None:
    apply(fake_engine)
    text = (fake_engine / "GPT_SoVITS/s2_train.py").read_text(encoding="utf-8")
    assert "ZLUDA_MIOpen_GUARD" in text
    assert "ZLUDA_CPU_FFT_GUARD" in text
    assert 'os.environ.get("ZLUDA_MODE") == "1"' in text
    assert "MIOpen.dll" in text
    assert "torch.stft = _zluda_fft_wrap(torch.stft)" in text
    # 守卫块必须紧跟在 import torch 之后（os 在整合包脚本里总是先导入）
    lines = text.splitlines()
    i = next(i for i, l in enumerate(lines) if l.strip() == "import torch")
    assert lines[i + 1].strip().startswith('if os.environ.get("ZLUDA_MODE")'), \
        "守卫应插在 import torch 的下一行"


def test_g2pw_multi_line_form(tmp_path: Path) -> None:
    eng = tmp_path / "eng"
    (eng / "GPT_SoVITS/text/g2pw").mkdir(parents=True)
    (eng / "api_v2.py").write_text("# entry\n", encoding="utf-8")
    (eng / "GPT_SoVITS/text/g2pw/onnx_api.py").write_text(
        G2PW_MULTI, encoding="utf-8")
    apply(eng)
    text = (eng / "GPT_SoVITS/text/g2pw/onnx_api.py").read_text(encoding="utf-8")
    compile(text, "onnx_api.py", "exec")
    assert 'and os.environ.get("ZLUDA_MODE") != "1"' in text


def test_missing_entry_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        apply(tmp_path)


def test_absent_targets_skipped(tmp_path: Path) -> None:
    """版本差异：目标脚本不存在时跳过而不是报错。"""
    eng = tmp_path / "eng"
    (eng / "GPT_SoVITS/text/g2pw").mkdir(parents=True)
    (eng / "api_v2.py").write_text("# entry\n", encoding="utf-8")
    (eng / "GPT_SoVITS/text/g2pw/onnx_api.py").write_text(
        G2PW_SINGLE, encoding="utf-8")
    done = apply(eng)
    assert done == ["g2pw 走 CPU EP（ORT CUDA EP 在 ZLUDA 下会崩）"]


# 旧版补丁升级 -----------------------------------------------------------------
def test_upgrade_old_unconditional_block(fake_engine: Path) -> None:
    """v0.2.6 及更早打的「无条件禁 cuDNN」两行块，应被升级为条件守卫。"""
    rel = "GPT_SoVITS/s2_train.py"
    f = fake_engine / rel
    old = (
        "import os\nimport torch\n"
        'if os.environ.get("ZLUDA_MODE") == "1":\n'
        "    torch.backends.cudnn.enabled = False  "
        "# ZLUDA: 官方 HIP SDK 无 MIOpen，禁用 cuDNN 走原生卷积\n"
        "print('ok')\n"
    )
    f.write_text(old, encoding="utf-8")
    done = apply(fake_engine)
    assert any("s2 训练" in d for d in done)
    text = f.read_text(encoding="utf-8")
    assert "ZLUDA_CPU_FFT_GUARD" in text
    assert "禁用 cuDNN 走原生卷积" not in text, "旧注释应被移除"
    assert text.count('if os.environ.get("ZLUDA_MODE") == "1":') == 1, \
        "旧 if 块应被整体替换而不是叠加"
    assert "print('ok')" in text
    compile(text, rel, "exec")
    assert apply(fake_engine) == []


def test_upgrade_v1_miopen_block(fake_engine: Path) -> None:
    """v1 多行 MIOpen 守卫（无 FFT 部分）也应整体升级为 v2。"""
    rel = "GPT_SoVITS/s2_train.py"
    f = fake_engine / rel
    v1 = (
        "import os\nimport torch\n"
        'if os.environ.get("ZLUDA_MODE") == "1":\n'
        '    # ZLUDA_MIOpen_GUARD: 有 MIOpen.dll 则保持 cuDNN 加速，否则禁用走原生卷积\n'
        '    _zdirs = [p for p in os.environ.get("PATH", "").split(";") if p]\n'
        '    _zhip = os.environ.get("HIP_PATH")\n'
        "    if _zhip:\n"
        '        _zdirs += [os.path.join(_zhip, "bin"), _zhip]\n'
        '    if not any(os.path.isfile(os.path.join(d, "MIOpen.dll")) for d in _zdirs):\n'
        "        torch.backends.cudnn.enabled = False\n"
        "print('keep')\n"
    )
    f.write_text(v1, encoding="utf-8")
    apply(fake_engine)
    text = f.read_text(encoding="utf-8")
    assert "ZLUDA_CPU_FFT_GUARD" in text
    assert text.count('if os.environ.get("ZLUDA_MODE") == "1":') == 1
    assert "print('keep')" in text
    compile(text, rel, "exec")


# device.zluda_env -------------------------------------------------------------
def _mk_zluda_dirs(tmp_path: Path) -> tuple[Path, Path]:
    zluda = tmp_path / "zluda"
    hip = tmp_path / "hip"
    zluda.mkdir()
    (hip / "bin").mkdir(parents=True)
    (zluda / "nvcuda.dll").write_bytes(b"x")
    (hip / "bin" / "amdhip64_7.dll").write_bytes(b"x")
    return zluda, hip


class _Cfg:
    def __init__(self, data: dict) -> None:
        self.data = data

    def __getitem__(self, k):
        return self.data.get(k)


def test_zluda_env_off_is_noop() -> None:
    assert zluda_env(_Cfg({"zluda_mode": False})) == {}


def test_zluda_env_validation(tmp_path: Path) -> None:
    cfg = _Cfg({"zluda_mode": True, "zluda_dir": "", "hip_path": ""})
    with pytest.raises(DeviceError):
        zluda_env(cfg)
    zluda, hip = _mk_zluda_dirs(tmp_path)
    cfg = _Cfg({"zluda_mode": True, "zluda_dir": str(zluda),
                "hip_path": str(hip)})
    env = zluda_env(cfg)
    assert env["ZLUDA_MODE"] == "1"
    assert env["HIP_PATH"] == str(hip)
    assert env["PATH"].startswith(str(zluda))
    # 目录残缺要报得明确
    (zluda / "nvcuda.dll").unlink()
    with pytest.raises(DeviceError):
        zluda_env(cfg)


# trainer 2-sv 产物守卫 --------------------------------------------------------
def test_verify_sv_guard(tmp_path: Path) -> None:
    from echosmith.trainer import TrainError, Trainer

    list_path = tmp_path / "ds.list"
    list_path.write_text(
        "a.wav|spk|ZH|x\nb.wav|spk|ZH|y\nc.wav|spk|ZH|z\n", encoding="utf-8")
    opt = tmp_path / "opt"
    (opt / "7-sv_cn").mkdir(parents=True)
    # 全军覆没（GPU 卷积失败被逐行吞掉）→ 必须拦下
    with pytest.raises(TrainError):
        Trainer._verify_sv(opt, list_path)
    # 部分缺失也要拦
    (opt / "7-sv_cn" / "a.pt").write_bytes(b"x")
    with pytest.raises(TrainError):
        Trainer._verify_sv(opt, list_path)
    # 齐了放行
    (opt / "7-sv_cn" / "b.pt").write_bytes(b"x")
    (opt / "7-sv_cn" / "c.pt").write_bytes(b"x")
    Trainer._verify_sv(opt, list_path)


# torch/lib DLL 垫片部署/还原 --------------------------------------------------
def _mk_shim_env(tmp_path: Path) -> tuple[Path, Path]:
    zluda = tmp_path / "zluda"
    zluda.mkdir()
    (zluda / "nvcuda.dll").write_bytes(b"drv")
    (zluda / "cufft64_10.dll").write_bytes(b"SHIM")
    (zluda / "cusparse64_12.dll").write_bytes(b"S12")
    eng = tmp_path / "eng"
    lib = eng / "runtime" / "Lib" / "site-packages" / "torch" / "lib"
    lib.mkdir(parents=True)
    (eng / "api_v2.py").write_text("# entry\n", encoding="utf-8")
    (lib / "cufft64_10.dll").write_bytes(b"REAL")
    return eng, zluda


def test_shim_deploy_and_restore(tmp_path: Path) -> None:
    eng, zluda = _mk_shim_env(tmp_path)
    lib = eng / "runtime" / "Lib" / "site-packages" / "torch" / "lib"

    done = deploy_runtime_shims(eng, zluda)
    assert done == ["cufft64_10.dll"], "torch/lib 没有同名 cusparse64_12 → 不部署"
    assert (lib / "cufft64_10.dll").read_bytes() == b"SHIM"
    backup = lib / "_zluda_nvidia_backup" / "cufft64_10.dll"
    assert backup.read_bytes() == b"REAL"

    assert deploy_runtime_shims(eng, zluda) == [], "幂等：已是垫片不再动"

    done = restore_runtime_shims(eng)
    assert done == ["cufft64_10.dll"]
    assert (lib / "cufft64_10.dll").read_bytes() == b"REAL"
    assert restore_runtime_shims(eng) == []
    assert not (lib / "_zluda_shims_state.json").exists()


def test_shim_deploy_rejects_bad_dirs(tmp_path: Path) -> None:
    import shutil

    eng, zluda = _mk_shim_env(tmp_path)
    with pytest.raises(ValueError):
        deploy_runtime_shims(eng, tmp_path / "nope")  # 无 nvcuda.dll
    shutil.rmtree(eng / "runtime" / "Lib" / "site-packages" / "torch")
    with pytest.raises(ValueError):
        deploy_runtime_shims(eng, zluda)  # torch/lib 不存在
