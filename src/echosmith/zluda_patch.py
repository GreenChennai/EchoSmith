"""ZLUDA 兼容补丁：让 GPT-SoVITS 整合包引擎能在 AMD 显卡上跑。

所有补丁都用 ``ZLUDA_MODE`` 环境变量守卫——不启用 ZLUDA 时行为与官方整合包
完全一致，因此可以常驻，不影响 CPU 训练/合成，无需回退。

补丁内容：

1. **g2pw 文本前端强制 CPU EP**：ONNX Runtime 的 CUDAExecutionProvider 在 ZLUDA
   下初始化必崩（ORT 不认 ZLUDA 转发的 cuDNN/cuBLAS）。g2pw 只是个小的
   多音字消歧模型，CPU 推理开销可忽略。
2. **ZLUDA 下按 MIOpen 可用性守卫 cuDNN**：ZLUDA 的 cuDNN 转发依赖 HIP SDK 里的
   MIOpen.dll，官方 HIP SDK 并不附带它（见 ZLUDA 文档 hip_sdk.html），缺的时候
   卷积直接挂或（2-get-sv 等）被逐行 try/except 吞掉造成静默废数据。守卫在
   ``PATH`` / ``HIP_PATH`` 下探测 MIOpen.dll：没有 → 禁用 cuDNN 走 torch 原生卷积；
   有（TheRock nightly 等 ML 构建）→ 保持 cuDNN 加速。

覆盖的脚本（进程边界即补丁边界，每个子进程都要有守卫）：
  训练编排   1-text 用 BERT（纯 matmul，不需要）、2-hubert / 2-sv / 3-semantic
             （卷积）、s2_train / s2_train_v3_lora / s1_train；
  推理引擎   TTS_infer_pack/TTS.py（api_v2.py 自身不 import torch，TTS 类在此）；
  人声分离   tools/uvr5/vr.py。

3. **torch/lib DLL 垫片部署（``deploy_runtime_shims``）**：torch 自带的 CUDA 运行库
   里，cuFFT/cuDNN 等只带 NVIDIA cubin（无 PTX），真 DLL 在 ZLUDA 上无法加载内核
   （``CUFFT_INTERNAL_ERROR`` / cuDNN 报错）；ZLUDA 同名垫片 DLL 会转译到 HIP 侧。
   按「同名才替换」匹配 torch/lib 里的 CUDA DLL，原版备份进
   ``_zluda_nvidia_backup/``，可随时 ``restore_runtime_shims`` 还原（关 GPU 时）。

本模块幂等：重复调用不会重复插入；旧版「无条件禁 cuDNN」补丁会自动升级为
MIOpen 条件守卫。
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_NEED_OS = "import os\n"

_BACKUP_DIR_NAME = "_zluda_nvidia_backup"
_STATE_FILE_NAME = "_zluda_shims_state.json"

# 旧版守卫片段（无条件禁 cuDNN 的两行块、v1 MIOpen 守卫、v2 FFT 守卫）
# ——发现即整体摘除，升级为 _CUDNN_GUARD
_OLD_CUDNN_BLOCK = re.compile(
    r'^if os\.environ\.get\("ZLUDA_MODE"\) == "1":\n'
    r"(?:[ \t]+.*\n?)*?"
    r"(?:[ \t]+torch\.istft = _zluda_fft_wrap\(torch\.istft\)\n"
    r"|[ \t]+torch\.backends\.cudnn\.enabled = False.*\n)",
    re.MULTILINE,
)

_CUDNN_GUARD = '''if os.environ.get("ZLUDA_MODE") == "1":
    # ZLUDA_MIOpen_GUARD: 有 MIOpen.dll 则保持 cuDNN 加速，否则禁用走原生卷积
    _zdirs = [p for p in os.environ.get("PATH", "").split(";") if p]
    _zhip = os.environ.get("HIP_PATH")
    if _zhip:
        _zdirs += [os.path.join(_zhip, "bin"), _zhip]
    if not any(os.path.isfile(os.path.join(d, "MIOpen.dll")) for d in _zdirs):
        torch.backends.cudnn.enabled = False
    if not os.environ.get("ZLUDA_NATIVE_FFT") == "1":
        # ZLUDA_CPU_FFT_GUARD: ZLUDA 的 cuFFT 转译未实现（全类型 NOT_SUPPORTED），
        # fft/stft 包一层 CPU 往返（autograd 不断，训练/推理均可）
        def _zluda_fft_wrap(fn):
            def _w(x, *a, **k):
                if torch.is_tensor(x) and x.is_cuda:
                    dev = x.device
                    return fn(x.cpu(), *a, **k).to(dev)
                return fn(x, *a, **k)
            return _w
        for _n in ("fft", "ifft", "rfft", "irfft", "fft2", "ifft2", "rfft2",
                   "irfft2", "hfft", "ihfft", "fftn", "ifftn"):
            setattr(torch.fft, _n, _zluda_fft_wrap(getattr(torch.fft, _n)))
        torch.stft = _zluda_fft_wrap(torch.stft)
        torch.istft = _zluda_fft_wrap(torch.istft)
        # ZLUDA_COMPLEX_ABS_GUARD: CUDA 复数 abs 走 jiterator 现场 NVRTC 编内核，
        # 旧 NVRTC 不认 ZLUDA 报告的架构 —— 复数 abs 同样 CPU 往返
        def _zluda_cplx_wrap(fn):
            def _w(x, *a, **k):
                if torch.is_tensor(x) and x.is_cuda and x.is_complex():
                    dev = x.device
                    return fn(x.cpu(), *a, **k).to(dev)
                return fn(x, *a, **k)
            return _w
        setattr(torch.Tensor, "abs", _zluda_cplx_wrap(torch.Tensor.abs))
        torch.abs = _zluda_cplx_wrap(torch.abs)
'''


@dataclass(frozen=True)
class _Patch:
    """一个补丁：在 anchor 行之后插入 snippet（幂等判定看 marker）。"""

    rel: str            # 相对 engine_dir 的路径
    anchor: str         # 锚点文本（首次出现处之后插入）
    snippet: str        # 要插入的代码
    marker: str         # 已打补丁的判定标记
    desc: str           # 人类可读描述


_G2PW = _Patch(
    rel="GPT_SoVITS/text/g2pw/onnx_api.py",
    anchor='"CUDAExecutionProvider" in onnxruntime.get_available_providers()',
    snippet='            and os.environ.get("ZLUDA_MODE") != "1"\n',
    marker='ZLUDA_MODE") != "1"',
    desc="g2pw 走 CPU EP（ORT CUDA EP 在 ZLUDA 下会崩）",
)


def _cudnn(rel: str, desc: str) -> _Patch:
    return _Patch(
        rel=rel,
        anchor="import torch",
        snippet=_CUDNN_GUARD,
        marker="ZLUDA_COMPLEX_ABS_GUARD",
        desc=desc,
    )


_PATCHES = (
    _G2PW,
    # ---- 训练编排（有卷积的 prepare 阶段 + 三个训练脚本） ----
    _cudnn("GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py", "2-hubert 阶段 MIOpen 守卫"),
    _cudnn("GPT_SoVITS/prepare_datasets/2-get-sv.py", "2-sv 阶段 MIOpen 守卫"),
    _cudnn("GPT_SoVITS/prepare_datasets/3-get-semantic.py", "3-semantic 阶段 MIOpen 守卫"),
    _cudnn("GPT_SoVITS/s2_train.py", "s2 训练 MIOpen 守卫"),
    _cudnn("GPT_SoVITS/s2_train_v3_lora.py", "s2 训练(v3 lora) MIOpen 守卫"),
    _cudnn("GPT_SoVITS/s1_train.py", "s1 训练 MIOpen 守卫"),
    # ---- 推理引擎 / 人声分离 ----
    _cudnn("GPT_SoVITS/TTS_infer_pack/TTS.py", "TTS 推理 MIOpen 守卫"),
    _cudnn("tools/uvr5/vr.py", "UVR5 分离 MIOpen 守卫"),
)


def _patch_g2pw(text: str, anchor: str) -> str | None:
    """给 g2pw 的 CUDA provider 判定加 ``and ZLUDA_MODE != "1"``。

    官方整合包有两种写法，都要能处理：

    - 单行：``if "CUDAExecutionProvider" in onnxruntime.get_available_providers():``
      → 改写成多行括号形式再加条件（单行没法直接插 and）
    - 多行：``if (\\n    "CUDAExecutionProvider" in ...\\n):``
      → 在条件行后插入一个 and 行

    返回 None 表示锚点没找到。
    """
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if anchor not in line:
            continue
        indent = line[: len(line) - len(line.lstrip())]
        body = line.lstrip()

        # 情况 1：单行 if COND:
        if body.startswith("if ") and body.rstrip().endswith(":"):
            cond = body[3:].rstrip()[:-1].strip()  # 去掉 "if " 和末尾 ":"
            lines[i] = (
                f"{indent}if (\n"
                f"{indent}    {cond}\n"
                f'{indent}    and os.environ.get("ZLUDA_MODE") != "1"\n'
                f"{indent}):\n"
            )
            return "".join(lines)

        # 情况 2：多行括号内的一行条件
        lines.insert(
            i + 1, f'{indent}and os.environ.get("ZLUDA_MODE") != "1"\n')
        return "".join(lines)

    return None


def _upgrade_old_cudnn(text: str) -> str:
    """旧版补丁（无条件禁 cuDNN）→ 新版 MIOpen 条件守卫。

    旧片段是本模块自己写入的两行 if 块，整块删除后按常规流程重打。
    """
    new_text, n = _OLD_CUDNN_BLOCK.subn("", text)
    if n:
        logger.info("升级旧版 cuDNN 补丁（%d 处）→ MIOpen 条件守卫", n)
    return new_text


def engine_has_entry(engine_dir: Path) -> bool:
    """判断这是否像一个 GPT-SoVITS 整合包目录。"""
    return (Path(engine_dir) / "api_v2.py").is_file()


def check(engine_dir: Path) -> tuple[list[_Patch], list[_Patch]]:
    """返回 (已应用的补丁, 缺失的补丁)。"""
    applied: list[_Patch] = []
    missing: list[_Patch] = []
    for p in _PATCHES:
        f = Path(engine_dir) / p.rel
        if not f.is_file():
            continue  # 版本差异，没有这个文件就跳过
        text = f.read_text(encoding="utf-8", errors="replace")
        (applied if p.marker in text else missing).append(p)
    return applied, missing


def apply(engine_dir: Path) -> list[str]:
    """应用所有缺失的补丁，返回本次实际打上的描述列表（幂等）。"""
    if not engine_has_entry(engine_dir):
        raise ValueError(f"不像 GPT-SoVITS 整合包目录（缺 api_v2.py）：{engine_dir}")

    done: list[str] = []
    for p in _PATCHES:
        f = Path(engine_dir) / p.rel
        if not f.is_file():
            logger.debug("补丁目标不存在，跳过：%s", p.rel)
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        if p.marker in text:
            continue  # 已打过（新版守卫就位）
        if p is not _G2PW:
            text = _upgrade_old_cudnn(text)  # 旧版无条件禁用 → 条件守卫

        # g2pw 补丁是往 if 条件里加一个 and，官方包有单行 / 多行括号两种写法
        if p is _G2PW:
            text = _patch_g2pw(text, p.anchor)
            if text is None:
                logger.warning("g2pw 补丁锚点未找到，跳过")
                continue
        else:
            lines = text.splitlines(keepends=True)
            hit = -1
            for i, line in enumerate(lines):
                if line.strip() == p.anchor:
                    hit = i
                    break
            if hit < 0:
                logger.warning("补丁锚点未找到，跳过：%s", p.rel)
                continue
            # 确保 os 已导入（补丁要用 os.environ）
            if not any(l.strip() == "import os" for l in lines[:60]):
                lines.insert(hit, _NEED_OS)
                hit += 1
            lines.insert(hit + 1, p.snippet)
            text = "".join(lines)

        f.write_text(text, encoding="utf-8")
        done.append(p.desc)
        logger.info("已打 ZLUDA 补丁：%s", p.rel)

    return done


# ---------------------------------------------------------------- torch/lib 垫片
def _torch_lib(engine_dir: Path) -> Path:
    return Path(engine_dir) / "runtime" / "Lib" / "site-packages" / "torch" / "lib"


def deploy_runtime_shims(engine_dir: Path, zluda_dir: Path) -> list[str]:
    """把 ZLUDA 垫片 DLL 部署进引擎 torch/lib（幂等，原版进备份目录）。

    只替换「ZLUDA 目录与 torch/lib 同名」的 CUDA 运行库（同名即同版本 ABI）：
    torch cu118 → *_64_11 / cudnn64_8 / cufft64_10。cuFFT/cuDNN 的官方 DLL 只带
    NVIDIA cubin，在 ZLUDA 上无法加载（CUFFT_INTERNAL_ERROR 等），必须用垫片；
    已经是垫片字节的跳过。返回本次实际替换的名字列表。
    """
    lib = _torch_lib(engine_dir)
    if not lib.is_dir():
        raise ValueError(f"引擎 torch/lib 不存在：{lib}")
    zluda_dir = Path(zluda_dir)
    if not (zluda_dir / "nvcuda.dll").is_file():
        raise ValueError(f"ZLUDA 目录无效（缺 nvcuda.dll）：{zluda_dir}")

    backup = lib / _BACKUP_DIR_NAME
    state_f = lib / _STATE_FILE_NAME
    state: dict = {}
    if state_f.is_file():
        try:
            state = json.loads(state_f.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            state = {}
    deployed: dict[str, bool] = state.get("deployed", {})

    done: list[str] = []
    for dll in sorted(zluda_dir.glob("*.dll")):
        target = lib / dll.name
        if not target.is_file():  # torch/lib 没有同名库 → 版本不符，不部署
            continue
        if deployed.get(dll.name):
            continue  # 本模块部署过且未还原
        if target.read_bytes() == dll.read_bytes():
            continue  # 已是垫片（如老版手工拷过），原样认可
        backup.mkdir(exist_ok=True)
        orig = backup / dll.name
        if not orig.is_file():
            orig.write_bytes(target.read_bytes())
        try:
            target.write_bytes(dll.read_bytes())
        except OSError as e:
            raise ValueError(
                f"替换 {dll.name} 失败（{e}）；请先停止引擎/训练等占用进程后重试"
            ) from e
        deployed[dll.name] = True
        done.append(dll.name)
        logger.info("已部署 ZLUDA 垫片：%s", dll.name)

    state["deployed"] = deployed
    state_f.write_text(json.dumps(state, indent=1), encoding="utf-8")
    return done


def restore_runtime_shims(engine_dir: Path) -> list[str]:
    """还原 deploy_runtime_shims 换掉的 DLL（关闭 ZLUDA / 交还 N 卡时用）。"""
    lib = _torch_lib(engine_dir)
    state_f = lib / _STATE_FILE_NAME
    if not state_f.is_file():
        return []
    try:
        state = json.loads(state_f.read_text("utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    backup = lib / _BACKUP_DIR_NAME
    done: list[str] = []
    for name, was in state.get("deployed", {}).items():
        orig = backup / name
        target = lib / name
        if was and orig.is_file() and target.is_file():
            try:
                target.write_bytes(orig.read_bytes())
            except OSError as e:
                raise ValueError(
                    f"还原 {name} 失败（{e}）；请先停止占用引擎的进程后重试"
                ) from e
            orig.unlink(missing_ok=True)
            done.append(name)
            logger.info("已还原 NVIDIA 原版 DLL：%s", name)
    state_f.unlink(missing_ok=True)
    return done
