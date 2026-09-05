"""ZLUDA 兼容补丁：让 GPT-SoVITS 整合包引擎能在 AMD 显卡上跑。

所有补丁都用 ``ZLUDA_MODE`` 环境变量守卫——不启用 ZLUDA 时行为与官方整合包
完全一致，因此可以常驻，不影响 CPU 训练/合成，无需回退。

补丁内容：

1. **g2pw 文本前端强制 CPU EP**：ONNX Runtime 的 CUDAExecutionProvider 在 ZLUDA
   下初始化必崩（ORT 不认 ZLUDA 转发的 cuDNN/cuBLAS）。g2pw 只是个小的
   多音字消歧模型，CPU 推理开销可忽略。
2. **训练脚本禁用 cuDNN**：ZLUDA 的 cuDNN 转发依赖 MIOpen.dll，而 AMD 官方
   HIP SDK 并不附带它，卷积会直接挂。禁用后 torch 走原生卷积实现。

本模块幂等：重复调用不会重复插入。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_NEED_OS = "import os\n"

_CUDNN_OFF = (
    'if os.environ.get("ZLUDA_MODE") == "1":\n'
    "    torch.backends.cudnn.enabled = False  "
    "# ZLUDA: 官方 HIP SDK 无 MIOpen，禁用 cuDNN 走原生卷积\n"
)


@dataclass(frozen=True)
class _Patch:
    """一个补丁：在 anchor 行之后插入 snippet（幂等判定看 marker）。"""

    rel: str            # 相对 engine_dir 的路径
    anchor: str         # 锚点文本（首次出现处之后插入）
    snippet: str        # 要插入的代码
    marker: str         # 已打补丁的判定标记
    desc: str           # 人类可读描述


_PATCHES = (
    _Patch(
        rel="GPT_SoVITS/text/g2pw/onnx_api.py",
        anchor='"CUDAExecutionProvider" in onnxruntime.get_available_providers()',
        snippet='            and os.environ.get("ZLUDA_MODE") != "1"\n',
        marker='ZLUDA_MODE") != "1"',
        desc="g2pw 走 CPU EP（ORT CUDA EP 在 ZLUDA 下会崩）",
    ),
    _Patch(
        rel="GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py",
        anchor="import torch",
        snippet=_CUDNN_OFF,
        marker="torch.backends.cudnn.enabled = False",
        desc="2-hubert 阶段禁用 cuDNN",
    ),
    _Patch(
        rel="GPT_SoVITS/s1_train.py",
        anchor="import torch",
        snippet=_CUDNN_OFF,
        marker="torch.backends.cudnn.enabled = False",
        desc="s1 训练禁用 cuDNN",
    ),
    _Patch(
        rel="GPT_SoVITS/s2_train.py",
        anchor="import torch",
        snippet=_CUDNN_OFF,
        marker="torch.backends.cudnn.enabled = False",
        desc="s2 训练禁用 cuDNN",
    ),
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
            continue  # 已打过

        # g2pw 补丁是往 if 条件里加一个 and，官方包有单行 / 多行括号两种写法
        if p.rel.endswith("onnx_api.py"):
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
