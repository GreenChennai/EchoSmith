"""GPT-SoVITS 训练编排：prepare_datasets×3 → s2_train(SoVITS) → s1_train(GPT)。

复刻官方 webui 的调用契约（子进程 + 环境变量传参，详见各阶段 env 构造），
CPU 模式（is_half=false，单进程）。预训练权重按引擎版本自动探测。
属「需真机联调」路径：各阶段脚本缺失/环境变量契约变化都会明确报错。
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
import threading
from pathlib import Path

from .config import Config
from .downloader import valid_engine
from .state import Shared

WIN_SILENT = 0x08000000 if sys.platform == "win32" else 0


class TrainError(RuntimeError):
    pass


def _glob_first(engine: Path, patterns: list[str]) -> Path | None:
    for pat in patterns:
        hits = sorted(glob.glob(str(engine / pat)))
        if hits:
            return Path(hits[0])
    return None


def detect_pretrains(engine: Path) -> dict[str, Path]:
    """按引擎版本探测预训练权重；缺关键项抛错。"""
    got: dict[str, Path] = {}
    for key, pats in {
        "s2G": ["pretrained_models/gsv-v2pro-pretrained/s2G*.pth",
                "pretrained_models/gsv-v2final-pretrained/s2G*.pth",
                "pretrained_models/s2G488k.pth"],
        "s2D": ["pretrained_models/gsv-v2pro-pretrained/s2D*.pth",
                "pretrained_models/gsv-v2final-pretrained/s2D*.pth",
                "pretrained_models/s2D488k.pth"],
        "s1": ["pretrained_models/gsv-v2final-pretrained/s1bert*.ckpt",
               "pretrained_models/s1bert*.ckpt"],
        "bert": ["pretrained_models/chinese-roberta-wwm-ext-large"],
        "hubert": ["pretrained_models/chinese-hubert-base"],
    }.items():
        p = _glob_first(engine, pats)
        if p is None:
            raise TrainError(f"未找到预训练权重（{key}）：请检查整合包 pretrained_models/ 是否完整")
        got[key] = p
    if not (engine / "GPT_SoVITS" / "configs" / "s2.json").is_file():
        raise TrainError("缺少 GPT_SoVITS/configs/s2.json")
    return got


class Trainer(threading.Thread):
    """五阶段顺序执行；遇错即停；cancel() 终止当前子进程。"""

    STAGES = [
        ("1-text", "GPT_SoVITS/prepare_datasets/1-get-text.py"),
        ("2-hubert-wav32k", "GPT_SoVITS/prepare_datasets/2-get-hubert_and_wav32k.py"),
        ("3-semantic", "GPT_SoVITS/prepare_datasets/3-get-semantic.py"),
        ("4-sovits", "GPT_SoVITS/s2_train.py"),
        ("5-gpt", "GPT_SoVITS/s1_train.py"),
    ]

    def __init__(self, cfg: Config, shared: Shared, list_path: Path, exp: str,
                 stages: list[str] | None = None) -> None:
        super().__init__(name="echosmith-train", daemon=True)
        self.cfg = cfg
        self.shared = shared
        self.list_path = list_path
        self.exp = exp
        self.only = set(stages) if stages else None
        self.cancelled = threading.Event()
        self.proc: subprocess.Popen | None = None

    def cancel(self) -> None:
        self.cancelled.set()
        p = self.proc
        if p is not None and p.poll() is None:
            try:
                p.terminate()
            except OSError:
                pass

    # ---------------------------------------------------------------- run
    def run(self) -> None:
        try:
            self._run()
        except (TrainError, OSError) as e:
            self.shared.log(f"[训练] 失败：{e}")
            self.shared.set_tr("失败", "遇错停止")
            self.shared.set_error(str(e))

    def _run(self) -> None:
        engine = self.cfg.engine_path
        py = valid_engine(engine)
        if py is None:
            raise TrainError("引擎未就绪，请先完成整合包下载/导入")
        pre = detect_pretrains(engine)
        opt_dir = engine / "logs" / self.exp
        opt_dir.mkdir(parents=True, exist_ok=True)
        if not self.list_path.is_file():
            raise TrainError(f"训练清单不存在：{self.list_path}")

        s2cfg = engine / "GPT_SoVITS" / "configs" / "s2.json"
        s1cfg_candidates = [engine / "GPT_SoVITS" / "configs" / "s1longer-v2.yaml",
                            engine / "GPT_SoVITS" / "configs" / "s1longer.yaml"]
        s1cfg = next((c for c in s1cfg_candidates if c.is_file()), s1cfg_candidates[-1])

        base = {
            "inp_text": str(self.list_path),
            "inp_wav_dir": "",
            "exp_name": self.exp,
            "opt_dir": str(opt_dir),
            "bert_path": str(pre["bert"]),
            "cnhubert_base_path": str(pre["hubert"]),
            "is_half": "false",
        }
        envs: dict[str, dict[str, str]] = {
            "1-text": dict(base),
            "2-hubert-wav32k": {k: base[k] for k in
                                ("inp_text", "inp_wav_dir", "exp_name", "opt_dir",
                                 "cnhubert_base_path", "is_half")},
            "3-semantic": {**base, "s2config_path": str(s2cfg), "s2model_path": str(pre["s2G"])},
            "4-sovits": {
                "exp_name": self.exp, "gpu_numbers1Ba": "0",
                "batch_size": str(self.cfg["batch_size"]),
                "total_epoch": str(self.cfg["s2_total_epoch"]),
                "text_low_lr_rate": str(self.cfg["text_low_lr_rate"]),
                "if_save_latest": "1", "if_save_every_weights": "1",
                "save_every_epoch": str(self.cfg["save_every_epoch"]),
                "if_freeze": "0", "version": self._engine_version(engine),
                "opt_dir": str(opt_dir), "pretrained_s2G": str(pre["s2G"]),
                "pretrained_s2D": str(pre["s2D"]), "s2config_path": str(s2cfg),
                "is_half": "false",
            },
            "5-gpt": {
                "exp_name": self.exp, "gpu_numbers1B2": "0",
                "total_epoch": str(self.cfg["s1_total_epoch"]),
                "if_save_latest": "1", "if_save_every_weights": "1",
                "save_every_epoch": str(self.cfg["save_every_epoch"]),
                "if_dpo": "0", "opt_dir": str(opt_dir),
                "s1pretrained_path": str(pre["s1"]), "s1config_path": str(s1cfg),
                "is_half": "false",
            },
        }

        runnable = [(s, f) for s, f in self.STAGES
                    if self.only is None or s in self.only]
        for i, (stage, script) in enumerate(runnable, 1):
            if self.cancelled.is_set():
                self.shared.set_tr("已取消", "停止")
                return
            full = engine / script
            if not full.is_file():
                raise TrainError(f"引擎缺少训练脚本：{script}（版本不匹配？）")
            self.shared.set_tr(stage, f"阶段 {i}/{len(runnable)}", i - 1, len(runnable))
            self.shared.log(f"[训练] 阶段 {i}/{len(runnable)}：{stage}（{script}）")
            env = {**os.environ, **{k: str(v) for k, v in envs[stage].items()}}
            self.proc = subprocess.Popen(
                [str(py), script], cwd=str(engine), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                creationflags=WIN_SILENT,
            )
            for raw in self.proc.stdout:  # type: ignore[union-attr]
                line = raw.decode("utf-8", "replace").rstrip()
                if line.strip():
                    self.shared.log(f"[训练:{stage}] {line[-200:]}")
            rc = self.proc.wait()
            self.proc = None
            if self.cancelled.is_set():
                self.shared.set_tr("已取消", "停止")
                return
            if rc != 0:
                raise TrainError(f"阶段 {stage} 退出码 {rc}（详见上方日志）")
            self.shared.set_tr(stage, "完成", i, len(runnable))

        # 产物
        w2 = self._newest(opt_dir / "SoVITS_weights_v2", "*.pth") or \
            self._newest(opt_dir / "SoVITS_weights", "*.pth")
        w1 = self._newest(opt_dir / "GPT_weights_v2", "*.ckpt") or \
            self._newest(opt_dir / "GPT_weights", "*.ckpt")
        if w2:
            self.shared.log(f"[训练] SoVITS 权重：{w2}")
        if w1:
            self.shared.log(f"[训练] GPT 权重：{w1}")
        self.shared.set_tr("完成", "训练结束，可在「声线卡」页打包")

    @staticmethod
    def _engine_version(engine: Path) -> str:
        if (engine / "pretrained_models" / "gsv-v2pro-pretrained").is_dir():
            return "v2Pro"
        if (engine / "pretrained_models" / "gsv-v2final-pretrained").is_dir():
            return "v2"
        return "v1"

    @staticmethod
    def _newest(d: Path, pat: str) -> Path | None:
        if not d.is_dir():
            return None
        hits = sorted(d.glob(pat), key=lambda p: p.stat().st_mtime)
        return hits[-1] if hits else None
