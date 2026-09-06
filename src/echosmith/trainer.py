"""GPT-SoVITS 训练编排（契约已对照官方源码核验，RVC-Boss/GPT-SoVITS main 分支）。

流程与 webui.py 完全一致：
  1-get-text.py → [merge 2-name2text-{i}.txt → 2-name2text.txt]
  2-get-hubert-wav32k.py →（v2Pro/v2ProPlus/v3/v4 额外 2-get-sv.py）
  3-get-semantic.py → [merge 6-name2semantic-{i}.tsv → 6-name2semantic.tsv]
  s2_train.py --config TEMP/tmp_s2.json（由 GPT_SoVITS/configs/s2*.json 改写）
  s1_train.py --config_file TEMP/tmp_s1.yaml（由 configs/s1longer-v2.yaml 改写）

prepare 阶段用环境变量（i_part=0 / all_parts=1 / _CUDA_VISIBLE_DEVICES=0 / is_half=False），
训练阶段用配置文件；CPU 模式下 fp16_run=False / precision="32"、batch_size 减半（与 webui 相同）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

from .config import Config
from .device import zluda_env
from .downloader import valid_engine
from .state import Shared

WIN_SILENT = 0x08000000 if sys.platform == "win32" else 0

# 与引擎根 config.py 同源的映射
SOVITS_WEIGHT_VERSION2ROOT = {
    "v1": "SoVITS_weights", "v2": "SoVITS_weights_v2", "v3": "SoVITS_weights_v3",
    "v4": "SoVITS_weights_v4", "v2Pro": "SoVITS_weights_v2Pro",
    "v2ProPlus": "SoVITS_weights_v2ProPlus",
}
GPT_WEIGHT_VERSION2ROOT = {
    "v1": "GPT_weights", "v2": "GPT_weights_v2", "v3": "GPT_weights_v3",
    "v4": "GPT_weights_v4", "v2Pro": "GPT_weights_v2Pro",
    "v2ProPlus": "GPT_weights_v2ProPlus",
}
S2_CONFIG_BY_VERSION = {
    "v1": "GPT_SoVITS/configs/s2.json", "v2": "GPT_SoVITS/configs/s2.json",
    "v3": "GPT_SoVITS/configs/s2.json", "v4": "GPT_SoVITS/configs/s2.json",
    "v2Pro": "GPT_SoVITS/configs/s2v2Pro.json",
    "v2ProPlus": "GPT_SoVITS/configs/s2v2ProPlus.json",
}
GPT_PRETRAIN_BY_VERSION = {
    "v1": "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
    "v2": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
    "v3": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
    "v4": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
    "v2Pro": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
    "v2ProPlus": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
}
SV_PATH = "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
BERT_DIR = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
HUBERT_DIR = "GPT_SoVITS/pretrained_models/chinese-hubert-base"

# 需要(sv)语音自监督特征的版本（webui: "Pro" in version；v3/v4 同理走 sv）
SV_VERSIONS = {"v2Pro", "v2ProPlus", "v3", "v4"}
# 走 s2_train.py 的版本；v3/v4 官方走 s2_train_v3_lora.py（实验性支持）
S2_SCRIPT_BY_VERSION = {
    "v1": "GPT_SoVITS/s2_train.py", "v2": "GPT_SoVITS/s2_train.py",
    "v2Pro": "GPT_SoVITS/s2_train.py", "v2ProPlus": "GPT_SoVITS/s2_train.py",
    "v3": "GPT_SoVITS/s2_train_v3_lora.py", "v4": "GPT_SoVITS/s2_train_v3_lora.py",
}


class TrainError(RuntimeError):
    pass


# 整合包 runtime 是 embeddable Python（python39._pth），PYTHONPATH 会被无视；
# 用 runpy 启动器把引擎根与 GPT_SoVITS/ 注入 sys.path 后再执行目标脚本。
LAUNCHER_NAME = "_echosmith_run.py"
LAUNCHER_SRC = '''# -*- coding: utf-8 -*-
# EchoSmith 注入的启动器：用法 python _echosmith_run.py <target.py> [args...]
import os
import runpy
import sys

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)
gs = os.path.join(root, "GPT_SoVITS")
if os.path.isdir(gs):
    sys.path.insert(0, gs)
target = sys.argv[1]
sys.argv = [target] + sys.argv[2:]
runpy.run_path(target, run_name="__main__")
'''


def detect_version_and_pretrains(engine: Path) -> dict:
    """按官方 config.py 的命名探测版本与预训练权重。"""
    pm = engine / "GPT_SoVITS" / "pretrained_models"
    candidates = [
        ("v2ProPlus", pm / "v2Pro" / "s2Gv2ProPlus.pth"),
        ("v2Pro", pm / "v2Pro" / "s2Gv2Pro.pth"),
        ("v2", pm / "gsv-v2final-pretrained" / "s2G2333k.pth"),
        ("v1", pm / "s2G488k.pth"),
    ]
    version, s2g = next(((v, p) for v, p in candidates if p.is_file()), (None, None))
    if version is None:
        raise TrainError(
            f"未识别到预训练 SoVITS 底模（{pm} 下缺 s2Gv2Pro*.pth / s2G2333k.pth / s2G488k.pth）"
        )
    s2d = Path(str(s2g).replace("s2G", "s2D"))
    if not s2d.is_file():
        raise TrainError(f"缺少判别器底模：{s2d.name}")
    s1 = engine / GPT_PRETRAIN_BY_VERSION[version]
    if not s1.is_file():
        raise TrainError(f"缺少 GPT 底模：{s1.name}（{version} 版本）")
    bert = engine / BERT_DIR
    hubert = engine / HUBERT_DIR
    if not bert.is_dir():
        raise TrainError(f"缺少 BERT 模型目录：{bert.name}")
    if not hubert.is_dir():
        raise TrainError(f"缺少 Hubert(SSL) 模型目录：{hubert.name}")
    sv = engine / SV_PATH
    if version in SV_VERSIONS and not sv.is_file():
        raise TrainError(f"{version} 版本需要 sv 模型：{sv.name}（整合包应自带）")
    s2config = engine / S2_CONFIG_BY_VERSION[version]
    if not s2config.is_file():
        raise TrainError(f"缺少训练基础配置：{s2config.name}")
    s1config = engine / ("GPT_SoVITS/configs/s1longer.yaml" if version == "v1"
                         else "GPT_SoVITS/configs/s1longer-v2.yaml")
    if not s1config.is_file():
        raise TrainError(f"缺少 GPT 训练基础配置：{s1config.name}")
    return {
        "version": version, "s2G": s2g, "s2D": s2d, "s1": s1,
        "bert": bert, "hubert": hubert, "sv": sv,
        "s2config": s2config, "s1config": s1config,
    }


class Trainer(threading.Thread):
    """顺序五阶段（Pro 版本六阶段）；遇错即停；cancel() 终止当前子进程。"""

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
        except Exception as e:  # 窄异常集兜底：否则线程静默死亡，界面永久卡在阶段上
            self.shared.log(f"[训练] 未预期异常：{type(e).__name__}: {e}")
            self.shared.set_tr("失败", "遇错停止")
            self.shared.set_error(f"{type(e).__name__}: {e}")

    def _run(self) -> None:
        engine = self.cfg.engine_path
        py = valid_engine(engine)
        if py is None:
            raise TrainError("引擎未就绪，请先完成整合包下载/导入")
        pre = detect_version_and_pretrains(engine)
        version = pre["version"]
        # ZLUDA 环境注入 + 兼容补丁兜底（补丁由 ZLUDA_MODE 守卫，幂等，CPU 下零副作用）
        zenv = zluda_env(self.cfg)
        if zenv:
            self.shared.log("[训练] ZLUDA GPU 加速已启用（AMD 经 CUDA 转译）")
            from .zluda_patch import deploy_runtime_shims  # noqa: PLC0415
        from .zluda_patch import apply as apply_zluda  # noqa: PLC0415
        try:
            done = apply_zluda(engine)
            if zenv:
                zluda_dir = Path(str(self.cfg["zluda_dir"]))
                done += [f"垫片 {n}" for n in deploy_runtime_shims(engine, zluda_dir)]
        except ValueError as e:
            raise TrainError(f"ZLUDA 补丁应用失败：{e}") from e
        if done:
            self.shared.log(f"[训练] 已补齐 {len(done)} 项 ZLUDA 兼容补丁")
        opt_dir = engine / "logs" / self.exp
        opt_dir.mkdir(parents=True, exist_ok=True)
        # webui 训练前预创建存档目录（my_save 不会自建）
        (opt_dir / f"logs_s2_{version}").mkdir(parents=True, exist_ok=True)
        (opt_dir / f"logs_s1_{version}").mkdir(parents=True, exist_ok=True)
        temp_dir = engine / "TEMP"
        temp_dir.mkdir(parents=True, exist_ok=True)
        # 子进程 cwd=引擎根，清单必须是绝对路径
        list_path = self.list_path.resolve()
        if not list_path.is_file():
            raise TrainError(f"训练清单不存在：{list_path}")
        self.shared.log(
            f"[训练] 版本 {version} ｜ s2底模 {pre['s2G'].name} ｜ s1底模 {pre['s1'].name}"
        )
        launcher = engine / LAUNCHER_NAME
        launcher.write_text(LAUNCHER_SRC, encoding="utf-8")

        base_env = {
            "inp_text": str(list_path),
            "inp_wav_dir": "",
            "exp_name": self.exp,
            "opt_dir": str(opt_dir),
            "i_part": "0",
            "all_parts": "1",
            "_CUDA_VISIBLE_DEVICES": "0",
            "is_half": "False",
        }
        # 引擎脚本跨目录导入：text/ 在 GPT_SoVITS/ 里，tools/ 在引擎根
        # （prepare 脚本 from text.cleaner / from tools.my_utils；训练脚本 import utils/module）
        path_env = {"PYTHONPATH": str(engine) + os.pathsep + str(engine / "GPT_SoVITS")}

        # (阶段名, 引擎相对脚本, 环境变量, 阶段后合并函数)
        def merge(pattern: str, target: str):
            def _merge() -> None:
                src = opt_dir / pattern.format(0)
                dst = opt_dir / target
                if src.is_file():
                    dst.write_text(src.read_text("utf-8"), encoding="utf-8")
                    src.unlink(missing_ok=True)
            return _merge

        need_sv = version in SV_VERSIONS
        stages: list[tuple[str, str, dict, object]] = [
            ("1-text", "GPT_SoVITS/prepare_datasets/1-get-text.py",
             {**base_env, **path_env, "bert_pretrained_dir": str(pre["bert"])},
             merge("2-name2text-{}.txt", "2-name2text.txt")),
            ("2-hubert-wav32k", "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py",
             {**base_env, **path_env, "cnhubert_base_dir": str(pre["hubert"]),
              "sv_path": str(pre["sv"])},
             None),
        ]
        if need_sv:
            stages.append(
                ("2-sv", "GPT_SoVITS/prepare_datasets/2-get-sv.py",
                 {**base_env, **path_env, "cnhubert_base_dir": str(pre["hubert"]),
                  "sv_path": str(pre["sv"])},
                 None))
        stages.append(
            ("3-semantic", "GPT_SoVITS/prepare_datasets/3-get-semantic.py",
             {**base_env, **path_env, "pretrained_s2G": str(pre["s2G"]),
              "s2config_path": str(pre["s2config"])},
             merge("6-name2semantic-{}.tsv", "6-name2semantic.tsv")))
        stages.append(("4-sovits", S2_SCRIPT_BY_VERSION[version], None, None))
        stages.append(("5-gpt", "GPT_SoVITS/s1_train.py", None, None))

        runnable = [s for s in stages if self.only is None or s[0] in self.only]
        total = len(runnable)
        for i, (stage, script, env, after) in enumerate(runnable, 1):
            if self.cancelled.is_set():
                self.shared.set_tr("已取消", "停止")
                return
            full = engine / script
            if not full.is_file():
                raise TrainError(f"引擎缺少训练脚本：{script}")
            self.shared.set_tr(stage, f"阶段 {i}/{total}", i - 1, total)
            self.shared.log(f"[训练] 阶段 {i}/{total}：{stage}")

            if env is not None:  # prepare 阶段：环境变量契约
                self.proc = subprocess.Popen(
                    [str(py), launcher, script], cwd=str(engine),
                    env={**os.environ, **zenv, **env},
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    creationflags=WIN_SILENT,
                )
            elif stage == "4-sovits":  # s2_train：JSON 配置契约
                cfg_path = self._build_s2_config(engine, temp_dir, pre, opt_dir)
                self.proc = subprocess.Popen(
                    [str(py), launcher, script, "--config", str(cfg_path)],
                    cwd=str(engine), env={**os.environ, **zenv, **path_env},
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, creationflags=WIN_SILENT,
                )
            else:  # s1_train：YAML 配置契约
                cfg_path = self._build_s1_config(engine, temp_dir, pre, opt_dir)
                self.proc = subprocess.Popen(
                    [str(py), launcher, script, "--config_file", str(cfg_path)],
                    cwd=str(engine), env={**os.environ, **zenv, **path_env,
                                          "hz": "25hz",
                                          "_CUDA_VISIBLE_DEVICES": "0"},
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
            if stage == "2-sv":  # 引擎脚本逐行吞错，必须数产物防静默废数据
                self._verify_sv(opt_dir, list_path)
            if after is not None:
                after()
            self.shared.set_tr(stage, "完成", i, total)

        sv_root = engine / SOVITS_WEIGHT_VERSION2ROOT[version]
        gp_root = engine / GPT_WEIGHT_VERSION2ROOT[version]
        w2 = self._newest(sv_root, "*.pth")
        w1 = self._newest(gp_root, "*.ckpt")
        if w2:
            self.shared.log(f"[训练] SoVITS 权重：{w2}")
        if w1:
            self.shared.log(f"[训练] GPT 权重：{w1}")
        self.shared.set_tr("完成", "训练结束，可在「声线卡」页打包")

    # -- s2_train JSON 配置（与 webui.open1Ba 逐键一致） -------------------------
    def _build_s2_config(self, engine: Path, temp_dir: Path, pre: dict, opt_dir: Path) -> Path:
        temp_dir.mkdir(parents=True, exist_ok=True)
        version = pre["version"]
        bs = max(1, int(self.cfg["batch_size"]) // 2)  # webui：CPU(fp16 off) 时减半
        data = json.loads(pre["s2config"].read_text("utf-8"))
        data["train"].update({
            "fp16_run": False,
            "batch_size": bs,
            "epochs": int(self.cfg["s2_total_epoch"]),
            "text_low_lr_rate": float(self.cfg["text_low_lr_rate"]),
            "pretrained_s2G": str(pre["s2G"]),
            "pretrained_s2D": str(pre["s2D"]),
            "if_save_latest": True,
            "if_save_every_weights": True,
            "save_every_epoch": int(self.cfg["save_every_epoch"]),
            "gpu_numbers": "0",
            "grad_ckpt": False,
            "lora_rank": 8,
        })
        data["model"]["version"] = version
        data["data"]["exp_dir"] = data["s2_ckpt_dir"] = str(opt_dir)
        data["save_weight_dir"] = SOVITS_WEIGHT_VERSION2ROOT[version]
        data["name"] = self.exp
        data["version"] = version
        out = temp_dir / "tmp_s2.json"
        out.write_text(json.dumps(data), encoding="utf-8")
        return out

    # -- s1_train YAML 配置（与 webui.open1Bb 逐键一致） -------------------------
    def _build_s1_config(self, engine: Path, temp_dir: Path, pre: dict, opt_dir: Path) -> Path:
        import yaml  # noqa: PLC0415
        temp_dir.mkdir(parents=True, exist_ok=True)
        version = pre["version"]
        bs = max(1, int(self.cfg["batch_size"]) // 2)
        data = yaml.safe_load(pre["s1config"].read_text("utf-8"))
        data["train"].update({
            "precision": "32",
            "batch_size": bs,
            "epochs": int(self.cfg["s1_total_epoch"]),
            "save_every_n_epoch": int(self.cfg["save_every_epoch"]),
            "if_save_every_weights": True,
            "if_save_latest": True,
            "if_dpo": False,
            "half_weights_save_dir": GPT_WEIGHT_VERSION2ROOT[version],
            "exp_name": self.exp,
        })
        data["pretrained_s1"] = str(pre["s1"])
        data["train_semantic_path"] = str(opt_dir / "6-name2semantic.tsv")
        data["train_phoneme_path"] = str(opt_dir / "2-name2text.txt")
        data["output_dir"] = str(opt_dir / f"logs_s1_{version}")
        out = temp_dir / "tmp_s1.yaml"
        out.write_text(yaml.dump(data, allow_unicode=True, default_flow_style=False),
                       encoding="utf-8")
        return out

    @staticmethod
    def _verify_sv(opt_dir: Path, list_path: Path) -> None:
        """2-sv 产物守卫：引擎脚本逐行 try/except 吞掉所有错误，GPU 卷积失败
        也会"正常"退出，若不数产物就会带着废数据进入后续训练。"""
        expect = sum(1 for l in list_path.read_text("utf-8").splitlines() if l.strip())
        got = len(list((opt_dir / "7-sv_cn").glob("*.pt")))
        if got < expect:
            raise TrainError(
                f"2-sv 语音特征缺失：仅 {got}/{expect} 条成功（脚本逐行吞错，"
                f"多为卷积失败）。ZLUDA 模式请确认 HIP SDK/补丁（设置页可探测）；"
                f"清空 {opt_dir / '7-sv_cn'} 后重试"
            )

    @staticmethod
    def _newest(d: Path, pat: str) -> Path | None:
        if not d.is_dir():
            return None
        hits = sorted(d.glob(pat), key=lambda p: p.stat().st_mtime)
        return hits[-1] if hits else None
