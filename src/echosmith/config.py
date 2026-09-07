"""EchoSmith 配置：config/echosmith.json，路径相对 app_root。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from . import APP_NAME, __version__

CONFIG_DIR_NAME = "config"
CONFIG_FILE_NAME = "echosmith.json"

DEFAULTS = {
    "engine_dir": "",                # GPT-SoVITS 整合包根目录（含 api_v2.py 与 runtime/）
    "engine_host": "127.0.0.1",      # 引擎 api_v2 绑定地址
    "engine_port": 9885,             # 引擎 api_v2 端口
    "input_dir": "input",            # 素材根目录
    "dataset_dir": "models/datasets",# 训练数据集根目录
    "output_dir": "output",
    "ffmpeg_path": "",               # 留空自动探测（PATH / MomentShift 内置）
    "ffprobe_path": "",
    # 本地 FunASR 打标服务（MomentShift 服务模式，OpenAI 兼容）
    "funasr_url": "http://127.0.0.1:8000/v1",
    "funasr_model": "paraformer-zh",
    # 切分参数
    "min_clip_ms": 800,
    "max_clip_ms": 9800,
    "silence_db": -35,
    "silence_min_dur": 0.5,
    # 数据集标注
    "speaker": "speaker0",
    "language": "ZH",
    # 训练参数（CPU 基线取小值）
    "s2_total_epoch": 8,
    "s1_total_epoch": 15,
    "batch_size": 6,
    "text_low_lr_rate": 0.4,
    "save_every_epoch": 4,
    # 声线卡输出目录（EchoRunner 兼容格式）
    "voices_dir": "models/voices",
    # 合成默认值
    "text_lang": "zh",               # 待合成文本语言 zh/en/ja/ko/yue
    "text_split_method": "cut5",     # 长文本切分方式
    "speed_factor": 1.0,             # 语速 0.5~2.0
    "save_mp3": False,               # 额外转出 mp3（需 ffmpeg）
    # 设备后端：CPU / NVIDIA CUDA 原生（引擎自动探测，无需配置）
}


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def resolve(root_relative: str | Path) -> Path:
    p = Path(root_relative)
    return p if p.is_absolute() else app_root() / p


class Config:
    def __init__(self) -> None:
        self.data: dict = dict(DEFAULTS)
        self.path = app_root() / CONFIG_DIR_NAME / CONFIG_FILE_NAME
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                self.data.update(json.loads(self.path.read_text("utf-8")))
            except (json.JSONDecodeError, OSError):
                try:
                    self.path.replace(self.path.with_suffix(".json.bad"))
                except OSError:
                    pass
        self._ensure_dirs()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _ensure_dirs(self) -> None:
        for key in ("input_dir", "dataset_dir", "output_dir"):
            try:
                resolve(self.data[key]).mkdir(parents=True, exist_ok=True)
            except OSError:
                pass

    def __getitem__(self, key: str):
        return self.data.get(key, DEFAULTS.get(key))

    def __setitem__(self, key: str, value) -> None:
        self.data[key] = value

    @property
    def engine_path(self) -> Path:
        return resolve(self.data["engine_dir"]) if self.data["engine_dir"] else Path()

    @property
    def input_path(self) -> Path:
        return resolve(self.data["input_dir"])

    @property
    def dataset_path(self) -> Path:
        return resolve(self.data["dataset_dir"])

    @property
    def voices_path(self) -> Path:
        return resolve(self.data["voices_dir"])

    @property
    def output_path(self) -> Path:
        return resolve(self.data["output_dir"])

    @property
    def engine_url(self) -> str:
        return f"http://{self.data['engine_host']}:{int(self.data['engine_port'])}"
