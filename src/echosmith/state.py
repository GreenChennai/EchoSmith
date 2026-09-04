"""线程共享状态（worker 线程写、UI 主线程读）。"""
from __future__ import annotations

import threading
from collections import deque


class Shared:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.app_log: deque[str] = deque(maxlen=500)
        # 下载（引擎整合包）
        self.dl_status = "未开始"
        self.dl_bytes = 0
        self.dl_total = 0
        self.dl_speed = 0.0
        # 流水线
        self.pl_status = "空闲"
        self.pl_stage = ""
        self.pl_index = 0
        self.pl_total = 0
        # 训练
        self.tr_stage = "空闲"
        self.tr_status = ""
        self.tr_index = 0
        self.tr_total = 0
        self.error = ""
        self.last_dataset = ""

    # 写入口 -----------------------------------------------------------------
    def log(self, msg: str) -> None:
        with self.lock:
            self.app_log.append(msg)

    def set_error(self, err: str) -> None:
        with self.lock:
            self.error = err

    # 下载进度
    def set_dl(self, status: str, done: int = -1, total: int = -1, speed: float = -1.0) -> None:
        with self.lock:
            self.dl_status = status
            if done >= 0:
                self.dl_bytes = done
            if total >= 0:
                self.dl_total = total
            if speed >= 0:
                self.dl_speed = speed

    # 流水线进度
    def set_pl(self, stage: str, status: str, index: int = -1, total: int = -1) -> None:
        with self.lock:
            self.pl_stage = stage
            self.pl_status = status
            if index >= 0:
                self.pl_index = index
            if total >= 0:
                self.pl_total = total

    # 训练进度
    def set_tr(self, stage: str, status: str = "", index: int = -1, total: int = -1) -> None:
        with self.lock:
            self.tr_stage = stage
            if status:
                self.tr_status = status
            if index >= 0:
                self.tr_index = index
            if total >= 0:
                self.tr_total = total

    def set_last_dataset(self, d: str) -> None:
        with self.lock:
            self.last_dataset = d

    # 读入口 -----------------------------------------------------------------
    def snapshot(self) -> dict:
        with self.lock:
            return {
                "app_log": list(self.app_log),
                "error": self.error,
                "dl": (self.dl_status, self.dl_bytes, self.dl_total, self.dl_speed),
                "pl": (self.pl_stage, self.pl_status, self.pl_index, self.pl_total),
                "tr": (self.tr_stage, self.tr_status, self.tr_index, self.tr_total),
                "last_dataset": self.last_dataset,
            }
