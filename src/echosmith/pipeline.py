"""素材流水线编排：提音轨 → 静音切分 → 切片 → 打标 → 训练清单。单线程，遇错即停。"""
from __future__ import annotations

import threading
from pathlib import Path

from .config import Config
from .ffmpeg_util import extract_audio, probe_duration
from .labeler import LabelerError, check_service, transcribe, write_list
from .slicer import build_segments, detect_silences
from .state import Shared


class PipelineError(RuntimeError):
    pass


class Pipeline(threading.Thread):
    def __init__(self, cfg: Config, shared: Shared, src: Path, dataset_name: str,
                 separate_vocals: bool = False) -> None:
        super().__init__(name="echosmith-pipeline", daemon=True)
        self.cfg = cfg
        self.shared = shared
        self.src = src
        self.dataset_name = dataset_name
        self.separate_vocals = separate_vocals
        self.cancelled = threading.Event()

    def run(self) -> None:
        try:
            self._run()
        except (PipelineError, LabelerError, OSError, ValueError) as e:
            self.shared.log(f"[流水线] 失败：{e}")
            self.shared.set_pl("失败", "遇错停止")
            self.shared.set_error(str(e))

    def _run(self) -> None:
        cfg, shared = self.cfg, self.shared
        if not self.src.is_file():
            raise PipelineError(f"素材不存在：{self.src}")
        name = self.dataset_name.strip()
        if not name:
            raise PipelineError("数据集名称为空")

        ds_root = cfg.dataset_path / name
        clips_dir = ds_root / "clips"
        ds_root.mkdir(parents=True, exist_ok=True)

        # 1. 提音轨 ---------------------------------------------------------
        shared.set_pl("提音轨", "进行中", 0, 1)
        source_wav = ds_root / "source_32k.wav"
        extract_audio(cfg, self.src, source_wav)
        shared.log(f"[流水线] 音轨就绪：{source_wav.name}")
        wav32k = source_wav

        # 1.5 可选：UVR5 人声分离 ---------------------------------------------
        if self.separate_vocals:
            from .uvr5 import separate  # noqa: PLC0415
            shared.set_pl("人声分离", "进行中", 0, 1)
            vocal = separate(cfg, shared, wav32k)
            source_wav.unlink(missing_ok=True)  # 原始混合音轨不再需要
            wav32k = vocal

        # 2. 静音检测 + 切分 --------------------------------------------------
        shared.set_pl("静音检测", "进行中", 0, 1)
        total_dur = probe_duration(cfg, wav32k)
        silences = detect_silences(cfg, wav32k)
        segs = build_segments(
            silences, total_dur, int(cfg["min_clip_ms"]), int(cfg["max_clip_ms"])
        )
        if not segs:
            raise PipelineError(
                f"没有切出有效片段（时长 {total_dur:.1f}s，静音参数太紧？）"
            )
        shared.log(
            f"[流水线] 总时长 {total_dur:.1f}s → {len(segs)} 个片段"
            f"（{len(silences)} 处静音）"
        )

        # 3. 切片 ------------------------------------------------------------
        shared.set_pl("切片", "进行中", 0, len(segs))
        clips: list[Path] = []
        for i, seg in enumerate(segs, 1):
            if self.cancelled.is_set():
                shared.log("[流水线] 已取消")
                shared.set_pl("已取消", "停止")
                return
            clip = clips_dir / f"clip_{i:03d}.wav"
            from .ffmpeg_util import cut_clip  # noqa: PLC0415
            cut_clip(cfg, wav32k, seg.start, seg.end, clip)
            clips.append(clip)
            shared.set_pl("切片", f"{i}/{len(segs)}", i, len(segs))

        # 4. 打标 ------------------------------------------------------------
        if not check_service(cfg):
            raise LabelerError(
                f"本地 FunASR 服务不可达（{cfg['funasr_url']}）。"
                "请启动 MomentShift 的 FunASR 服务模式后再试。"
            )
        shared.set_pl("打标", "进行中", 0, len(clips))
        entries: list[tuple[Path, str]] = []
        for i, clip in enumerate(clips, 1):
            if self.cancelled.is_set():
                shared.log("[流水线] 已取消")
                shared.set_pl("已取消", "停止")
                return
            text = transcribe(cfg, clip)
            entries.append((clip, text))
            shared.set_pl("打标", f"{i}/{len(clips)}", i, len(clips))

        # 5. 写训练清单 ---------------------------------------------------------
        list_path = ds_root / f"{name}.list"
        write_list(entries, list_path, str(cfg["speaker"]), str(cfg["language"]))
        shared.set_last_dataset(str(ds_root))
        shared.set_pl("完成", f"{len(entries)} 条 → {list_path.name}")
        shared.log(f"[流水线] 完成：{list_path}（{len(entries)} 条标注）")
        # 清理中间长音频（仅原始混合音轨），保持目录干净
        source_wav.unlink(missing_ok=True)
