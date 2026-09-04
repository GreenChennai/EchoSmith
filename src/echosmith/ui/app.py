"""EchoSmith 主界面（Dear PyGui）：引擎向导 / 素材流水线 / 设置 / 日志。"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import dearpygui.dearpygui as dpg

from .. import APP_NAME, __version__
from ..config import Config
from ..downloader import DownloadTask, load_manifest, valid_engine
from ..fontutil import load_cjk_font
from ..pipeline import Pipeline
from ..state import Shared
from ..trainer import Trainer

T_PKG, T_DL_BAR, T_DL_STATUS, T_ENG_VALID = "pkg", "dl_bar", "dl_status", "eng_valid"
T_SRC, T_DSNAME, T_PL_STAGE, T_PL_BAR, T_PL_STATUS, T_LASTDS = (
    "src", "dsname", "pl_stage", "pl_bar", "pl_status", "lastds")
T_FFMPEG, T_FUNASR_URL, T_FUNASR_MODEL = "ffmpeg", "funasr_url", "funasr_model"
T_MINCLIP, T_MAXCLIP, T_SDB, T_SDUR, T_SPEAKER, T_LANG = (
    "minclip", "maxclip", "sdb", "sdur", "speaker", "lang")
T_LOG, T_ERR, T_BTN_DL, T_BTN_PL = "log", "err", "btn_dl", "btn_pl"
T_SEPARATE = "separate"
T_DS_COMBO, T_EXP, T_S2EP, T_S1EP, T_BS = "ds_combo", "exp", "s2ep", "s1ep", "bs"
T_TR_STAGE, T_TR_BAR, T_TR_STATUS, T_BTN_TRAIN = "tr_stage", "tr_bar", "tr_status", "btn_train"
T_CARD_EXP, T_CARD_NAME, T_CARD_DEF, T_CARD_ROWS, T_CARD_STATUS = (
    "card_exp", "card_name", "card_def", "card_rows", "card_status")


class App:
    def __init__(self) -> None:
        self.cfg = Config()
        self.shared = Shared()
        self.packages: list[dict] = []
        self.dl_task: DownloadTask | None = None
        self.pl_task: Pipeline | None = None
        self.trainer: Trainer | None = None
        self.card_rows: list[dict] = []
        self._picking = False
        self._src_file = ""

    def run(self) -> None:
        dpg.create_context()
        if not load_cjk_font():
            self.shared.log("[UI] 警告：未找到中文字体，中文可能显示为方块")
        self._build()
        dpg.create_viewport(title=f"{APP_NAME} v{__version__} — 声音克隆工作台",
                            width=1040, height=700, min_width=920, min_height=580)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main", True)
        self._refresh_engine_valid()
        self._sync_settings()
        self._install_poll()
        dpg.start_dearpygui()
        dpg.destroy_context()
        if self.dl_task and self.dl_task.running.is_set():
            self.dl_task.cancelled.set()
        if self.pl_task and self.pl_task.is_alive():
            self.pl_task.cancelled.set()
        if self.trainer and self.trainer.is_alive():
            self.trainer.cancel()

    # ------------------------------------------------------------ build ui
    def _build(self) -> None:
        with dpg.window(tag="main"):
            with dpg.tab_bar():
                with dpg.tab(label="引擎"):
                    self._build_engine_tab()
                with dpg.tab(label="素材流水线"):
                    self._build_pipeline_tab()
                with dpg.tab(label="训练"):
                    self._build_train_tab()
                with dpg.tab(label="声线卡"):
                    self._build_card_tab()
                with dpg.tab(label="设置"):
                    self._build_settings_tab()
                with dpg.tab(label="日志"):
                    dpg.add_text("应用日志", color=(160, 160, 160))
                    with dpg.child_window(tag=T_LOG, height=-1, horizontal_scrollbar=True):
                        pass

    def _build_engine_tab(self) -> None:
        dpg.add_text("GPT-SoVITS 整合包下载（GB 级，支持断点续传与镜像回退）",
                     color=(160, 160, 160))
        with dpg.group(horizontal=True):
            dpg.add_text("版本")
            dpg.add_combo([], tag=T_PKG, width=380, default_value="")
            dpg.add_button(label="刷新清单", callback=lambda: threading.Thread(
                target=self._load_packages, daemon=True).start())
        with dpg.group(horizontal=True):
            dpg.add_button(label="开始下载", tag=T_BTN_DL, callback=self._start_download)
            dpg.add_button(label="取消下载", callback=self._cancel_download)
            dpg.add_button(label="从本地 .7z / 已有目录导入", callback=self._pick_engine_source)
        dpg.add_spacer(height=6)
        dpg.add_progress_bar(tag=T_DL_BAR, default_value=0.0, width=-1, overlay="—")
        dpg.add_text("未开始", tag=T_DL_STATUS, color=(160, 160, 160))
        dpg.add_spacer(height=8)
        dpg.add_separator()
        dpg.add_text("引擎状态", color=(160, 160, 160))
        dpg.add_text("", tag=T_ENG_VALID, wrap=950)

    def _build_pipeline_tab(self) -> None:
        dpg.add_text("素材（视频/音频）→ 提音轨 → 静音切分 → FunASR 打标 → 训练清单",
                     color=(160, 160, 160))
        with dpg.group(horizontal=True):
            dpg.add_text("素材文件")
            dpg.add_input_text(tag=T_SRC, width=560, readonly=True,
                               hint="点击右侧按钮选择或拖入 input/ 目录")
            dpg.add_button(label="浏览…", callback=self._pick_src)
        with dpg.group(horizontal=True):
            dpg.add_text("数据集名称")
            dpg.add_input_text(tag=T_DSNAME, width=220, hint="如 xiaoyu")
        with dpg.group(horizontal=True):
            dpg.add_button(label="开始处理", tag=T_BTN_PL, callback=self._start_pipeline)
            dpg.add_button(label="取消", callback=self._cancel_pipeline)
            dpg.add_button(label="打开数据集目录", callback=self._open_dataset_dir)
        dpg.add_spacer(height=6)
        dpg.add_text("", tag=T_PL_STAGE)
        dpg.add_progress_bar(tag=T_PL_BAR, default_value=0.0, width=-1, overlay="—")
        dpg.add_text("空闲", tag=T_PL_STATUS, color=(160, 160, 160))
        dpg.add_text("", tag=T_LASTDS, color=(140, 140, 140), wrap=950)
        dpg.add_spacer(height=4)
        dpg.add_spacer(height=4)
        dpg.add_checkbox(tag=T_SEPARATE, label="先做人声/背景音分离（UVR5，需引擎就绪，人声干净素材可不勾）")
        dpg.add_text("", tag=T_ERR, color=(232, 84, 84), wrap=950)
        dpg.add_spacer(height=4)
        dpg.add_text("注：切分/打标参数在「设置」页调整。", color=(170, 150, 90))

    def _build_train_tab(self) -> None:
        dpg.add_text("数据集 → prepare_datasets×3 → SoVITS(s2) → GPT(s1)，CPU 单卡模式",
                     color=(160, 160, 160))
        with dpg.group(horizontal=True):
            dpg.add_text("训练清单")
            dpg.add_combo([], tag=T_DS_COMBO, width=380)
            dpg.add_button(label="刷新", callback=lambda: self._refresh_datasets())
        with dpg.group(horizontal=True):
            dpg.add_text("实验名")
            dpg.add_input_text(tag=T_EXP, width=180, hint="与数据集同名即可")
            dpg.add_text("SoVITS 轮数")
            dpg.add_input_int(tag=T_S2EP, width=90)
            dpg.add_text("GPT 轮数")
            dpg.add_input_int(tag=T_S1EP, width=90)
            dpg.add_text("批大小")
            dpg.add_input_int(tag=T_BS, width=90)
        with dpg.group(horizontal=True):
            dpg.add_button(label="开始训练", tag=T_BTN_TRAIN, callback=self._start_train)
            dpg.add_button(label="停止", callback=self._cancel_train)
        dpg.add_spacer(height=6)
        dpg.add_text("", tag=T_TR_STAGE)
        dpg.add_progress_bar(tag=T_TR_BAR, default_value=0.0, width=-1, overlay="—")
        dpg.add_text("空闲", tag=T_TR_STATUS, color=(160, 160, 160))
        dpg.add_spacer(height=4)
        dpg.add_text("CPU 训练耗时以分钟~小时计；轮数/批大小在「设置」页可调，日志实时滚到「日志」页。",
                     color=(170, 150, 90))

    def _build_card_tab(self) -> None:
        dpg.add_text("训练产物 + 参考音频 → card.json（EchoRunner 声线卡，落 models/voices/）",
                     color=(160, 160, 160))
        with dpg.group(horizontal=True):
            dpg.add_text("实验名")
            dpg.add_input_text(tag=T_CARD_EXP, width=180, hint="如 xiaoyu")
            dpg.add_button(label="扫描权重", callback=self._scan_weights)
            dpg.add_text("声线名")
            dpg.add_input_text(tag=T_CARD_NAME, width=140)
            dpg.add_text("默认情感")
            dpg.add_input_text(tag=T_CARD_DEF, width=110)
        dpg.add_text("权重：未扫描", tag=T_CARD_STATUS, color=(160, 160, 160), wrap=950)
        dpg.add_spacer(height=4)
        dpg.add_separator()
        dpg.add_text("参考音频（每行 = 一种情感；路径相对引擎目录或绝对路径）",
                     color=(160, 160, 160))
        with dpg.child_window(tag=T_CARD_ROWS, height=220):
            pass
        with dpg.group(horizontal=True):
            dpg.add_button(label="+ 添加情感行", callback=self._add_card_row)
            dpg.add_button(label="生成 card.json", callback=self._generate_card)
        dpg.add_spacer(height=4)
        dpg.add_text("参考音频建议 3~10 秒、人声干净；prompt_text 填该音频里说的原话，越准克隆越稳。",
                     color=(170, 150, 90))

    def _card_row_widget(self, row: dict) -> None:
        with dpg.group(horizontal=True, parent=T_CARD_ROWS):
            row["label"] = dpg.add_input_text(width=90, hint="情感标签")
            row["ref"] = dpg.add_input_text(width=420, hint="参考音频路径")
            dpg.add_button(label="选择…",
                           callback=lambda s, a, u=row: self._pick_ref_audio(u))
            row["prompt"] = dpg.add_input_text(width=320, hint="该音频里说的原话")

    def _add_card_row(self) -> None:
        row: dict = {}
        self._card_row_widget(row)
        self.card_rows.append(row)

    def _pick_ref_audio(self, row: dict) -> None:
        if self._picking:
            return
        self._picking = True

        def _cb(sender, app_data):
            self._picking = False
            path = app_data.get("file_path_name") if isinstance(app_data, dict) else None
            if path:
                dpg.set_value(row["ref"], path)

        dpg.add_file_dialog(directory_selector=False, show_label=False, modal=True,
                            callback=_cb,
                            cancel_callback=lambda *a: setattr(self, "_picking", False))

    def _scan_weights(self) -> None:
        from ..card_gen import scan_weights  # noqa: PLC0415
        exp = dpg.get_value(T_CARD_EXP).strip()
        if not exp:
            dpg.set_value(T_CARD_STATUS, "请先填实验名")
            return
        w = scan_weights(self.cfg.engine_path, exp)
        sv = w["sovits"][-1].name if w["sovits"] else "—"
        gp = w["gpt"][-1].name if w["gpt"] else "—"
        self._weights_ok = bool(w["sovits"] and w["gpt"])
        dpg.set_value(T_CARD_STATUS, f"SoVITS 最新：{sv}　|　GPT 最新：{gp}")

    _weights_ok = False

    def _generate_card(self) -> None:
        from ..card_gen import RefRow, build_card  # noqa: PLC0415
        rows = [RefRow(
            label=dpg.get_value(r["label"]),
            ref=dpg.get_value(r["ref"]).strip(),
            prompt_text=dpg.get_value(r["prompt"]).strip(),
        ) for r in self.card_rows]
        try:
            out = build_card(
                self.cfg, self.cfg.engine_path,
                exp=dpg.get_value(T_CARD_EXP).strip(),
                voice_name=dpg.get_value(T_CARD_NAME),
                default_emotion=dpg.get_value(T_CARD_DEF),
                rows=rows,
            )
            self.shared.log(f"[声线卡] 已生成：{out}")
            dpg.set_value(T_CARD_STATUS, f"card.json 已生成 ✔ {out}")
        except Exception as e:  # CardError 等直接透出
            self.shared.set_error(str(e))
            dpg.set_value(T_CARD_STATUS, f"生成失败：{e}")

    def _refresh_datasets(self) -> None:
        lists = sorted(self.cfg.dataset_path.glob("*/*.list"))
        items = [p.as_posix() for p in lists]
        dpg.configure_item(T_DS_COMBO, items=items,
                           default_value=items[0] if items else "")
        self.shared.log(f"[训练] 找到 {len(items)} 个训练清单")

    def _start_train(self) -> None:
        dpg.set_value(T_ERR, "")
        self.shared.set_error("")
        list_path = dpg.get_value(T_DS_COMBO).strip()
        if not list_path:
            dpg.set_value(T_ERR, "没有训练清单：先在「素材流水线」产出数据集")
            return
        if self.trainer and self.trainer.is_alive():
            dpg.set_value(T_ERR, "已有训练在运行")
            return
        self.cfg["s2_total_epoch"] = max(1, int(dpg.get_value(T_S2EP)))
        self.cfg["s1_total_epoch"] = max(1, int(dpg.get_value(T_S1EP)))
        self.cfg["batch_size"] = max(1, int(dpg.get_value(T_BS)))
        exp = dpg.get_value(T_EXP).strip() or Path(list_path).stem
        self.trainer = Trainer(self.cfg, self.shared, Path(list_path), exp)
        self.trainer.start()

    def _cancel_train(self) -> None:
        if self.trainer and self.trainer.is_alive():
            self.trainer.cancel()

    def _build_settings_tab(self) -> None:
        with dpg.group(horizontal=True):
            dpg.add_text("ffmpeg")
            dpg.add_input_text(tag=T_FFMPEG, width=560, hint="留空自动探测（PATH / MomentShift 内置）")
        dpg.add_spacer(height=4)
        with dpg.group(horizontal=True):
            dpg.add_text("FunASR 地址")
            dpg.add_input_text(tag=T_FUNASR_URL, width=380)
            dpg.add_text("模型")
            dpg.add_input_text(tag=T_FUNASR_MODEL, width=170)
        with dpg.group(horizontal=True):
            dpg.add_text("说话人标识")
            dpg.add_input_text(tag=T_SPEAKER, width=140)
            dpg.add_text("语言")
            dpg.add_input_text(tag=T_LANG, width=80)
        dpg.add_spacer(height=6)
        dpg.add_separator()
        dpg.add_text("切分参数", color=(160, 160, 160))
        with dpg.group(horizontal=True):
            dpg.add_text("最短片段 ms")
            dpg.add_input_int(tag=T_MINCLIP, width=110)
            dpg.add_text("最长片段 ms")
            dpg.add_input_int(tag=T_MAXCLIP, width=110)
        with dpg.group(horizontal=True):
            dpg.add_text("静音阈值 dB")
            dpg.add_input_int(tag=T_SDB, width=110)
            dpg.add_text("静音最短时长 s")
            dpg.add_input_float(tag=T_SDUR, width=110, format="%.1f")
        dpg.add_spacer(height=6)
        dpg.add_button(label="保存设置", callback=self._save_settings)

    # ------------------------------------------------------------ polling
    def _install_poll(self) -> None:
        def _poll() -> None:
            self._refresh_dynamic()
            dpg.set_frame_callback(dpg.get_frame_count() + 12, _poll)
        dpg.set_frame_callback(8, _poll)

    def _refresh_dynamic(self) -> None:
        s = self.shared.snapshot()
        # 待应用的清单（后台线程 → 主线程中转）
        if self._pending_labels is not None:
            labels, self._pending_labels = self._pending_labels, None
            dpg.configure_item(T_PKG, items=labels,
                               default_value=labels[0] if labels else "")
        # 下载
        status, done, total, speed = s["dl"]
        pct = (done / total) if total else 0.0
        dpg.set_value(T_DL_BAR, pct)
        dpg.configure_item(T_DL_BAR, overlay=f"{pct*100:.1f}%" if total else "—")
        dpg.set_value(T_DL_STATUS, status)
        # 流水线
        stage, plstatus, idx, tot = s["pl"]
        dpg.set_value(T_PL_STAGE, f"阶段：{stage}" if stage else "")
        ppct = (idx / tot) if tot else 0.0
        dpg.set_value(T_PL_BAR, ppct)
        dpg.configure_item(T_PL_BAR, overlay=f"{idx}/{tot}" if tot else "—")
        dpg.set_value(T_PL_STATUS, plstatus)
        dpg.set_value(T_LASTDS, f"最近数据集：{s['last_dataset']}" if s["last_dataset"] else "")
        dpg.set_value(T_ERR, s["error"])
        # 训练
        tr_stage, tr_status, tr_idx, tr_tot = s["tr"]
        dpg.set_value(T_TR_STAGE, f"阶段：{tr_stage}" if tr_stage else "")
        tpct = (tr_idx / tr_tot) if tr_tot else 0.0
        dpg.set_value(T_TR_BAR, tpct)
        dpg.configure_item(T_TR_BAR, overlay=f"{tr_idx}/{tr_tot}" if tr_tot else "—")
        dpg.set_value(T_TR_STATUS, tr_status)
        # 日志
        lines = s["app_log"]
        children = dpg.get_item_children(T_LOG, 1)
        n_shown = len(children) if children else 0
        if n_shown > len(lines):
            for c in list(children):
                dpg.delete_item(c)
            n_shown = 0
        for line in lines[n_shown:]:
            dpg.add_text(line, parent=T_LOG)
        self._refresh_engine_valid()

    def _refresh_engine_valid(self) -> None:
        eng = self.cfg.engine_path
        if not str(self.cfg["engine_dir"]):
            dpg.set_value(T_ENG_VALID, "尚未配置引擎目录。")
            return
        if valid_engine(eng):
            dpg.set_value(T_ENG_VALID, f"引擎就绪 ✔  {eng}")
        else:
            dpg.set_value(T_ENG_VALID,
                          f"已配置目录但校验失败（缺 api_v2.py 或 runtime/python.exe）：{eng}")

    # ------------------------------------------------------------ packages
    def _load_packages(self) -> None:
        self.packages = load_manifest()
        self.shared.log(f"[下载] 清单加载：{len(self.packages)} 个版本")
        self._pending_labels = [p["label"] for p in self.packages]

    _pending_labels: list[str] | None = None

    # ------------------------------------------------------------ actions
    def _start_download(self) -> None:
        dpg.set_value(T_ERR, "")
        if not self.packages:
            self.shared.log("[下载] 清单为空，先点「刷新清单」")
            return
        label = dpg.get_value(T_PKG)
        pkg = next((p for p in self.packages if p["label"] == label), None)
        if pkg is None:
            return
        self.dl_task = DownloadTask(pkg, self.cfg, self.shared)
        self.dl_task.start()

    def _cancel_download(self) -> None:
        if self.dl_task and self.dl_task.running.is_set():
            self.dl_task.cancelled.set()

    def _pick_engine_source(self) -> None:
        if self._picking:
            return
        self._picking = True

        def _cb(sender, app_data):
            self._picking = False
            path = app_data.get("file_path_name") if isinstance(app_data, dict) else None
            if not path:
                return
            p = Path(path)
            threading.Thread(target=self._import_engine, args=(p,), daemon=True).start()

        dpg.add_file_dialog(directory_selector=True, show_label=False, modal=True,
                            callback=_cb,
                            cancel_callback=lambda *a: setattr(self, "_picking", False))

    def _import_engine(self, p: Path) -> None:
        """本地 .7z 导入或直接指定已解压的整合包目录。"""
        from ..downloader import DownloaderError, _models_dir  # noqa: PLC0415
        try:
            if p.is_file() and p.suffix.lower() == ".7z":
                dest = _models_dir(self.cfg)
                task = DownloadTask({"url": str(p), "format": "7z"}, self.cfg, self.shared)
                task._extract(p, dest)  # noqa: SLF001
                task._validate(dest)  # noqa: SLF001
            elif p.is_dir():
                if valid_engine(p) is None:
                    raise DownloaderError("所选目录缺 api_v2.py 或 runtime/python.exe")
                self.cfg["engine_dir"] = str(p)
                self.cfg.save()
                self.shared.log(f"[引擎] 已写入配置：{p}")
                self.shared.set_dl("引擎就绪 ✔（本地目录）")
            else:
                raise DownloaderError("请选择 .7z 文件或已解压的整合包目录")
        except (DownloaderError, OSError) as e:
            self.shared.log(f"[引擎] 导入失败：{e}")
            self.shared.set_error(str(e))

    def _pick_src(self) -> None:
        if self._picking:
            return
        self._picking = True

        def _cb(sender, app_data):
            self._picking = False
            path = app_data.get("file_path_name") if isinstance(app_data, dict) else None
            if path:
                self._src_file = path
                dpg.set_value(T_SRC, path)

        dpg.add_file_dialog(directory_selector=False, show_label=False, modal=True,
                            callback=_cb,
                            cancel_callback=lambda *a: setattr(self, "_picking", False))

    def _start_pipeline(self) -> None:
        dpg.set_value(T_ERR, "")
        self.shared.set_error("")
        src = dpg.get_value(T_SRC).strip()
        if not src:
            dpg.set_value(T_ERR, "请先选择素材文件")
            return
        if self.pl_task and self.pl_task.is_alive():
            dpg.set_value(T_ERR, "已有流水线在运行")
            return
        self.pl_task = Pipeline(self.cfg, self.shared, Path(src),
                                dpg.get_value(T_DSNAME).strip(),
                                separate_vocals=bool(dpg.get_value(T_SEPARATE)))
        self.pl_task.start()

    def _cancel_pipeline(self) -> None:
        if self.pl_task and self.pl_task.is_alive():
            self.pl_task.cancelled.set()

    def _open_dataset_dir(self) -> None:
        s = self.shared.snapshot()
        target = s["last_dataset"] or str(self.cfg.dataset_path)
        p = Path(target)
        p.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(p))  # type: ignore[attr-defined]  # noqa: S606
        except OSError:
            pass

    def _sync_settings(self) -> None:
        dpg.set_value(T_FFMPEG, str(self.cfg["ffmpeg_path"] or ""))
        dpg.set_value(T_FUNASR_URL, str(self.cfg["funasr_url"]))
        dpg.set_value(T_FUNASR_MODEL, str(self.cfg["funasr_model"]))
        dpg.set_value(T_SPEAKER, str(self.cfg["speaker"]))
        dpg.set_value(T_LANG, str(self.cfg["language"]))
        dpg.set_value(T_MINCLIP, int(self.cfg["min_clip_ms"]))
        dpg.set_value(T_MAXCLIP, int(self.cfg["max_clip_ms"]))
        dpg.set_value(T_SDB, int(self.cfg["silence_db"]))
        dpg.set_value(T_SDUR, float(self.cfg["silence_min_dur"]))
        dpg.set_value(T_S2EP, int(self.cfg["s2_total_epoch"]))
        dpg.set_value(T_S1EP, int(self.cfg["s1_total_epoch"]))
        dpg.set_value(T_BS, int(self.cfg["batch_size"]))

    def _save_settings(self) -> None:
        self.cfg["ffmpeg_path"] = dpg.get_value(T_FFMPEG).strip()
        self.cfg["funasr_url"] = dpg.get_value(T_FUNASR_URL).strip()
        self.cfg["funasr_model"] = dpg.get_value(T_FUNASR_MODEL).strip()
        self.cfg["speaker"] = dpg.get_value(T_SPEAKER).strip() or "speaker0"
        self.cfg["language"] = dpg.get_value(T_LANG).strip() or "ZH"
        self.cfg["min_clip_ms"] = max(100, int(dpg.get_value(T_MINCLIP)))
        self.cfg["max_clip_ms"] = max(1000, int(dpg.get_value(T_MAXCLIP)))
        self.cfg["silence_db"] = int(dpg.get_value(T_SDB))
        self.cfg["silence_min_dur"] = max(0.1, float(dpg.get_value(T_SDUR)))
        try:
            self.cfg.save()
            self.shared.log("[设置] 已保存 config/echosmith.json")
        except OSError as e:
            self.shared.set_error(f"保存失败：{e}")


def run() -> None:
    App().run()
