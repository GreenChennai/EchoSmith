"""EchoSmith 命令行入口 —— Agent/Skill 的自动化主路径(无 GUI 依赖)。

用法(系统 Python + pip install -r requirements.txt):
  python -m echosmith.cli deploy-engine [--mirror] [--local PATH]
  python -m echosmith.cli dataset --src <媒体> --name <数据集名> [--separate]
  python -m echosmith.cli train --list <清单> --exp <实验名> [--s2 8] [--s1 15] [--device cpu|cuda]
  python -m echosmith.cli card --exp <> --voice <> --ref <wav> --prompt <> [--emotion 平静]
  python -m echosmith.cli synth --voice Jimi --text-file <txt> [--out <目录>] [--emotion 平静]
  python -m echosmith.cli probe

约定:遇错即停,退出码非 0;全程日志走 stdout。
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import time
from pathlib import Path

from .config import Config
from .state import Shared


def _make_shared() -> Shared:
    """Shared 日志同步打到 stdout,供 Agent 实时读取。"""
    shared = Shared()
    orig = shared.log

    def tee(msg: str) -> None:
        orig(msg)
        print(msg, flush=True)

    shared.log = tee  # type: ignore[method-assign]
    return shared


def _cfg_with(args: argparse.Namespace) -> Config:
    cfg = Config()
    if getattr(args, "engine_dir", None):
        cfg["engine_dir"] = args.engine_dir
    return cfg


# ------------------------------------------------------------------ deploy-engine
def cmd_deploy(args: argparse.Namespace) -> None:
    from .downloader import DownloadTask, _models_dir, load_manifest, valid_engine
    cfg = _cfg_with(args)
    shared = _make_shared()
    dest = _models_dir(cfg)
    final = dest / "GPT-SoVITS"

    if args.local:
        p = Path(args.local)
        task = DownloadTask({"url": str(p), "format": "7z"}, cfg, shared)
        if p.is_file() and p.suffix.lower() == ".7z":
            task._extract(p, dest)  # noqa: SLF001
        elif p.is_dir():
            import shutil
            shutil.rmtree(final, ignore_errors=True)
            shutil.copytree(p, final)
        else:
            raise SystemExit(f"--local 路径不存在:{p}")
        task._validate(dest)  # noqa: SLF001
    else:
        pkgs = load_manifest()
        if not pkgs:
            raise SystemExit("引擎清单加载失败(网络?)——可 --mirror 重试或 --local 导入")
        pkg = pkgs[0]
        if args.mirror and pkg.get("mirror_url"):
            pkg = {**pkg, "url": pkg["mirror_url"]}
        task = DownloadTask(pkg, cfg, shared)
        task.running.set()
        task._run()  # 下载→解压→校验→补丁,同步执行

    if valid_engine(final) is None:
        raise SystemExit(f"部署校验失败:{final}")
    print(f"引擎部署完成:{final}", flush=True)


# ------------------------------------------------------------------ dataset
def cmd_dataset(args: argparse.Namespace) -> None:
    from .pipeline import Pipeline
    cfg = _cfg_with(args)
    if args.ffmpeg:
        cfg["ffmpeg_path"] = args.ffmpeg
    shared = _make_shared()
    src = Path(args.src)
    if not src.is_file():
        raise SystemExit(f"素材不存在:{src}")
    pl = Pipeline(cfg, shared, src, args.name, separate_vocals=args.separate)
    pl.start()
    pl.join()
    err = shared.snapshot().get("error")
    if err:
        raise SystemExit(f"流水线失败:{err}")
    print(f"数据集完成:{cfg.dataset_path / args.name}", flush=True)


# ------------------------------------------------------------------ train
def cmd_train(args: argparse.Namespace) -> None:
    from .trainer import Trainer
    cfg = _cfg_with(args)
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 强制 CPU,即使存在 N 卡
    list_path = Path(args.list)
    if not list_path.is_file():
        raise SystemExit(f"训练清单不存在:{list_path}")
    if args.s2:
        cfg["s2_total_epoch"] = args.s2
    if args.s1:
        cfg["s1_total_epoch"] = args.s1
    if args.batch:
        cfg["batch_size"] = args.batch
    shared = _make_shared()
    tr = Trainer(cfg, shared, list_path, args.exp,
                 stages=args.stages.split(",") if args.stages else None)
    t0 = time.time()
    tr.start()
    tr.join()
    err = shared.snapshot().get("error")
    if err:
        raise SystemExit(f"训练失败:{err}")
    print(f"训练完成:{args.exp},耗时 {(time.time()-t0)/60:.1f} 分钟", flush=True)


# ------------------------------------------------------------------ card
def cmd_card(args: argparse.Namespace) -> None:
    from .card_gen import RefRow, build_card
    cfg = _cfg_with(args)
    card = build_card(
        cfg, cfg.engine_path,
        exp=args.exp,
        voice_name=args.voice,
        default_emotion=args.emotion,
        rows=[RefRow(label=args.emotion, ref=args.ref, prompt_text=args.prompt)],
    )
    print(f"声线卡已生成:{card}", flush=True)


# ------------------------------------------------------------------ synth
def _next_free_index(out_root: Path, voice: str) -> int:
    def safe(name: str) -> str:
        return re.sub(r'[\\/:*?"<>|]', "_", name).strip() or "voice"
    n = 1
    while any(out_root.glob(f"{safe(voice)}_*_{n:03d}.*")):
        n += 1
    return n


def cmd_synth(args: argparse.Namespace) -> None:
    from .engine import Engine
    from .tts_client import TTSClient
    from .voice_card import load_card

    cfg = _cfg_with(args)
    if args.port:
        cfg["engine_port"] = args.port
    card_path = Path(args.card) if args.card else \
        cfg.voices_path / args.voice / "card.json"
    if not card_path.is_file():
        raise SystemExit(f"声线卡不存在:{card_path}(或用 --card 指定)")
    card = load_card(card_path)
    if card is None:
        raise SystemExit(f"声线卡解析失败:{card_path}")
    emo = card.pick_emotion(args.emotion)

    text = Path(args.text_file).read_text("utf-8") if args.text_file else (args.text or "")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise SystemExit("没有有效文本行(--text-file 或 --text)")

    shared = _make_shared()
    engine = Engine(cfg, shared)
    t0 = time.time()
    engine.start()
    print(f"引擎就绪,耗时 {time.time()-t0:.0f}s", flush=True)
    client = TTSClient(cfg.engine_url, read_timeout=int(cfg["tts_read_timeout"] or 600))
    out_dir = Path(args.out) if args.out else \
        cfg.output_path / card.name / _dt.date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        client.set_voice(card, force=False)
        n = _next_free_index(out_dir, card.name)
        for i, line in enumerate(lines, 1):
            m = re.match(r"^【\s*(.+?)\s*】\s*", line)
            emo_i = card.pick_emotion(m.group(1) if m else None)
            body = line[m.end():] if m else line
            out = out_dir / f"{card.name}_{emo_i.label}_{n:03d}.wav"
            client.tts_to_file(
                out, text=body, text_lang=args.lang,
                ref_audio_path=emo_i.ref, prompt_text=emo_i.prompt_text,
                prompt_lang="zh", text_split_method=args.split,
                speed_factor=float(args.speed))
            n += 1
            print(f"[{i}/{len(lines)}] ✔ {out.name}", flush=True)
        print(f"合成完成:{len(lines)}/{len(lines)} 条 → {out_dir}", flush=True)
    except Exception as e:
        print(f"遇错停止:{type(e).__name__}: {e}", flush=True)
        raise SystemExit(1)
    finally:
        client.session.close()
        engine.stop()


# ------------------------------------------------------------------ probe
def cmd_probe(args: argparse.Namespace) -> None:
    from .device import probe
    cfg = _cfg_with(args)
    r = probe(cfg.engine_path, cfg)
    print({k: v for k, v in r.items()}, flush=True)
    if not r.get("ok"):
        raise SystemExit(1)


# ------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="echosmith.cli",
        description="声匠 EchoSmith 自动化命令行(部署引擎 / 数据集 / 训练 / 声线卡 / 合成)")
    ap.add_argument("--engine-dir", help="覆盖配置中的引擎目录")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("deploy-engine", help="下载/导入 GPT-SoVITS 整合包并校验")
    p.add_argument("--mirror", action="store_true", help="优先 hf-mirror 镜像")
    p.add_argument("--local", default=None, help="从本地 .7z 或已解压目录导入")
    p.set_defaults(fn=cmd_deploy)

    p = sub.add_parser("dataset", help="素材 → 切分 → 打标 → 训练清单")
    p.add_argument("--src", required=True, help="素材文件(视频/音频)")
    p.add_argument("--name", required=True, help="数据集名称")
    p.add_argument("--separate", action="store_true", help="先做 UVR5 人声分离")
    p.add_argument("--ffmpeg", default=None, help="ffmpeg 路径(留空自动探测)")
    p.set_defaults(fn=cmd_dataset)

    p = sub.add_parser("train", help="六阶段训练")
    p.add_argument("--list", required=True, help="训练清单 .list")
    p.add_argument("--exp", required=True, help="实验名")
    p.add_argument("--s2", type=int, default=None, help="SoVITS 轮数(默认取配置)")
    p.add_argument("--s1", type=int, default=None, help="GPT 轮数(默认取配置)")
    p.add_argument("--batch", type=int, default=None, help="批大小")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--stages", default=None, help="逗号分隔的阶段子集,如 4-sovits,5-gpt")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("card", help="生成声线卡")
    p.add_argument("--exp", required=True, help="实验名")
    p.add_argument("--voice", required=True, help="声线名")
    p.add_argument("--ref", required=True, help="参考音频 wav(3~10 秒)")
    p.add_argument("--prompt", required=True, help="参考音频里说的原话")
    p.add_argument("--emotion", default="平静", help="情感标签/默认情感")
    p.set_defaults(fn=cmd_card)

    p = sub.add_parser("synth", help="批量合成(自动起停引擎)")
    p.add_argument("--voice", default=None, help="声线名(models/voices 下的目录名)")
    p.add_argument("--card", default=None, help="或直接指定 card.json 路径")
    p.add_argument("--text-file", default=None, help="文本文件(每行一条)")
    p.add_argument("--text", default=None, help="或直接给文本(每行一条)")
    p.add_argument("--out", default=None, help="输出目录(默认 output/<声线>/<日期>/)")
    p.add_argument("--emotion", default=None, help="默认情感")
    p.add_argument("--lang", default="zh", help="文本语言")
    p.add_argument("--split", default="cut5", help="长文本切分方式")
    p.add_argument("--speed", type=float, default=1.0, help="语速 0.5~2.0")
    p.add_argument("--port", type=int, default=None, help="引擎端口(默认取配置)")
    p.set_defaults(fn=cmd_synth)

    p = sub.add_parser("probe", help="探测引擎 runtime 的 torch CUDA 可用性")
    p.set_defaults(fn=cmd_probe)

    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
