"""引擎整合包下载器：manifest 驱动，断点续传，镜像回退，7z 解压，目录校验。"""
from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path

import requests

from .config import Config, app_root
from .state import Shared

MANIFEST_BUNDLED = app_root() / "assets" / "engine_manifest.json"
MANIFEST_REMOTE = (
    "https://raw.githubusercontent.com/GreenChennai/EchoSmith/main/assets/engine_manifest.json"
)


class DownloaderError(RuntimeError):
    pass


def load_manifest() -> list[dict]:
    """加载引擎清单：远端优先（可热更），失败回退本地内置。"""
    for src in (MANIFEST_REMOTE, MANIFEST_BUNDLED):
        try:
            if str(src).startswith("http"):
                r = requests.get(src, timeout=10)
                if r.status_code != 200:
                    continue
                data = r.json()
            else:
                if not src.is_file():
                    continue
                data = json.loads(src.read_text("utf-8"))
            pkgs = data.get("packages") or []
            if pkgs:
                return pkgs
        except (requests.RequestException, json.JSONDecodeError, OSError):
            continue
    return []


def valid_engine(engine_dir: Path) -> Path | None:
    """校验整合包目录，返回其 runtime python 路径；无效返回 None。"""
    if not engine_dir.is_dir():
        return None
    if not (engine_dir / "api_v2.py").is_file():
        return None
    for name in ("python.exe", "python3.exe"):
        p = engine_dir / "runtime" / name
        if p.is_file():
            return p
    p = engine_dir / "python" / "python.exe"
    return p if p.is_file() else None


class DownloadTask(threading.Thread):
    """下载 → 解压 → 校验 → 写配置。单实例；通过 cancel() 请求中断。"""

    def __init__(self, pkg: dict, cfg: Config, shared: Shared) -> None:
        super().__init__(name="echosmith-dl", daemon=True)
        self.pkg = pkg
        self.cfg = cfg
        self.shared = shared
        self.cancelled = threading.Event()
        self.running = threading.Event()

    # -- 入口 -----------------------------------------------------------------
    def run(self) -> None:
        self.running.set()
        try:
            self._run()
        except (DownloaderError, requests.RequestException, OSError) as e:
            self.shared.log(f"[下载] 失败：{e}")
            self.shared.set_dl("失败")
            self.shared.set_error(str(e))

    def _run(self) -> None:
        dest_dir = _models_dir(self.cfg)
        target = dest_dir / Path(self.pkg["url"].split("?")[0]).name
        self._download(target)
        if self.cancelled.is_set():
            return
        self._extract(target, dest_dir)
        if self.cancelled.is_set():
            return
        self._validate(dest_dir)

    # -- 下载（断点续传 + 镜像回退） ---------------------------------------------
    def _download(self, target: Path) -> None:
        part = target.with_suffix(target.suffix + ".part")
        urls = [u for u in (self.pkg.get("url"), self.pkg.get("mirror_url")) if u]
        last_err: Exception | None = None
        for url in urls:
            if self.cancelled.is_set():
                return
            try:
                self._download_one(url, target, part)
                return
            except (requests.RequestException, DownloaderError) as e:
                if self.cancelled.is_set():
                    return
                last_err = e
                self.shared.log(f"[下载] 源失败，尝试下一个镜像：{e}")
        raise DownloaderError(f"所有下载源均失败：{last_err}")

    def _download_one(self, url: str, target: Path, part: Path) -> None:
        headers = {}
        if part.exists():
            headers["Range"] = f"bytes={part.stat().st_size}-"
        r = requests.get(url, headers=headers, stream=True, timeout=(10, 60), allow_redirects=True)
        total = int(r.headers.get("Content-Length", 0))
        if r.status_code == 206:
            total += part.stat().st_size
            done = part.stat().st_size
            mode = "ab"
        elif r.status_code == 200:
            done = 0
            mode = "wb"
        else:
            raise DownloaderError(f"HTTP {r.status_code}")
        if total and part.exists() and part.stat().st_size >= total and mode == "ab":
            self.shared.set_dl("校验已有缓存", part.stat().st_size, total, 0)
            part.replace(target)
            return
        name = target.name
        self.shared.log(f"[下载] 开始：{name}（已续传 {done / 1e6:.0f}MB / 共 {total / 1e9:.2f}GB）")
        speed_win, speed_t0 = done, time.monotonic()
        with part.open(mode) as f:
            for chunk in r.iter_content(chunk_size=1024 * 512):
                if self.cancelled.is_set():
                    self.shared.log("[下载] 已取消（.part 缓存保留，可续传）")
                    self.shared.set_dl("已取消")
                    return
                if chunk:
                    f.write(chunk)
                    done += len(chunk)
                    now = time.monotonic()
                    if now - speed_t0 >= 1.0:
                        speed = (done - speed_win) / (now - speed_t0)
                        speed_win, speed_t0 = done, now
                        pct = f"{done * 100 / total:.1f}%" if total else f"{done / 1e6:.0f}MB"
                        self.shared.set_dl(
                            f"下载中 {pct}  {speed / 1e6:.1f} MB/s", done, total, speed
                        )
        self.shared.set_dl("下载完成", done, total, 0)
        part.replace(target)

    # -- 解压 -------------------------------------------------------------------
    @staticmethod
    def _find_7z() -> str | None:
        import shutil as _sh  # noqa: PLC0415
        for cand in (
            _sh.which("7z"),
            r"C:\Program Files\7-Zip\7z.exe",
            r"C:\Program Files (x86)\7-Zip\7z.exe",
        ):
            if cand and Path(cand).is_file():
                return cand
        return None

    def _extract(self, archive: Path, dest_dir: Path) -> None:
        self.shared.set_pl("解压", "进行中")
        self.shared.set_dl("解压中（大包需要几分钟）")
        extract_to = dest_dir / "_extract_tmp"
        shutil.rmtree(extract_to, ignore_errors=True)
        extract_to.mkdir(parents=True)
        # 官方整合包是 BCJ2 压缩，py7zr 不支持——优先用系统 7-Zip
        seven_zip = self._find_7z()
        if seven_zip:
            import subprocess  # noqa: PLC0415
            import sys as _sys  # noqa: PLC0415
            silent = 0x08000000 if _sys.platform == "win32" else 0
            r = subprocess.run(
                [seven_zip, "x", "-y", f"-o{extract_to}", str(archive)],
                capture_output=True, creationflags=silent, timeout=7200,
            )
            if r.returncode != 0:
                raise DownloaderError(
                    f"7-Zip 解压失败（exit {r.returncode}），日志尾部："
                    + r.stderr.decode("utf-8", "replace")[-300:]
                )
        else:
            try:
                import py7zr  # noqa: PLC0415
            except ImportError as e:
                raise DownloaderError(
                    "未找到 7-Zip 且缺少 py7zr，无法解压。"
                    "官方整合包为 BCJ2 压缩，py7zr 也不支持——请安装 7-Zip 后重试。"
                ) from e
            try:
                with py7zr.SevenZipFile(archive, mode="r") as z:
                    z.extractall(path=extract_to)
            except py7zr.Bad7zFile as e:
                raise DownloaderError("7z 包损坏（下载不完整？删除 .part 重新下载）") from e
        # 定位包含 api_v2.py 的目录（可能是包内顶层目录或解压根）
        src_root = extract_to
        if not (src_root / "api_v2.py").is_file():
            for sub in sorted(extract_to.iterdir()):
                if sub.is_dir() and (sub / "api_v2.py").is_file():
                    src_root = sub
                    break
        if not (src_root / "api_v2.py").is_file():
            raise DownloaderError("解压后未找到 api_v2.py，包结构异常")
        final = dest_dir / "GPT-SoVITS"
        shutil.rmtree(final, ignore_errors=True)
        src_root.replace(final)
        shutil.rmtree(extract_to, ignore_errors=True)
        archive.unlink(missing_ok=True)
        self.shared.log(f"[下载] 解压完成 → {final}")

    # -- 校验 -------------------------------------------------------------------
    def _validate(self, dest_dir: Path) -> None:
        final = dest_dir / "GPT-SoVITS"
        python = valid_engine(final)
        if python is None:
            raise DownloaderError(f"解压目录校验失败：{final}")
        self.cfg["engine_dir"] = str(final)
        self.cfg.save()
        self.shared.set_dl("引擎就绪 ✔")
        self.shared.log(f"[下载] 引擎校验通过，已写入配置：{final}")


# -- 小工具 -----------------------------------------------------------------------
def _models_dir(cfg: Config):
    from .config import resolve  # noqa: PLC0415
    d = resolve("models")
    d.mkdir(parents=True, exist_ok=True)
    return d
