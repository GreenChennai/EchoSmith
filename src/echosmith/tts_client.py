"""GPT-SoVITS api_v2 HTTP 客户端：切换权重 + 流式落盘 TTS。"""
from __future__ import annotations

from pathlib import Path

import requests

from .voice_card import VoiceCard

DEFAULT_TIMEOUT = (5, 600)  # 连接 5s，读取 10min（CPU 推理慢，长文本要等）


class TTSClientError(RuntimeError):
    pass


class TTSClient:
    def __init__(self, base_url: str, read_timeout: int = 600) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.timeout = (DEFAULT_TIMEOUT[0], int(read_timeout))
        self.loaded: tuple[str, str] | None = None  # (gpt, sovits) 当前已加载权重

    # -- 权重 ----------------------------------------------------------------
    def set_voice(self, card: VoiceCard, force: bool = False) -> None:
        """切换到声线卡对应的 GPT/SoVITS 权重；相同权重时跳过。"""
        target = (card.gpt, card.sovits)
        if not force and self.loaded == target:
            return
        if not card.gpt or not card.sovits:
            raise TTSClientError(f"声线卡「{card.name}」缺少 gpt/sovits 权重路径")
        self._get_json("/set_gpt_weights", {"weights_path": card.gpt})
        self._get_json("/set_sovits_weights", {"weights_path": card.sovits})
        self.loaded = target

    def _get_json(self, path: str, params: dict) -> dict:
        url = self.base_url + path
        try:
            r = self.session.get(url, params=params, timeout=self.timeout)
        except requests.RequestException as e:
            raise TTSClientError(f"请求引擎失败 {path}: {e}") from e
        if r.status_code != 200:
            raise TTSClientError(f"{path} 返回 {r.status_code}: {r.text[:200]}")
        data = r.json()
        msg = str(data.get("message", "")).lower()
        if data.get("code", 0) not in (0, 200, "0", "200") or "fail" in msg:
            raise TTSClientError(f"{path} 业务失败: {data}")
        return data

    # -- 合成 ------------------------------------------------------------------
    def tts_to_file(
        self,
        out_path: Path,
        *,
        text: str,
        text_lang: str,
        ref_audio_path: str,
        prompt_text: str = "",
        prompt_lang: str = "zh",
        text_split_method: str = "cut5",
        speed_factor: float = 1.0,
        aux_ref_audio_paths: list[str] | None = None,
    ) -> None:
        """合成一段语音并以引擎返回格式（wav）写盘。失败抛 TTSClientError。"""
        payload = {
            "text": text,
            "text_lang": text_lang,
            "ref_audio_path": ref_audio_path,
            "prompt_text": prompt_text,
            "prompt_lang": prompt_lang,
            "text_split_method": text_split_method,
            "speed_factor": float(speed_factor),
            "aux_ref_audio_paths": aux_ref_audio_paths or [],
            "streaming_mode": False,
        }
        url = self.base_url + "/tts"
        try:
            r = self.session.post(url, json=payload, timeout=self.timeout, stream=True)
        except requests.RequestException as e:
            raise TTSClientError(f"TTS 请求失败: {e}") from e
        if r.status_code != 200:
            # api_v2 业务错误以 400 + json {"message": ...} 返回
            detail = r.text[:200]
            raise TTSClientError(f"TTS 返回 {r.status_code}: {detail}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + ".part")
        try:
            with tmp.open("wb") as f:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        f.write(chunk)
            tmp.replace(out_path)
        finally:
            tmp.unlink(missing_ok=True)
