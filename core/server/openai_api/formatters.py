# coding: utf-8
"""OpenAI 兼容响应格式化。"""

from __future__ import annotations

import json
from typing import Iterable, List, Tuple

from core.server.schema import Result


def result_text(result: Result) -> str:
    """优先返回精确拼接文本，缺失时回退到普通文本。"""
    return (result.text_accu or result.text or "").strip()


def format_response(result: Result, response_format: str) -> Tuple[bytes, str]:
    fmt = (response_format or "json").lower()
    text = result_text(result)

    if fmt == "text":
        return text.encode("utf-8"), "text/plain; charset=utf-8"
    if fmt == "srt":
        return _format_srt(result).encode("utf-8"), "text/plain; charset=utf-8"
    if fmt == "vtt":
        return _format_vtt(result).encode("utf-8"), "text/vtt; charset=utf-8"
    if fmt == "verbose_json":
        payload = {
            "task": "transcribe",
            "language": None,
            "duration": result.duration,
            "text": text,
            "segments": _segments(result),
        }
        return _json(payload), "application/json; charset=utf-8"

    return _json({"text": text}), "application/json; charset=utf-8"


def error_response(message: str, status: int = 400, err_type: str = "invalid_request_error") -> bytes:
    payload = {
        "error": {
            "message": message,
            "type": err_type,
            "param": None,
            "code": status,
        }
    }
    return _json(payload)


def _json(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _segments(result: Result) -> List[dict]:
    text = result_text(result)
    if not text:
        return []
    return [
        {
            "id": 0,
            "seek": 0,
            "start": 0.0,
            "end": round(float(result.duration or 0.0), 3),
            "text": text,
            "tokens": result.tokens,
            "temperature": 0.0,
            "avg_logprob": 0.0,
            "compression_ratio": 0.0,
            "no_speech_prob": 0.0,
        }
    ]


def _format_srt(result: Result) -> str:
    text = result_text(result)
    if not text:
        return ""
    return f"1\n{_srt_time(0.0)} --> {_srt_time(result.duration)}\n{text}\n"


def _format_vtt(result: Result) -> str:
    text = result_text(result)
    if not text:
        return "WEBVTT\n\n"
    return f"WEBVTT\n\n{_vtt_time(0.0)} --> {_vtt_time(result.duration)}\n{text}\n"


def _srt_time(seconds: float) -> str:
    h, m, s, ms = _split_time(seconds)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _vtt_time(seconds: float) -> str:
    h, m, s, ms = _split_time(seconds)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _split_time(seconds: float) -> Tuple[int, int, int, int]:
    total_ms = max(0, int(round(float(seconds or 0) * 1000)))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return h, m, s, ms
