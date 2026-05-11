# coding: utf-8
"""音频解码工具：把上传文件转换为服务端内部 float32/16k/mono 字节流。"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class AudioDecodeError(RuntimeError):
    """上传音频无法解码。"""


def decode_to_f32le(audio_path: Path) -> bytes:
    """使用 ffmpeg 将常见音视频文件转成 f32le/16k/mono。"""
    if shutil.which("ffmpeg") is None:
        raise AudioDecodeError("未找到 ffmpeg，OpenAI 兼容接口需要 ffmpeg 解码上传音频")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(audio_path),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-",
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, check=False)
    except OSError as exc:
        raise AudioDecodeError(f"ffmpeg 启动失败: {exc}") from exc

    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="ignore").strip()
        raise AudioDecodeError(err or "ffmpeg 解码失败")

    if not proc.stdout:
        raise AudioDecodeError("ffmpeg 未输出有效音频")

    return proc.stdout
