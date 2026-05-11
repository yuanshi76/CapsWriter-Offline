# coding: utf-8
"""OpenAI 兼容语音转写 HTTP 服务。"""

from __future__ import annotations

import json
import queue
import tempfile
import threading
import time
import uuid
from email.parser import BytesParser
from email.policy import default
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Tuple
from urllib.parse import urlparse

try:
    from config_server import OpenAIAPIConfig
except ImportError:
    class OpenAIAPIConfig:
        enable = False
        addr = "0.0.0.0"
        port = 6017
        api_key = ""
        max_upload_mb = 200
        request_timeout = 600
        seg_duration = 60.0
        seg_overlap = 4.0
from core.constants import AudioFormat
from core.server.schema import Task
from .audio import AudioDecodeError, decode_to_f32le
from .formatters import error_response, format_response
from .. import logger


class OpenAIAPIServer:
    """与现有 ASR 队列解耦的 OpenAI 兼容 HTTP 适配器。"""

    def __init__(self, app):
        self.app = app
        self.enabled = bool(getattr(OpenAIAPIConfig, "enable", False))
        self.addr = getattr(OpenAIAPIConfig, "addr", "0.0.0.0")
        self.port = int(getattr(OpenAIAPIConfig, "port", 6017))
        self.api_key = getattr(OpenAIAPIConfig, "api_key", "")
        self.max_upload_bytes = int(getattr(OpenAIAPIConfig, "max_upload_mb", 200)) * 1024 * 1024
        self.request_timeout = float(getattr(OpenAIAPIConfig, "request_timeout", 600))
        self.seg_duration = float(getattr(OpenAIAPIConfig, "seg_duration", 60.0))
        self.seg_overlap = float(getattr(OpenAIAPIConfig, "seg_overlap", 4.0))

        self._server = None
        self._thread = None

    def start(self) -> None:
        if not self.enabled or self._server:
            return

        handler = self._build_handler()
        self._server = ThreadingHTTPServer((self.addr, self.port), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, name="OpenAIAPI", daemon=True)
        self._thread.start()
        logger.info(f"OpenAI 兼容接口已启动: http://{self.addr}:{self.port}/v1")

    def stop(self) -> None:
        if not self._server:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        self._thread = None
        logger.info("OpenAI 兼容接口已停止")

    def register_waiter(self, task_id: str) -> queue.Queue:
        waiter = queue.Queue()
        self.app.state.openapi_waiters[task_id] = waiter
        return waiter

    def unregister_waiter(self, task_id: str) -> None:
        self.app.state.openapi_waiters.pop(task_id, None)

    def transcribe(self, audio_bytes: bytes, prompt: str = "", language: str = "auto"):
        task_id = f"openai-{uuid.uuid4().hex}"
        socket_id = task_id
        waiter = self.register_waiter(task_id)
        sockets_id = self.app.state.sockets_id
        if sockets_id is not None:
            sockets_id.append(socket_id)

        try:
            self._submit_audio(task_id, socket_id, audio_bytes, prompt, language)
            deadline = time.time() + self.request_timeout
            last_result = None

            while time.time() < deadline:
                try:
                    result = waiter.get(timeout=0.2)
                except queue.Empty:
                    continue
                last_result = result
                if result.is_final:
                    return result

            if last_result is not None:
                return last_result
            raise TimeoutError("等待识别结果超时")
        finally:
            self.unregister_waiter(task_id)
            if sockets_id is not None and socket_id in sockets_id:
                sockets_id.remove(socket_id)

    def _submit_audio(self, task_id: str, socket_id: str, audio_bytes: bytes, prompt: str, language: str) -> None:
        time_start = time.time()
        stride = AudioFormat.seconds_to_bytes(self.seg_duration)
        overlap_bytes = AudioFormat.seconds_to_bytes(self.seg_overlap)

        if len(audio_bytes) <= stride + overlap_bytes:
            self.app.state.queue_in.put(
                self._task(task_id, socket_id, audio_bytes, 0.0, True, time_start, prompt, language)
            )
            return

        offset = 0.0
        pos = 0
        while pos < len(audio_bytes):
            end = min(len(audio_bytes), pos + stride + overlap_bytes)
            is_final = end >= len(audio_bytes)
            segment = audio_bytes[pos:end]
            self.app.state.queue_in.put(
                self._task(task_id, socket_id, segment, offset, is_final, time_start, prompt, language)
            )
            if is_final:
                break
            pos += stride
            offset += self.seg_duration

    def _task(
        self,
        task_id: str,
        socket_id: str,
        data: bytes,
        offset: float,
        is_final: bool,
        time_start: float,
        prompt: str,
        language: str,
    ) -> Task:
        return Task(
            source="file",
            data=data,
            offset=offset,
            overlap=self.seg_overlap,
            task_id=task_id,
            socket_id=socket_id,
            is_final=is_final,
            time_start=time_start,
            time_submit=time.time(),
            context=prompt or "",
            language=language or "auto",
        )

    def _build_handler(self):
        api = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "CapsWriterOpenAI/1.0"

            def do_GET(self):
                path = urlparse(self.path).path.rstrip("/")
                if path == "/v1/models":
                    if not self._authorized():
                        self._send_error("Unauthorized", HTTPStatus.UNAUTHORIZED, "authentication_error")
                        return
                    self._json(
                        {
                            "object": "list",
                            "data": [
                                {
                                    "id": "capswriter-offline",
                                    "object": "model",
                                    "created": 0,
                                    "owned_by": "capswriter",
                                }
                            ],
                        }
                    )
                    return
                self._send_error("Not found", HTTPStatus.NOT_FOUND)

            def do_POST(self):
                path = urlparse(self.path).path
                if path not in ("/v1/audio/transcriptions", "/v1/audio/translations"):
                    self._send_error("Not found", HTTPStatus.NOT_FOUND)
                    return
                if not self._authorized():
                    self._send_error("Unauthorized", HTTPStatus.UNAUTHORIZED, "authentication_error")
                    return

                try:
                    fields, filename, file_data = self._read_multipart()
                    response_format = fields.get("response_format", "json")
                    prompt = fields.get("prompt", "")
                    language = fields.get("language", "auto")

                    with tempfile.TemporaryDirectory(prefix="capswriter-openai-") as tmpdir:
                        suffix = Path(filename or "audio").suffix or ".audio"
                        upload_path = Path(tmpdir) / f"upload{suffix}"
                        upload_path.write_bytes(file_data)
                        audio_bytes = decode_to_f32le(upload_path)

                    result = api.transcribe(audio_bytes, prompt=prompt, language=language)
                    body, content_type = format_response(result, response_format)
                    self._send(body, HTTPStatus.OK, content_type)
                except AudioDecodeError as exc:
                    self._send_error(str(exc), HTTPStatus.BAD_REQUEST)
                except TimeoutError as exc:
                    self._send_error(str(exc), HTTPStatus.REQUEST_TIMEOUT)
                except ValueError as exc:
                    self._send_error(str(exc), HTTPStatus.BAD_REQUEST)
                except Exception as exc:
                    logger.error(f"OpenAI 兼容接口请求失败: {exc}", exc_info=True)
                    self._send_error("Internal server error", HTTPStatus.INTERNAL_SERVER_ERROR, "server_error")

            def log_message(self, fmt, *args):
                logger.debug("OpenAIAPI " + fmt, *args)

            def _authorized(self) -> bool:
                if not api.api_key:
                    return True
                auth = self.headers.get("Authorization", "")
                return auth == f"Bearer {api.api_key}"

            def _read_multipart(self) -> Tuple[Dict[str, str], str, bytes]:
                content_length = int(self.headers.get("Content-Length", "0") or "0")
                if content_length <= 0:
                    raise ValueError("请求体为空")
                if content_length > api.max_upload_bytes:
                    raise ValueError("上传文件超过大小限制")

                content_type = self.headers.get("Content-Type", "")
                if "multipart/form-data" not in content_type:
                    raise ValueError("请求必须使用 multipart/form-data")

                raw_body = self.rfile.read(content_length)
                message = BytesParser(policy=default).parsebytes(
                    f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + raw_body
                )
                if not message.is_multipart():
                    raise ValueError("请求必须使用 multipart/form-data")

                fields: Dict[str, str] = {}
                filename = "audio"
                file_data = b""

                for part in message.iter_parts():
                    name = part.get_param("name", header="content-disposition")
                    if not name:
                        continue
                    payload = part.get_payload(decode=True) or b""
                    if name == "file":
                        filename = part.get_filename() or filename
                        file_data = payload
                        continue
                    charset = part.get_content_charset() or "utf-8"
                    fields[name] = payload.decode(charset, errors="ignore")

                if not file_data:
                    raise ValueError("缺少 file 字段或上传文件为空")

                return fields, filename, file_data

            def _json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK):
                self._send(json.dumps(payload, ensure_ascii=False).encode("utf-8"), status, "application/json; charset=utf-8")

            def _send_error(self, message: str, status: HTTPStatus, err_type: str = "invalid_request_error"):
                self._send(error_response(message, int(status), err_type), status, "application/json; charset=utf-8")

            def _send(self, body: bytes, status: HTTPStatus, content_type: str):
                self.send_response(int(status))
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return Handler
