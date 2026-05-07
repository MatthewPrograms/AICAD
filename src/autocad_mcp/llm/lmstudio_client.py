"""LM Studio OpenAI-compatible client helpers."""

from __future__ import annotations
import base64

import json
import mimetypes
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx
_UNUSED_TOKEN_PATTERN = re.compile(r"<unused\d+>", re.IGNORECASE)


@dataclass
class LMStudioConfig:
    """Runtime configuration for LM Studio connection."""

    base_url: str = os.environ.get("AUTOCAD_MCP_LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
    model: str = os.environ.get("AUTOCAD_MCP_LMSTUDIO_MODEL", "local-model")
    timeout_seconds: float = float(os.environ.get("AUTOCAD_MCP_LMSTUDIO_TIMEOUT", "180.0"))
    timeout_retry_count: int = int(os.environ.get("AUTOCAD_MCP_LMSTUDIO_TIMEOUT_RETRIES", "0"))
    timeout_retry_backoff_seconds: float = float(os.environ.get("AUTOCAD_MCP_LMSTUDIO_TIMEOUT_BACKOFF", "1.5"))
    temperature: float = float(os.environ.get("AUTOCAD_MCP_LMSTUDIO_TEMPERATURE", "0.1"))
    max_json_tokens: int = int(os.environ.get("AUTOCAD_MCP_LMSTUDIO_MAX_JSON_TOKENS", "320"))
    max_text_tokens: int = int(os.environ.get("AUTOCAD_MCP_LMSTUDIO_MAX_TEXT_TOKENS", "280"))


class LMStudioClient:
    """Minimal local client for LM Studio's OpenAI-compatible API."""

    def __init__(self, config: LMStudioConfig | None = None):
        self.config = config or LMStudioConfig()
        self._client = httpx.Client(
            timeout=httpx.Timeout(
                connect=min(10.0, self.config.timeout_seconds),
                read=self.config.timeout_seconds,
                write=min(30.0, self.config.timeout_seconds),
                pool=min(10.0, self.config.timeout_seconds),
            )
        )

    def close(self) -> None:
        self._client.close()

    def _url(self, path: str) -> str:
        return f"{self.config.base_url.rstrip('/')}/{path.lstrip('/')}"

    def list_models(self) -> list[str]:
        resp = self._client.get(self._url("/models"))
        resp.raise_for_status()
        payload = resp.json()
        models = payload.get("data", [])
        return [m.get("id", "unknown") for m in models]

    def _resolve_model(self) -> str:
        """Pick a usable model id from config + LM Studio loaded models."""
        configured = (self.config.model or "").strip()
        models = self.list_models()
        if not models:
            raise RuntimeError("LM Studio is reachable but no models are loaded.")

        # Common placeholder values should auto-resolve to a loaded model.
        if configured.lower() in ("", "auto", "local-model"):
            return models[0]
        if configured in models:
            return configured
        # Configured model not present; fall back to first available.
        return models[0]

    def health(self) -> tuple[bool, str]:
        """Return (is_healthy, message)."""
        try:
            models = self.list_models()
            if not models:
                return False, "Reachable, but no models are loaded in LM Studio."
            return True, f"OK ({len(models)} model(s)): {', '.join(models)}"
        except Exception as ex:  # pragma: no cover - defensive path
            return False, f"LM Studio error: {ex}"

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        on_token: Callable[[str], None] | None = None,
    ) -> dict:
        """Generate a JSON object response from LM Studio."""
        return self._chat_json_from_messages(system_prompt, user_prompt, [], on_token=on_token)

    def chat_json_with_images(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[str] | None = None,
        image_b64_pngs: list[str] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> dict:
        """Generate a JSON response with image context."""
        image_data_urls: list[str] = []
        for path in (image_paths or []):
            image_data_urls.append(self._image_path_to_data_url(path))
        for b64_png in (image_b64_pngs or []):
            image_data_urls.append(f"data:image/png;base64,{b64_png}")
        return self._chat_json_from_messages(system_prompt, user_prompt, image_data_urls, on_token=on_token)
    def chat_text(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[str] | None = None,
        image_b64_pngs: list[str] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        """Generate a plain-text response from LM Studio."""
        image_data_urls: list[str] = []
        for path in (image_paths or []):
            image_data_urls.append(self._image_path_to_data_url(path))
        for b64_png in (image_b64_pngs or []):
            image_data_urls.append(f"data:image/png;base64,{b64_png}")
        return self._chat_text_from_messages(system_prompt, user_prompt, image_data_urls, on_token=on_token)

    def _chat_json_from_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        image_data_urls: list[str],
        on_token: Callable[[str], None] | None = None,
    ) -> dict:
        model_id = self._resolve_model()
        if image_data_urls:
            user_content: str | list[dict] = [{"type": "text", "text": user_prompt}]
            for image_url in image_data_urls:
                user_content.append({"type": "image_url", "image_url": {"url": image_url}})
        else:
            user_content = user_prompt

        request_payload = {
            "model": model_id,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_json_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "response_format": {"type": "json_object"},
        }
        if on_token is not None:
            content = self._stream_chat_completion_json(
                request_payload,
                on_token,
                allow_response_format_fallback=True,
            )
        else:
            data = self._post_chat_completion(request_payload, allow_response_format_fallback=True)
            content = data["choices"][0]["message"]["content"]
        content = self._sanitize_model_content(content)
        if not content.strip():
            raise RuntimeError("LM Studio returned empty content while JSON output was expected.")
        try:
            return _loads_json(content)
        except json.JSONDecodeError as ex:
            raw_content = str(content)
            if len(raw_content) > 800:
                raw_content = raw_content[:800] + "..."
            if not raw_content:
                raw_content = "<empty>"
            raise RuntimeError(
                f"LM Studio returned non-JSON content; ensure your model follows JSON-only instructions. "
                f"Full content: {raw_content}"
            ) from ex

    def _stream_chat_completion_json(
        self,
        payload: dict,
        on_token: Callable[[str], None],
        allow_response_format_fallback: bool,
    ) -> str:
        stream_payload = dict(payload)
        stream_payload["stream"] = True
        attempts = 1 + max(0, self.config.timeout_retry_count)
        for attempt in range(attempts):
            try:
                chunks: list[str] = []
                with self._client.stream("POST", self._url("/chat/completions"), json=stream_payload) as resp:
                    resp.raise_for_status()
                    for raw_line in resp.iter_lines():
                        if raw_line is None:
                            continue
                        line = raw_line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data_line = line[5:].strip()
                        if not data_line:
                            continue
                        if data_line == "[DONE]":
                            break
                        try:
                            event = json.loads(data_line)
                        except json.JSONDecodeError:
                            continue
                        choices = event.get("choices")
                        if not isinstance(choices, list) or not choices:
                            continue
                        choice = choices[0] if isinstance(choices[0], dict) else {}
                        delta = choice.get("delta")
                        if isinstance(delta, dict):
                            delta_content = delta.get("content")
                            if isinstance(delta_content, str) and delta_content:
                                cleaned_delta = self._sanitize_model_content(delta_content, trim=False)
                                if not cleaned_delta:
                                    continue
                                chunks.append(cleaned_delta)
                                for fragment in self._iter_stream_fragments(cleaned_delta):
                                    on_token(fragment)
                                continue
                        message = choice.get("message")
                        if isinstance(message, dict):
                            message_content = message.get("content")
                            if isinstance(message_content, str) and message_content:
                                cleaned_message = self._sanitize_model_content(message_content, trim=False)
                                if not cleaned_message:
                                    continue
                                chunks.append(cleaned_message)
                                for fragment in self._iter_stream_fragments(cleaned_message):
                                    on_token(fragment)
                if chunks:
                    return "".join(chunks).strip()
                data = self._post_chat_completion(
                    payload,
                    allow_response_format_fallback=allow_response_format_fallback,
                )
                content = data["choices"][0]["message"]["content"]
                content_text = self._sanitize_model_content(content)
                for fragment in self._iter_stream_fragments(content_text):
                    on_token(fragment)
                return content_text
            except httpx.TimeoutException as ex:
                if attempt < attempts - 1:
                    time.sleep(self.config.timeout_retry_backoff_seconds)
                    continue
                raise RuntimeError(
                    f"LM Studio request timed out after {attempts} attempt(s). "
                    f"Current timeout is {self.config.timeout_seconds:.1f}s. "
                    f"Set AUTOCAD_MCP_LMSTUDIO_TIMEOUT to a higher value if needed."
                ) from ex
            except httpx.HTTPStatusError as ex:
                status = ex.response.status_code
                body = _safe_response_text(ex.response).lower()
                if status == 400 and ("image_url" in body or "vision" in body or "image" in body):
                    raise RuntimeError(
                        "LM Studio rejected image input for this model/provider. "
                        "Load a vision-capable model and retry."
                    ) from ex
                if status == 400 and allow_response_format_fallback:
                    if "response_format" in payload:
                        fallback_payload = dict(payload)
                        fallback_payload.pop("response_format", None)
                        return self._stream_chat_completion_json(
                            fallback_payload,
                            on_token,
                            allow_response_format_fallback=False,
                        )
                if status == 400:
                    data = self._post_chat_completion(
                        payload,
                        allow_response_format_fallback=allow_response_format_fallback,
                    )
                    content = data["choices"][0]["message"]["content"]
                    content_text = self._sanitize_model_content(content)
                    for fragment in self._iter_stream_fragments(content_text):
                        on_token(fragment)
                    return content_text
                raise RuntimeError(_format_http_error(ex)) from ex
            except httpx.HTTPError as ex:
                raise RuntimeError(f"LM Studio request failed: {ex}") from ex

        raise RuntimeError("LM Studio streaming JSON request failed unexpectedly.")
    def _chat_text_from_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        image_data_urls: list[str],
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        model_id = self._resolve_model()
        if image_data_urls:
            user_content: str | list[dict] = [{"type": "text", "text": user_prompt}]
            for image_url in image_data_urls:
                user_content.append({"type": "image_url", "image_url": {"url": image_url}})
        else:
            user_content = user_prompt

        request_payload = {
            "model": model_id,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_text_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        if on_token is not None:
            content = self._stream_chat_completion_text(request_payload, on_token)
        else:
            data = self._post_chat_completion(request_payload, allow_response_format_fallback=False)
            content = data["choices"][0]["message"]["content"]
        content = self._sanitize_model_content(content)
        if self._looks_repetitive(content):
            raise RuntimeError(
                "LM Studio output appears repetitive/looping. "
                "Try a smaller prompt or adjust model settings."
            )
        if isinstance(content, str):
            return content.strip()
        return str(content).strip()

    def _stream_chat_completion_text(self, payload: dict, on_token: Callable[[str], None]) -> str:
        stream_payload = dict(payload)
        stream_payload["stream"] = True
        attempts = 1 + max(0, self.config.timeout_retry_count)
        for attempt in range(attempts):
            try:
                chunks: list[str] = []
                with self._client.stream("POST", self._url("/chat/completions"), json=stream_payload) as resp:
                    resp.raise_for_status()
                    for raw_line in resp.iter_lines():
                        if raw_line is None:
                            continue
                        line = raw_line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data_line = line[5:].strip()
                        if not data_line:
                            continue
                        if data_line == "[DONE]":
                            break
                        try:
                            event = json.loads(data_line)
                        except json.JSONDecodeError:
                            continue
                        choices = event.get("choices")
                        if not isinstance(choices, list) or not choices:
                            continue
                        choice = choices[0] if isinstance(choices[0], dict) else {}
                        delta = choice.get("delta")
                        if isinstance(delta, dict):
                            delta_content = delta.get("content")
                            if isinstance(delta_content, str) and delta_content:
                                cleaned_delta = self._sanitize_model_content(delta_content, trim=False)
                                if not cleaned_delta:
                                    continue
                                chunks.append(cleaned_delta)
                                for fragment in self._iter_stream_fragments(cleaned_delta):
                                    on_token(fragment)
                                continue
                        message = choice.get("message")
                        if isinstance(message, dict):
                            message_content = message.get("content")
                            if isinstance(message_content, str) and message_content:
                                cleaned_message = self._sanitize_model_content(message_content, trim=False)
                                if not cleaned_message:
                                    continue
                                chunks.append(cleaned_message)
                                for fragment in self._iter_stream_fragments(cleaned_message):
                                    on_token(fragment)
                if chunks:
                    return "".join(chunks).strip()
                data = self._post_chat_completion(payload, allow_response_format_fallback=False)
                content = data["choices"][0]["message"]["content"]
                return self._sanitize_model_content(content)
            except httpx.TimeoutException as ex:
                if attempt < attempts - 1:
                    time.sleep(self.config.timeout_retry_backoff_seconds)
                    continue
                raise RuntimeError(
                    f"LM Studio request timed out after {attempts} attempt(s). "
                    f"Current timeout is {self.config.timeout_seconds:.1f}s. "
                    f"Set AUTOCAD_MCP_LMSTUDIO_TIMEOUT to a higher value if needed."
                ) from ex
            except httpx.HTTPStatusError as ex:
                status = ex.response.status_code
                body = _safe_response_text(ex.response).lower()
                if status == 400:
                    data = self._post_chat_completion(payload, allow_response_format_fallback=False)
                    content = data["choices"][0]["message"]["content"]
                    return self._sanitize_model_content(content)
                raise RuntimeError(_format_http_error(ex)) from ex
            except httpx.HTTPError as ex:
                raise RuntimeError(f"LM Studio request failed: {ex}") from ex

        raise RuntimeError("LM Studio streaming request failed unexpectedly.")

    @staticmethod
    def _iter_stream_fragments(text: str) -> list[str]:
        if not text:
            return []
        fragments = re.findall(r"\s+|[^\s]+", text)
        return fragments or [text]

    @staticmethod
    def _sanitize_model_content(content: object, trim: bool = True) -> str:
        text = content if isinstance(content, str) else str(content)
        cleaned = _UNUSED_TOKEN_PATTERN.sub("", text)
        return cleaned.strip() if trim else cleaned

    def _post_chat_completion(self, payload: dict, allow_response_format_fallback: bool) -> dict:
        attempts = 1 + max(0, self.config.timeout_retry_count)
        for attempt in range(attempts):
            try:
                resp = self._client.post(self._url("/chat/completions"), json=payload)
                resp.raise_for_status()
                return resp.json()
            except httpx.TimeoutException as ex:
                if attempt < attempts - 1:
                    time.sleep(self.config.timeout_retry_backoff_seconds)
                    continue
                raise RuntimeError(
                    f"LM Studio request timed out after {attempts} attempt(s). "
                    f"Current timeout is {self.config.timeout_seconds:.1f}s. "
                    f"Set AUTOCAD_MCP_LMSTUDIO_TIMEOUT to a higher value if needed."
                ) from ex
            except httpx.HTTPStatusError as ex:
                status = ex.response.status_code
                body = _safe_response_text(ex.response).lower()
                if status == 400 and allow_response_format_fallback:
                    # Some LM Studio model/providers reject response_format=json_object,
                    # and some return empty 400 bodies.
                    if "response_format" in payload or "response_format" in body or "json_object" in body:
                        fallback_payload = dict(payload)
                        fallback_payload.pop("response_format", None)
                        return self._post_chat_completion(fallback_payload, allow_response_format_fallback=False)
                if status == 400 and ("image_url" in body or "vision" in body or "image" in body):
                    raise RuntimeError(
                        "LM Studio rejected image input for this model/provider. "
                        "Load a vision-capable model and retry."
                    ) from ex
                raise RuntimeError(_format_http_error(ex)) from ex
            except httpx.HTTPError as ex:
                raise RuntimeError(f"LM Studio request failed: {ex}") from ex

        # Defensive fallback; loop always returns/raises.
        raise RuntimeError("LM Studio request failed unexpectedly.")

    @staticmethod
    def _image_path_to_data_url(path: str) -> str:
        image_path = Path(path)
        if not image_path.exists():
            raise RuntimeError(f"Image path does not exist: {path}")
        mime, _ = mimetypes.guess_type(str(image_path))
        if not mime or not mime.startswith("image/"):
            mime = "image/png"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    @staticmethod
    def _looks_repetitive(content: object) -> bool:
        text = str(content or "").strip()
        if len(text) < 600:
            return False

        normalized = re.sub(r"\s+", " ", text).strip().lower()
        if len(normalized) < 600:
            return False

        tail_window = 120
        if len(normalized) >= tail_window * 4:
            tail = normalized[-tail_window:]
            if tail and normalized.count(tail) >= 4:
                return True

        tokens = re.findall(r"\S+", normalized)
        repeat_window = 40
        if len(tokens) >= repeat_window * 4:
            trailing = tokens[-(repeat_window * 4):]
            if (
                trailing[0:repeat_window] == trailing[repeat_window : repeat_window * 2]
                and trailing[0:repeat_window] == trailing[repeat_window * 2 : repeat_window * 3]
                and trailing[0:repeat_window] == trailing[repeat_window * 3 :]
            ):
                return True

        return False


def _loads_json(content: str) -> dict:
    """Parse model JSON with cleanup/recovery for common wrapper patterns."""
    text = _UNUSED_TOKEN_PATTERN.sub("", content).strip()
    if not text:
        raise json.JSONDecodeError("Empty JSON payload", text, 0)
    fenced = _strip_markdown_fence(text)
    parse_candidates: list[str] = [fenced]
    embedded = _extract_balanced_json_candidate(fenced)
    if embedded and embedded not in parse_candidates:
        parse_candidates.append(embedded)
    last_error: json.JSONDecodeError | None = None
    for candidate in parse_candidates:
        try:
            parsed = json.loads(candidate.strip())
        except json.JSONDecodeError as ex:
            last_error = ex
            continue
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"actions": parsed}
        return {"value": parsed}
    if last_error is not None:
        raise last_error
    raise json.JSONDecodeError("Unable to decode JSON payload", text, 0)


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    without_opening = stripped[3:]
    newline_index = without_opening.find("\n")
    if newline_index >= 0:
        header = without_opening[:newline_index].strip().lower()
        body = without_opening[newline_index + 1 :]
        if header in ("", "json", "javascript", "js"):
            stripped = body
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    return stripped.strip()


def _extract_balanced_json_candidate(text: str) -> str | None:
    start_positions = [idx for idx, ch in enumerate(text) if ch in "{["]
    for start in start_positions:
        end = _find_balanced_json_end(text, start)
        if end is None:
            continue
        candidate = text[start : end + 1].strip()
        if candidate:
            return candidate
    return None


def _find_balanced_json_end(text: str, start_index: int) -> int | None:
    openers = {"{": "}", "[": "]"}
    closers = {"}": "{", "]": "["}
    stack: list[str] = []
    in_string = False
    escape = False
    for idx in range(start_index, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == "\"":
                in_string = False
            continue
        if ch == "\"":
            in_string = True
            continue
        if ch in openers:
            stack.append(ch)
            continue
        if ch in closers:
            if not stack:
                return None
            opener = stack.pop()
            if opener != closers[ch]:
                return None
            if not stack:
                return idx
    return None


def _format_http_error(ex: httpx.HTTPStatusError) -> str:
    status = ex.response.status_code
    body = _safe_response_text(ex.response).strip()
    if len(body) > 600:
        body = body[:600] + "..."
    return f"LM Studio HTTP {status}: {body or '<empty response body>'}"


def _safe_response_text(response: httpx.Response) -> str:
    try:
        return response.text or ""
    except RuntimeError:
        pass
    except Exception:
        pass

    try:
        raw = response.read()
    except Exception:
        return ""

    if isinstance(raw, bytes):
        try:
            return raw.decode(response.encoding or "utf-8", errors="replace")
        except Exception:
            return raw.decode("utf-8", errors="replace")
    return str(raw or "")
