"""
Remote Qwen-VL backend over an OpenAI-compatible chat completions API.

This is langvio's default backend: it runs the whole pipeline (query parsing,
grounding, scene description, explanation) against a hosted Qwen-VL model, so
nothing is downloaded locally. Works with any OpenAI-compatible endpoint:
OpenRouter (default), Alibaba Model Studio / DashScope, the Hugging Face
router or a self-hosted vLLM server.

Backend selection (``resolve_backend``):
    * ``LANGVIO_BACKEND`` env var ("api" / "local") wins if set;
    * otherwise the processor's ``backend`` setting, which defaults to "api";
    * "local" opts into the in-process weights (needs ``langvio[vlm]``);
    * "auto" keeps the older behaviour: API when a key is configured, else local.
"""

import base64
import io
import logging
import os
import random
import threading
import time
from typing import Any, Dict, List, Optional, Union

import requests
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_API_BASE_URL = "https://openrouter.ai/api/v1"
# Qwen3-VL 30B-A3B: a sparse MoE, so it is priced and served like a small model
# while grounding about as well as the dense 72B. Alternatives on the same
# endpoint: qwen/qwen3-vl-8b-instruct (cheaper), qwen/qwen3-vl-235b-a22b-instruct
# (strongest), qwen/qwen2.5-vl-72b-instruct (pixel-space boxes).
DEFAULT_API_MODEL = "qwen/qwen3-vl-30b-a3b-instruct"
DEFAULT_API_KEY_ENV = "OPENROUTER_API_KEY"

_RETRY_STATUS = {408, 429, 500, 502, 503, 504}

# Shared per (endpoint, model) so concurrent callers back off together.
_COOLDOWN_LOCK = threading.Lock()
_COOLDOWN_UNTIL: Dict[str, float] = {}


def api_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve API settings from processor config, then env vars, then defaults."""
    key_env = config.get("api_key_env") or DEFAULT_API_KEY_ENV
    return {
        "base_url": config.get("api_base_url")
        or os.environ.get("QWEN_API_BASE_URL")
        or DEFAULT_API_BASE_URL,
        "model": config.get("api_model")
        or os.environ.get("QWEN_API_MODEL")
        or DEFAULT_API_MODEL,
        "api_key": os.environ.get(key_env) or os.environ.get("QWEN_API_KEY"),
        "key_env": key_env,
        "extra_body": config.get("api_extra_body"),
        "timeout": float(config.get("api_timeout", 180)),
        "max_retries": int(config.get("api_max_retries", 6)),
    }


def resolve_backend(config: Dict[str, Any]) -> str:
    """Return "api" or "local" for a processor config (see module docstring)."""
    choice = (
        os.environ.get("LANGVIO_BACKEND") or config.get("backend") or "api"
    ).lower()
    if choice in ("api", "local"):
        return choice
    # "auto": fall back to local weights only when no key is configured.
    return "api" if api_settings(config)["api_key"] else "local"


def bbox_space(model: str, bbox_format: str = "auto") -> str:
    """Coordinate space of grounding output: "pixel" or "norm1000".

    Qwen2/2.5-VL answer in pixels of the image they were given; Qwen3 and
    later answer on a 0-1000 grid regardless of the image size.
    """
    if bbox_format in ("pixel", "norm1000"):
        return bbox_format
    name = model.lower()
    return "pixel" if ("qwen2" in name or "qwen-2" in name) else "norm1000"


def image_to_data_url(image: Image.Image, quality: int = 90) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


class QwenAPIClient:
    """Thin, thread-safe client for an OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        model: str = DEFAULT_API_MODEL,
        base_url: str = DEFAULT_API_BASE_URL,
        api_key: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: float = 180,
        max_retries: int = 6,
    ):
        if not api_key:
            raise ValueError(
                "No API key for the Qwen API backend. Get one at "
                "https://openrouter.ai/keys and set OPENROUTER_API_KEY (or "
                "QWEN_API_KEY) in your environment / .env file. To run the "
                "model locally instead, install 'pip install langvio[vlm]' "
                "and set LANGVIO_BACKEND=local."
            )
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        if extra_body is None and "openrouter.ai" in self.base_url:
            # Hybrid-thinking Qwen models otherwise spend the token budget on
            # reasoning before emitting the JSON / answer.
            extra_body = {"reasoning": {"enabled": False}}
        self.extra_body = extra_body or {}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "QwenAPIClient":
        s = api_settings(config)
        return cls(
            model=s["model"],
            base_url=s["base_url"],
            api_key=s["api_key"],
            extra_body=s["extra_body"],
            timeout=s["timeout"],
            max_retries=s["max_retries"],
        )

    @staticmethod
    def user_content(
        images: List[Image.Image], text: str
    ) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": image_to_data_url(img)}}
            for img in images
        ]
        content.append({"type": "text", "text": text})
        return content

    def chat(
        self,
        messages: List[Dict[str, Union[str, List[Dict[str, Any]]]]],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **self.extra_body,
        }
        if top_p is not None:
            payload["top_p"] = top_p
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": "langvio",
        }

        last_error = ""
        for attempt in range(self.max_retries + 1):
            self._wait_for_cooldown()
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
            except requests.RequestException as e:
                last_error = str(e)
            else:
                if response.status_code == 200:
                    data = response.json()
                    if "choices" in data and data["choices"]:
                        return data["choices"][0]["message"].get("content") or ""
                    last_error = str(data.get("error", data))[:300]
                else:
                    last_error = self._describe_error(response)
                    if response.status_code not in _RETRY_STATUS:
                        break
                    if self._daily_limit_hit(response):
                        break

            if attempt < self.max_retries:
                # Jitter keeps parallel keyframe requests from retrying in lockstep.
                delay = round(min(30, 2 ** attempt * 2) * random.uniform(0.7, 1.3), 1)
                if last_error.startswith("HTTP 429"):
                    # A rate limit applies to every in-flight request, so pause
                    # them all instead of letting each one keep hammering.
                    self._start_cooldown(delay)
                logger.warning(
                    f"Qwen API: {last_error} - retry {attempt + 1}/"
                    f"{self.max_retries} in {delay}s"
                )
                time.sleep(delay)

        raise RuntimeError(f"Qwen API request failed: {last_error}")

    @property
    def _cooldown_key(self) -> str:
        return f"{self.base_url}|{self.model}"

    def _start_cooldown(self, seconds: float) -> None:
        with _COOLDOWN_LOCK:
            until = time.monotonic() + seconds
            _COOLDOWN_UNTIL[self._cooldown_key] = max(
                _COOLDOWN_UNTIL.get(self._cooldown_key, 0.0), until
            )

    def _wait_for_cooldown(self) -> None:
        with _COOLDOWN_LOCK:
            remaining = _COOLDOWN_UNTIL.get(self._cooldown_key, 0.0) - time.monotonic()
        if remaining > 0:
            time.sleep(remaining + random.uniform(0, 0.5))

    @staticmethod
    def _describe_error(response: requests.Response) -> str:
        try:
            error = response.json().get("error", {})
            message = (error.get("metadata") or {}).get("raw") or error.get("message")
        except (ValueError, AttributeError):
            message = None
        message = " ".join(str(message or response.text).split())
        if "rate-limited upstream" in message:
            message = "free model is rate-limited upstream"
        return f"HTTP {response.status_code}: {message[:200]}"

    @staticmethod
    def _daily_limit_hit(response: requests.Response) -> bool:
        # Retrying cannot help once the free per-day quota is exhausted.
        text = response.text.lower()
        return response.status_code == 429 and "per-day" in text
