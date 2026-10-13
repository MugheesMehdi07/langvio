"""Tests for the hosted Qwen API backend (no network access)."""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from langvio.llm.qwen_api import QwenAPIClient, bbox_space, resolve_backend
from langvio.vision.qwen_vlm.detector import QwenVisionProcessor


def test_bbox_space_by_model_family():
    assert bbox_space("Qwen/Qwen2.5-VL-3B-Instruct") == "pixel"
    assert bbox_space("qwen/qwen2.5-vl-72b-instruct") == "pixel"
    assert bbox_space("qwen/qwen3-vl-30b-a3b-instruct") == "norm1000"
    assert bbox_space("qwen/qwen3-vl-30b-a3b-instruct", "pixel") == "pixel"


def test_resolve_backend_defaults_to_auto(monkeypatch):
    """By default the API is used when a key is configured, else local weights."""
    monkeypatch.delenv("LANGVIO_BACKEND", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    assert resolve_backend({}) == "local"
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    assert resolve_backend({}) == "api"


def test_resolve_backend_explicit_choices(monkeypatch):
    monkeypatch.delenv("LANGVIO_BACKEND", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    assert resolve_backend({"backend": "local"}) == "local"
    # LANGVIO_BACKEND wins over the processor config.
    monkeypatch.setenv("LANGVIO_BACKEND", "local")
    assert resolve_backend({"backend": "api"}) == "local"
    monkeypatch.setenv("LANGVIO_BACKEND", "api")
    assert resolve_backend({"backend": "local"}) == "api"


def test_resolve_backend_auto_follows_key(monkeypatch):
    """"auto" keeps the legacy behaviour: API only when a key is present."""
    monkeypatch.delenv("LANGVIO_BACKEND", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    assert resolve_backend({"backend": "auto"}) == "local"
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    assert resolve_backend({"backend": "auto"}) == "api"


def test_api_settings_prefers_config_over_env(monkeypatch):
    from langvio.llm.qwen_api import DEFAULT_API_MODEL, api_settings

    monkeypatch.setenv("QWEN_API_MODEL", "from/env")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env")
    assert api_settings({})["model"] == "from/env"
    assert api_settings({"api_model": "from/config"})["model"] == "from/config"

    monkeypatch.delenv("QWEN_API_MODEL", raising=False)
    assert api_settings({})["model"] == DEFAULT_API_MODEL
    # A per-model api_key_env lets different processors use different keys.
    monkeypatch.setenv("MY_KEY", "sk-custom")
    assert api_settings({"api_key_env": "MY_KEY"})["api_key"] == "sk-custom"


def test_client_requires_key():
    with pytest.raises(ValueError):
        QwenAPIClient(api_key=None)


def _response(status, payload=None, text=""):
    response = MagicMock(status_code=status, text=text)
    response.json.return_value = payload or {}
    return response


def test_chat_retries_then_succeeds():
    client = QwenAPIClient(api_key="sk-test", max_retries=2)
    ok = _response(200, {"choices": [{"message": {"content": "hi"}}]})
    with patch("langvio.llm.qwen_api.requests.post") as post, patch(
        "langvio.llm.qwen_api.time.sleep"
    ):
        post.side_effect = [_response(503, text="busy"), ok]
        assert client.chat([{"role": "user", "content": "hello"}]) == "hi"
        body = post.call_args.kwargs["json"]
        assert body["reasoning"] == {"enabled": False}


def test_chat_stops_on_daily_limit():
    client = QwenAPIClient(api_key="sk-test", max_retries=3)
    limited = _response(429, text="Rate limit exceeded: free-models-per-day")
    with patch("langvio.llm.qwen_api.requests.post", return_value=limited) as post:
        with pytest.raises(RuntimeError):
            client.chat([{"role": "user", "content": "hello"}])
        assert post.call_count == 1


def test_rate_limit_error_is_summarised():
    client = QwenAPIClient(api_key="sk-test", max_retries=0)
    limited = _response(
        429,
        {"error": {"metadata": {"raw": "x is temporarily rate-limited upstream."}}},
    )
    with patch("langvio.llm.qwen_api.requests.post", return_value=limited):
        with pytest.raises(RuntimeError, match="rate-limited upstream"):
            client.chat([{"role": "user", "content": "hello"}])


def test_video_skips_failed_keyframes(tmp_path):
    import cv2
    import numpy as np

    path = str(tmp_path / "clip.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 48))
    for _ in range(15):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()

    processor = QwenVisionProcessor(
        "qwen_vlm", backend="api", scene_description=False, keyframes_per_second=1
    )
    processor.api_client = QwenAPIClient(api_key="sk-test")
    processor.model = processor.api_client
    calls = {"n": 0}

    def flaky_ground(image, params):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("Qwen API request failed: HTTP 429")
        return []

    with patch.object(processor, "_ground", side_effect=flaky_ground):
        result = processor.process_video(path, {"target_objects": ["person"]})
    assert "error" not in result
    assert result["processing_info"]["frames_failed"] == 1


def test_api_grounding_scales_norm1000_boxes():
    processor = QwenVisionProcessor("qwen_vlm", backend="api")
    processor.api_client = QwenAPIClient(api_key="sk-test")
    processor.model = processor.api_client
    answer = '[{"bbox_2d": [100, 200, 300, 600], "label": "person"}]'
    with patch.object(processor.api_client, "chat", return_value=answer):
        detections = processor._ground(
            Image.new("RGB", (2000, 1000)), {"target_objects": ["person"]}
        )
    assert detections[0]["bbox"] == [200, 200, 600, 600]


def test_register_without_key_or_vlm_extra_exits(monkeypatch):
    """No key and no transformers: exit with a message naming both fixes."""
    import pytest

    from langvio.llm import factory

    monkeypatch.delenv("LANGVIO_BACKEND", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    monkeypatch.setattr(factory, "is_package_available", lambda name: False)
    with pytest.raises(SystemExit):
        factory.register_llm_processors(None)


def test_register_without_key_falls_back_to_local(monkeypatch, caplog):
    """No key with transformers installed: register for local and warn."""
    from unittest.mock import MagicMock

    from langvio.llm import factory

    monkeypatch.delenv("LANGVIO_BACKEND", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    monkeypatch.setattr(factory, "is_package_available", lambda name: True)
    registry = MagicMock()
    with caplog.at_level("WARNING", logger="langvio.llm.factory"):
        factory.register_llm_processors(registry)
    assert registry.register_llm_processor.called
    assert "running Qwen-VL locally" in caplog.text
