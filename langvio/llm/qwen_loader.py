"""
Shared loader for the local Qwen(2/2.5)-VL model.

Both the language processor (``VLMProcessor``) and the vision/grounding processor
(``QwenVisionProcessor``) run on the *same* Qwen2.5-VL weights. Loading a multi-GB
model twice would waste memory and time, so this module memoizes the loaded
(model, processor) pair per model name and hands the same instance to every
caller - making a genuine "one model does the whole cycle" setup.

Device selection prefers CUDA, then Apple Metal (MPS), then CPU. On MPS the
PyTorch CPU fallback is enabled so any op without an MPS kernel still runs.
"""

import logging
import os
import threading
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_VLM_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

# The shared model is not safe for concurrent generate() calls (e.g. from a
# threaded Flask server), so every caller must hold this lock while generating.
GENERATION_LOCK = threading.Lock()

# transformers class names to probe, in preference order (varies by release
# and by Qwen2-VL vs Qwen2.5-VL).
_MODEL_CLASS_CANDIDATES = [
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen2VLForConditionalGeneration",
    "AutoModelForImageTextToText",
    "AutoModelForVision2Seq",
]

# Cache: model_name -> (model, processor). Shared across all processors.
_CACHE: Dict[str, Tuple[Any, Any]] = {}


def select_device() -> str:
    """Pick the best available torch device: cuda > mps > cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        # Let unsupported ops fall back to CPU instead of erroring.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps"
    return "cpu"


def _load_model_class(model_name: str, load_kwargs: Dict[str, Any]) -> Any:
    import transformers

    last_error: Optional[Exception] = None
    for class_name in _MODEL_CLASS_CANDIDATES:
        model_class = getattr(transformers, class_name, None)
        if model_class is None:
            continue
        try:
            model = model_class.from_pretrained(model_name, **load_kwargs)
            logger.debug(f"Loaded Qwen via {class_name}")
            return model
        except Exception as e:  # noqa: BLE001 - try next candidate
            last_error = e
            logger.debug(f"{class_name} failed for {model_name}: {e}")
    raise ImportError(
        f"Could not load '{model_name}' with any known Qwen-VL model class "
        f"({', '.join(_MODEL_CLASS_CANDIDATES)}). Ensure 'transformers' is recent "
        f"enough for this model. Last error: {last_error}"
    )


def load_qwen(
    model_name: str = DEFAULT_VLM_MODEL,
    load_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Any]:
    """
    Load (or return the cached) Qwen model + processor for ``model_name``.

    Args:
        model_name: HuggingFace repo id.
        load_kwargs: Extra kwargs forwarded to ``from_pretrained``. ``torch_dtype``
            and device placement are chosen automatically if not supplied.

    Returns:
        (model, processor) tuple, shared across all callers for this model name.
    """
    if model_name in _CACHE:
        return _CACHE[model_name]

    import torch
    from transformers import AutoProcessor

    load_kwargs = dict(load_kwargs or {})
    device = select_device()

    # dtype: fp16 on GPU/MPS, fp32 on CPU (unless the caller overrode it).
    if "torch_dtype" not in load_kwargs:
        load_kwargs["torch_dtype"] = (
            torch.float16 if device in ("cuda", "mps") else torch.float32
        )

    logger.info(
        f"Loading shared Qwen model '{model_name}' on {device} "
        f"(dtype={load_kwargs['torch_dtype']}) - this may take a while"
    )
    model = _load_model_class(model_name, load_kwargs)
    model = model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(model_name)

    _CACHE[model_name] = (model, processor)
    logger.info(f"Shared Qwen model ready: {model_name} on {device}")
    return _CACHE[model_name]
