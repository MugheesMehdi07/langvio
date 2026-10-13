"""
Factory for LLM processor registration and creation.

langvio uses a Qwen-VL Vision Language Model as its language layer. When an
API key (OpenRouter) is configured it is reached over an OpenAI-compatible API;
without one the same processor runs the weights in-process, which needs
``pip install langvio[vlm]``. ``LANGVIO_BACKEND=api|local`` forces either.
"""

import importlib.util
import logging
import os
import sys

from langvio.core.registry import ModelRegistry

logger = logging.getLogger(__name__)

def _gen_kwargs() -> dict:
    """Fresh copy per registration - processors may mutate their own kwargs."""
    return {"temperature": 0.2, "max_new_tokens": 1024}


# Registry name -> (model id, default generation kwargs). ``model_name`` is the
# HuggingFace repo id used by the local backend; with the API backend (the
# default) the model is chosen by api_model / QWEN_API_MODEL instead, so these
# names are aliases for the same processor.
_VLM_MODELS = {
    "qwen": ("Qwen/Qwen2.5-VL-3B-Instruct", _gen_kwargs()),
    "qwen-3b": ("Qwen/Qwen2.5-VL-3B-Instruct", _gen_kwargs()),
    "qwen-7b": ("Qwen/Qwen2.5-VL-7B-Instruct", _gen_kwargs()),
    "qwen2-vl-7b": ("Qwen/Qwen2-VL-7B-Instruct", _gen_kwargs()),
}


def register_llm_processors(registry: ModelRegistry) -> None:
    """
    Register available LLM processors with the registry.

    A single Qwen-VL processor is registered under several aliases. Registration
    itself is backend-agnostic and never loads a model; the backend is resolved
    when the processor is initialized. Only the local backend needs
    'transformers', so its absence is an error solely when local is requested.

    Args:
        registry: The model registry to register processors with
    """
    from langvio.llm.qwen_api import resolve_backend

    backend = resolve_backend({})
    # Local because no key was found, rather than because the user asked for it.
    fell_back = (
        backend == "local"
        and os.environ.get("LANGVIO_BACKEND", "auto").lower() == "auto"
    )
    if backend == "local" and not is_package_available("transformers"):
        if fell_back:
            error_msg = (
                "ERROR: No API key found (OPENROUTER_API_KEY / QWEN_API_KEY), so "
                "langvio tried to run Qwen-VL locally, but the local VLM backend "
                "is not installed. Either:\n"
                "  - set OPENROUTER_API_KEY in your environment / .env file "
                "(get one at https://openrouter.ai/keys), or\n"
                "  - pip install langvio[vlm]  to run the model on this machine."
            )
        else:
            error_msg = (
                "ERROR: The local VLM backend is not installed.\n"
                "Please install it with:\n"
                "  pip install langvio[vlm]\n"
                "(this pulls in 'transformers' and 'accelerate'),\n"
                "or unset LANGVIO_BACKEND and set OPENROUTER_API_KEY to use the "
                "hosted Qwen API instead."
            )
        logger.error(error_msg)
        sys.exit(1)
    if fell_back:
        logger.warning(
            "No API key found (OPENROUTER_API_KEY / QWEN_API_KEY): running "
            "Qwen-VL locally. The first run downloads the model weights from "
            "HuggingFace (several GB); a GPU is strongly recommended."
        )

    try:
        from langvio.llm.qwen_vlm import VLMProcessor

        for name, (model_name, model_kwargs) in _VLM_MODELS.items():
            registry.register_llm_processor(
                name,
                VLMProcessor,
                model_name=model_name,
                model_kwargs=model_kwargs,
            )

        # Default processor: the small, most accessible model.
        registry.register_llm_processor(
            "default",
            VLMProcessor,
            model_name=_VLM_MODELS["qwen"][0],
            model_kwargs=_VLM_MODELS["qwen"][1],
        )

        logger.info(f"Registered Qwen-VL LLM processors (backend: {backend})")
    except Exception as e:
        error_msg = f"ERROR: Failed to register the Qwen-VL processor: {e}"
        logger.error(error_msg)
        sys.exit(1)


def is_package_available(package_name: str) -> bool:
    """
    Check if a Python package is available.

    Args:
        package_name: Name of the package to check

    Returns:
        True if the package is available, False otherwise
    """
    return importlib.util.find_spec(package_name) is not None
