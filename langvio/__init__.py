"""
langvio: Connect language models to vision models for natural language visual analysis
"""

__version__ = "0.0.2"

# === Imports ===

# Standard library
import os
import sys

# Third-party
import cv2
import torch
from dotenv import load_dotenv

from langvio.core.pipeline import Pipeline

# langvio modules
from langvio.core.registry import ModelRegistry
from langvio.llm.base import BaseLLMProcessor
from langvio.llm.factory import register_llm_processors
from langvio.vision.base import BaseVisionProcessor
from langvio.vision.yolo_world.detector import YOLOWorldProcessor

# === Initialization ===

# Load environment variables from .env file
# Ensure .env is loaded before any other imports that might need env vars

# Try to find .env file in multiple locations:
# 1. Current working directory
# 2. langvio package directory
# 3. Parent directory (project root)
env_paths = [
    os.path.join(os.getcwd(), ".env"),  # Current working directory
    os.path.join(os.path.dirname(__file__), "..", ".env"),  # langvio/.env
    os.path.join(os.path.dirname(__file__), "..", "..", ".env"),  # project root/.env
]

for env_path in env_paths:
    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)
        break
else:
    # If no .env file found, try default load_dotenv() behavior
    load_dotenv(override=True)


# OpenCV optimizations
cv2.setNumThreads(4)
cv2.setUseOptimized(True)

# PyTorch optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Initialize model registry
registry = ModelRegistry()

# Register YOLO-World vision processors
registry.register_vision_processor(
    "yolo_world_v2_s", YOLOWorldProcessor, model_name="yolov8s-worldv2", confidence=0.5
)

registry.register_vision_processor(
    "yolo_world_v2_m", YOLOWorldProcessor, model_name="yolov8m-worldv2", confidence=0.45
)

registry.register_vision_processor(
    "yolo_world_v2_l", YOLOWorldProcessor, model_name="yolov8l-worldv2", confidence=0.5
)

registry.register_vision_processor(
    "yolo_world_v2x", YOLOWorldProcessor, model_name="yolov8x-worldv2", confidence=0.5
)

# Register config-based processors as aliases
# Map config processor names to YOLO-World model names
_config_processor_mapping = {
    "yolo11n": ("yolov8s-worldv2", 0.8),
    "yolo": ("yolov8s-worldv2", 0.7),
    "yolo_medium": ("yolov8m-worldv2", 0.5),
    "yolo_large": ("yolov8x-worldv2", 0.5),
    "yoloe": ("yolov8s-worldv2", 0.8),
    "yoloe_medium": ("yolov8m-worldv2", 0.5),
    "yoloe_large": ("yolov8l-worldv2", 0.5),
}

for proc_name, (model_name, confidence) in _config_processor_mapping.items():
    registry.register_vision_processor(
        proc_name, YOLOWorldProcessor, model_name=model_name, confidence=confidence
    )

# Register LLM processors
register_llm_processors(registry)

# Register config-based LLM processors as aliases (if OpenAI is available)
# These are additional GPT models that may be in config but not registered by factory
try:
    import importlib.util
    import logging

    logger = logging.getLogger(__name__)

    if importlib.util.find_spec("langchain_openai") is not None:
        from langvio.llm.openai import OpenAIProcessor

        # Register additional GPT models from config
        _config_llm_mapping = {
            "gpt-4.1-mini": ("gpt-4.1-mini", {"temperature": 0.2, "max_tokens": 1024}),
            "gpt-4.1-mini-latest": (
                "gpt-4.1-mini-2025-04-14",
                {"temperature": 0.2, "max_tokens": 1024},
            ),
            "gpt-4.1-nano": ("gpt-4.1-nano", {"temperature": 0.2, "max_tokens": 1024}),
            "gpt-4.1-nano-latest": (
                "gpt-4.1-nano-2025-04-14",
                {"temperature": 0.2, "max_tokens": 1024},
            ),
            "gpt-4o-mini": ("gpt-4o-mini", {"temperature": 0.2, "max_tokens": 1024}),
            "gpt-4o-mini-latest": (
                "gpt-4o-mini-2024-07-18",
                {"temperature": 0.2, "max_tokens": 1024},
            ),
            "gpt-5-nano": ("gpt-5-nano", {"temperature": 0.1, "max_tokens": 2048}),
            "gpt-5-nano-latest": (
                "gpt-5-nano-2025-08-07",
                {"temperature": 0.1, "max_tokens": 2048},
            ),
            "gpt-3": (
                "gpt-3.5-turbo",
                {"temperature": 0.0, "max_tokens": 1024},
            ),  # Alias for gpt-3.5
        }

        registered_count = 0
        for proc_name, (model_name, model_kwargs) in _config_llm_mapping.items():
            # Only register if not already registered
            if proc_name not in registry.list_llm_processors():
                registry.register_llm_processor(
                    proc_name,
                    OpenAIProcessor,
                    model_name=model_name,
                    model_kwargs=model_kwargs,
                )
                registered_count += 1

        if registered_count > 0:
            logger.debug(
                f"Registered {registered_count} additional config-based LLM processors"
            )
    else:
        logger.debug(
            "langchain_openai not available, "
            "skipping config-based LLM processor registration"
        )
except Exception as e:
    # Log the error for debugging
    import logging

    logger = logging.getLogger(__name__)
    logger.debug(f"Failed to register config-based LLM processors: {e}")


# === Pipeline Creator ===


def create_pipeline(config_path=None, llm_name=None, vision_name=None):  # noqa: C901
    """
    Create a pipeline with optional configuration.

    Args:
        config_path: Path to a configuration file
        llm_name: Name of LLM processor to use
        vision_name: Name of vision processor to use (default: "yoloe")

    Returns:
        A configured Pipeline instance
    """
    pipeline = Pipeline(config_path)

    if vision_name:
        pipeline.set_vision_processor(vision_name)
    else:
        # Try to use the default vision processor from config
        vision_set = False
        try:
            default_vision = pipeline.config.config.get("vision", {}).get("default")
            if default_vision:
                # Check if it's both registered and in config
                if default_vision in registry.list_vision_processors():
                    pipeline.set_vision_processor(default_vision)
                    vision_set = True
        except Exception:
            pass

        if not vision_set:
            # Fallback: try registered processors that are also in config
            available_vision = list(registry.list_vision_processors().keys())
            config_vision_models = (
                pipeline.config.config.get("vision", {}).get("models", {}).keys()
            )

            # Find processors that are both registered and in config
            valid_processors = [
                v for v in available_vision if v in config_vision_models
            ]

            if valid_processors:
                # Try preferred processors first
                preferred = ["yolo_world_v2_m", "yolo_world_v2_s", "yolo_world_v2_l"]
                for proc_name in preferred + [
                    v for v in valid_processors if v not in preferred
                ]:
                    try:
                        pipeline.set_vision_processor(proc_name)
                        vision_set = True
                        break
                    except Exception:
                        continue

            if not vision_set:
                error_msg = (
                    f"ERROR: No valid vision processor found. "
                    f"Registered processors: {list(available_vision)}\n"
                    f"Config processors: {list(config_vision_models)}\n"
                    f"Please ensure at least one vision processor is "
                    f"both registered and configured."
                )
                print(error_msg, file=sys.stderr)
                sys.exit(1)

    if llm_name:
        pipeline.set_llm_processor(llm_name)
    else:
        try:
            default_llm = pipeline.config.config["llm"]["default"]
            pipeline.set_llm_processor(default_llm)
        except Exception:
            if len(registry.list_llm_processors()) == 0:
                error_msg = (
                    "ERROR: No LLM providers are installed. "
                    "Please install at least one provider:\n"
                    "- For OpenAI: pip install langvio[openai]\n"
                    "- For Google Gemini: pip install langvio[google]\n"
                    "- For all providers: pip install langvio[all-llm]"
                )
                print(error_msg, file=sys.stderr)
                sys.exit(1)
            else:
                available_llm = next(iter(registry.list_llm_processors()))
                pipeline.set_llm_processor(available_llm)

    return pipeline


# === Public Exports ===

__all__ = [
    "Pipeline",
    "create_pipeline",
    "registry",
    "BaseLLMProcessor",
    "BaseVisionProcessor",
]
