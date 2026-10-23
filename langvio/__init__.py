"""
langvio: Connect language models to vision models for natural language visual analysis
"""

__version__ = "0.0.9"

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
from langvio.vision.qwen_vlm import QwenVisionProcessor
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

# Register Qwen2.5-VL vision processors (default). They share weights with the
# qwen LLM processors, so the model is only loaded once.
_vlm_vision_mapping = {
    "qwen_vlm": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen_vlm_3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen_vlm_7b": "Qwen/Qwen2.5-VL-7B-Instruct",
}
for proc_name, model_name in _vlm_vision_mapping.items():
    registry.register_vision_processor(
        proc_name, QwenVisionProcessor, model_name=model_name
    )

# Legacy YOLO-World vision processors (require ultralytics)
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

# The local VLM (Qwen2.5-VL) aliases are registered by register_llm_processors()
# above; there are no cloud-provider config aliases to register here anymore.


# === Pipeline Creator ===


def create_pipeline(config_path=None, llm_name=None, vision_name=None):  # noqa: C901
    """
    Create a pipeline with optional configuration.

    Args:
        config_path: Path to a configuration file
        llm_name: Name of LLM processor to use
        vision_name: Name of vision processor to use (default: config's
            vision.default, "qwen_vlm")

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
                # Only fall back to Qwen-VL processors; other detectors (YOLO)
                # are used solely when requested by name.
                preferred = ["qwen_vlm", "qwen_vlm_3b", "qwen_vlm_7b"]
                for proc_name in [p for p in preferred if p in valid_processors]:
                    try:
                        pipeline.set_vision_processor(proc_name)
                        vision_set = True
                        break
                    except Exception:
                        continue

            if not vision_set:
                error_msg = (
                    "ERROR: Could not set up the Qwen-VL vision processor "
                    "(qwen_vlm). Set OPENROUTER_API_KEY for the hosted Qwen API, "
                    "or pip install langvio[vlm] to run it locally."
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
                    "ERROR: No Qwen-VL LLM processor is available. Set "
                    "OPENROUTER_API_KEY for the hosted Qwen API, or install the "
                    "local backend:\n"
                    "- pip install langvio[vlm]"
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
