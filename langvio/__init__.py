"""
langvio: Connect language models to vision models for natural language visual analysis
"""

__version__ = "0.3.0"

# === Imports ===

# Standard library
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
from langvio.vision.yolo.detector import YOLOProcessor

# === Initialization ===

# Load environment variables
load_dotenv()

# OpenCV optimizations
cv2.setNumThreads(4)
cv2.setUseOptimized(True)

# PyTorch optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Initialize model registry
registry = ModelRegistry()

# Register YOLO and YOLOe vision processors
registry.register_vision_processor(
    "yolo", YOLOProcessor, model_path="yolo11n.pt", confidence=0.5
)

registry.register_vision_processor(
    "yoloe",
    YOLOProcessor,
    model_path="yoloe-11s-seg-pf.pt",
    confidence=0.5,
    model_type="yoloe",
)

registry.register_vision_processor(
    "yoloe_medium",
    YOLOProcessor,
    model_path="yoloe-11m-seg-pf.pt",
    confidence=0.5,
    model_type="yoloe",
)

registry.register_vision_processor(
    "yoloe_large",
    YOLOProcessor,
    model_path="yoloe-11l-seg-pf.pt",
    confidence=0.5,
    model_type="yoloe",
)

# Register LLM processors
register_llm_processors(registry)


# === Pipeline Creator ===


def create_pipeline(config_path=None, llm_name=None, vision_name=None):
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
        try:
            pipeline.set_vision_processor("yoloe_large")
        except Exception:
            pipeline.set_vision_processor("yolo")

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
