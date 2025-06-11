"""
langvio: Connect language models to vision models for natural language visual analysis
"""

__version__ = "0.3.0"

# Try to load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import cv2
import torch

# OpenCV optimizations
cv2.setNumThreads(4)  # Adjust based on your CPU
cv2.setUseOptimized(True)

# PyTorch optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Initialize the global model registry
from langvio.core.registry import ModelRegistry
registry = ModelRegistry()

# Import main components for easier access
from langvio.core.pipeline import Pipeline
from langvio.llm.base import BaseLLMProcessor
from langvio.vision.base import BaseVisionProcessor

# Register YOLOe and YOLO processors
from langvio.vision.yolo.detector import YOLOProcessor

# Register the YOLO processor with different configurations
registry.register_vision_processor(
    "yolo",
    YOLOProcessor,
    model_path="yolo11n.pt",
    confidence=0.5
)

# Register the YOLOe processor with different sizes
registry.register_vision_processor(
    "yoloe",
    YOLOProcessor,
    model_path="yoloe-11s-seg-pf.pt",
    confidence=0.5,
    model_type="yoloe"
)

registry.register_vision_processor(
    "yoloe_medium",
    YOLOProcessor,
    model_path="yoloe-11m-seg-pf.pt",
    confidence=0.5,
    model_type="yoloe"
)

registry.register_vision_processor(
    "yoloe_large",
    YOLOProcessor,
    model_path="yoloe-11l-seg-pf.pt",
    confidence=0.5,
    model_type="yoloe"
)

# Register LLM processors using the factory
from langvio.llm.factory import register_llm_processors
register_llm_processors(registry)


# Default pipeline creator with better defaults
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
    import sys

    # Create the pipeline
    pipeline = Pipeline(config_path)

    # Set the vision processor (use YOLOe by default)
    if vision_name:
        pipeline.set_vision_processor(vision_name)
    else:
        # Default to YOLOe for best performance
        try:
            pipeline.set_vision_processor("yoloe_large")
        except:
            # Fall back to YOLO if YOLOe is not available
            pipeline.set_vision_processor("yolo")

    # Set the LLM processor
    if llm_name:
        # This will exit if the processor is not available
        pipeline.set_llm_processor(llm_name)
    else:
        # If no specific LLM is requested, try to use the default from config
        try:
            default_llm = pipeline.config.config["llm"]["default"]
            pipeline.set_llm_processor(default_llm)
        except Exception:
            # If we can't set a default LLM, check if any LLMs are available
            if len(registry.list_llm_processors()) == 0:
                error_msg = (
                    "ERROR: No LLM providers are installed. Please install at least one provider:\n"
                    "- For OpenAI: pip install langvio[openai]\n"
                    "- For Google Gemini: pip install langvio[google]\n"
                    "- For all providers: pip install langvio[all-llm]"
                )
                print(error_msg, file=sys.stderr)
                sys.exit(1)
            else:
                # Use the first available LLM
                available_llm = next(iter(registry.list_llm_processors()))
                pipeline.set_llm_processor(available_llm)

    return pipeline


# Version info and exports
__all__ = [
    "Pipeline",
    "create_pipeline",
    "registry",
    "BaseLLMProcessor",
    "BaseVisionProcessor",
]