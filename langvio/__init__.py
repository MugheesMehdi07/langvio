"""
langvio: Connect language models to vision models for natural language visual analysis
"""

__version__ = "0.3.0"

# Try to load environment variables from .env file

from langvio.core.pipeline import Pipeline
from langvio.core.registry import ModelRegistry

# Initialize the global model registry
registry = ModelRegistry()

# Import main components for easier access
from langvio.llm.base import BaseLLMProcessor
from langvio.vision.base import BaseVisionProcessor

# Register the YOLO processor
from langvio.vision.yolo.detector import YOLOProcessor
registry.register_vision_processor("yolo", YOLOProcessor)
registry.register_vision_processor("yoloe_large", YOLOProcessor)
registry.register_vision_processor("yoloe", YOLOProcessor)


# Register LLM processors using the factory
from langvio.llm.factory import register_llm_processors
register_llm_processors(registry)


# Default pipeline creator
def create_pipeline(config_path=None, llm_name=None, vision_name=None):
    """
    Create a pipeline with optional configuration.

    Args:
        config_path: Path to a configuration file
        llm_name: Name of LLM processor to use
        vision_name: Name of vision processor to use

    Returns:
        A configured Pipeline instance
    """
    from langvio.core.pipeline import Pipeline
    import sys

    # Create the pipeline
    pipeline = Pipeline(config_path)

    # Set the vision processor (YOLO is always available)
    if vision_name:
        pipeline.set_vision_processor(vision_name)
    else:
        pipeline.set_vision_processor("yoloe_large")

    # Set the LLM processor if specified
    if llm_name:
        # This will exit if the processor is not available
        pipeline.set_llm_processor(llm_name)
    else:
        # If no specific LLM is requested, try to use the default from config
        try:
            default_llm = pipeline.config.config["llm"]["default"]
            pipeline.set_llm_processor(default_llm)
        except Exception as e:
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


# Version info
__all__ = [
    "Pipeline",
    "create_pipeline",
    "registry",
    "BaseLLMProcessor",
    "BaseVisionProcessor"
]