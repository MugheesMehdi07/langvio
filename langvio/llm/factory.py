"""
Factory for LLM processor registration and creation
"""

import importlib.util
import logging
import sys

from langvio.core.registry import ModelRegistry

logger = logging.getLogger(__name__)


def register_llm_processors(registry: ModelRegistry) -> None:
    """
    Register available LLM processors with the registry.
    This function performs lazy loading to avoid unnecessary imports.
    If no LLM providers are available, it will log an error but NOT register a fallback.

    Args:
        registry: The model registry to register processors with
    """
    llm_providers_found = False

    # Register OpenAI processor if available
    if is_package_available("langchain_openai"):
        try:
            from langvio.llm.openai import OpenAIProcessor

            # Register GPT-3.5 model
            registry.register_llm_processor(
                "gpt-3.5",
                OpenAIProcessor,
                model_name="gpt-3.5-turbo",
                model_kwargs={"temperature": 0.2},
            )

            # Register GPT-4 model
            registry.register_llm_processor(
                "gpt-4",
                OpenAIProcessor,
                model_name="gpt-4-turbo",
                model_kwargs={"temperature": 0.2},
            )

            llm_providers_found = True
            logger.info("Registered OpenAI LLM processors")
        except Exception as e:
            logger.warning(f"Failed to register OpenAI processors: {e}")

    # Register Google Gemini processor if available
    if is_package_available("langchain_google_genai"):
        try:
            from langvio.llm.google import GeminiProcessor

            # # Register Gemini Pro model
            registry.register_llm_processor(
                "gemini",
                GeminiProcessor,
                model_name="gemini-pro",
                model_kwargs={"temperature": 0.2},
            )

            # Set as default if available
            registry.register_llm_processor(
                "default",
                GeminiProcessor,
                model_name="gemini-pro",
                model_kwargs={"temperature": 0.2},
            )

            llm_providers_found = True
            logger.info("Registered Google Gemini LLM processors")
        except Exception as e:
            logger.warning(f"Failed to register Google Gemini processors: {e}")

    # If no LLM providers are available, log an error but DO NOT register a fallback
    if not llm_providers_found:
        error_msg = (
            "ERROR: No LLM providers are installed. "
            "Please install at least one provider:\n"
            "- For OpenAI: pip install langvio[openai]\n"
            "- For Google Gemini: pip install langvio[google]\n"
            "- For all providers: pip install langvio[all-llm]"
        )
        logger.error(error_msg)
        # print(error_msg, file=sys.stderr)
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
