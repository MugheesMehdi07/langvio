"""
OpenAI-specific LLM processor implementation
"""

import logging
from typing import Any, Dict, Optional

from langvio.llm.base import BaseLLMProcessor


class OpenAIProcessor(BaseLLMProcessor):
    """LLM processor using OpenAI models via LangChain"""

    def __init__(
        self,
        name: str = "openai",
        model_name: str = "gpt-3.5-turbo",
        model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize OpenAI processor.

        Args:
            name: Processor name
            model_name: Name of the OpenAI model to use
            model_kwargs: Additional model parameters (temperature, etc.)
            **kwargs: Additional processor parameters
        """
        config = {
            "model_name": model_name,
            "model_kwargs": model_kwargs or {},
            **kwargs,
        }
        super().__init__(name, config)
        self.logger = logging.getLogger(__name__)

    def _initialize_llm(self) -> None:
        """
        Initialize the OpenAI model via LangChain.
        This is the only method that needs to be implemented.
        """
        try:
            if not self.is_package_installed("langchain_openai"):
                raise ImportError(
                    "The 'langchain-openai' package is required to use OpenAI models. "
                    "Please install it with 'pip install langvio[openai]'"
                )

            # Import necessary components
            import os

            from langchain_openai import ChatOpenAI

            # Get model configuration
            model_name = self.config["model_name"]
            model_kwargs = self.config["model_kwargs"].copy()

            if "OPENAI_API_KEY" not in os.environ:
                # Log a warning rather than setting it from config
                self.logger.warning(
                    "OPENAI_API_KEY environment variable not found. "
                    "Please set it using 'export OPENAI_API_KEY=your_key' "
                    "or add it to your .env file"
                )
                raise ValueError(
                    "OPENAI_API_KEY environment variable is required. "
                    "Please set it using 'export OPENAI_API_KEY=your_key' "
                    "or add it to your .env file"
                )
            else:
                # Create the OpenAI LLM
                self.llm = ChatOpenAI(model_name=model_name, **model_kwargs)

            self.logger.info(f"Initialized OpenAI model: {model_name}")
        except Exception as e:
            self.logger.error(f"Error initializing OpenAI model: {e}")
            raise
