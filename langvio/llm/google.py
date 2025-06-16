"""
Google Gemini-specific LLM processor implementation
"""

import logging
from typing import Any, Dict, Optional

from langvio.llm.base import BaseLLMProcessor


class GeminiProcessor(BaseLLMProcessor):
    """LLM processor using Google Gemini models via LangChain"""

    def __init__(
            self,
            name: str = "gemini",
            model_name: str = "gemini-pro",
            model_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs,
    ):
        """
        Initialize Gemini processor.

        Args:
            name: Processor name
            model_name: Name of the Gemini model to use
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
        Initialize the Google Gemini model via LangChain.
        This is the only method that needs to be implemented.
        """
        try:

            if not self.is_package_installed("langchain_google_genai"):
                raise ImportError(
                    "The 'langchain-google-genai' package is required to use Gemini models. "
                    "Please install it with 'pip install langvio[google]'"
                )

            # Import necessary components
            import os

            from langchain_google_genai import ChatGoogleGenerativeAI

            # Get model configuration
            model_name = self.config["model_name"]
            model_kwargs = self.config["model_kwargs"].copy()

            if "GOOGLE_API_KEY" not in os.environ:
                # Log a warning rather than setting it from config
                self.logger.warning(
                    "GOOGLE_API_KEY environment variable not found. "
                    "Please set it using 'export GOOGLE_API_KEY=your_key' o"
                )
                raise

            # Create the Gemini LLM
            self.llm = ChatGoogleGenerativeAI(model=model_name, **model_kwargs)

            self.logger.info(f"Initialized Google Gemini model: {model_name}")
        except Exception as e:
            self.logger.error(f"Error initializing Google Gemini model: {e}")
            raise
