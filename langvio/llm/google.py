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
                    "Gemini models uses 'langchain-google-genai' package. "
                    "Please install it with 'pip install langvio[google]'"
                )

            # Import necessary components
            import os

            # Import tenacity first to avoid circular import issues
            # langchain_google_genai depends on tenacity, so we need to ensure
            # tenacity is fully initialized before importing langchain_google_genai
            try:
                import tenacity
                # Access an attribute to ensure the module is fully loaded
                _ = tenacity.stop
            except (ImportError, AttributeError):
                # tenacity may not be available, but that's okay
                # langchain_google_genai will handle it
                pass

            # Now import langchain_google_genai
            from langchain_google_genai import ChatGoogleGenerativeAI

            # Get model configuration
            model_name = self.config["model_name"]
            model_kwargs = self.config["model_kwargs"].copy()

            # Check for API key in environment (supports both GOOGLE_API_KEY and GEMINI_API_KEY)
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            
            if not api_key:
                # Log a warning rather than setting it from config
                self.logger.warning(
                    "GOOGLE_API_KEY or GEMINI_API_KEY environment variable not found. "
                    "Please set it using 'export GOOGLE_API_KEY=your_key' "
                    "or add it to your .env file"
                )
                raise ValueError(
                    "GOOGLE_API_KEY or GEMINI_API_KEY environment variable is required. "
                    "Please set it using 'export GOOGLE_API_KEY=your_key' "
                    "or add it to your .env file"
                )
            
            # Log which API key variable was used (without exposing the key)
            if os.getenv("GOOGLE_API_KEY"):
                self.logger.debug("Using GOOGLE_API_KEY from environment")
            elif os.getenv("GEMINI_API_KEY"):
                self.logger.debug("Using GEMINI_API_KEY from environment")

            # Create the Gemini LLM
            # Try the model name as-is first, but also handle common variations
            try:
                self.llm = ChatGoogleGenerativeAI(model=model_name, **model_kwargs)
                self.logger.info(f"Initialized Google Gemini model: {model_name}")
            except Exception as model_error:
                # If model not found, try alternative names
                error_str = str(model_error).lower()
                if "not found" in error_str or "404" in error_str:
                    # Try common alternative model names
                    alternatives = {
                        "gemini-2.0-flash": ["gemini-2.0-flash-exp", "gemini-2.0-flash-thinking-exp"],
                        "gemini-2.0-flash-exp": ["gemini-2.0-flash", "gemini-1.5-pro"],
                        "gemini-1.5-flash": ["gemini-1.5-pro", "gemini-pro"],
                    }
                    
                    for alt_model in alternatives.get(model_name, []):
                        try:
                            self.logger.info(f"Trying alternative model: {alt_model}")
                            self.llm = ChatGoogleGenerativeAI(model=alt_model, **model_kwargs)
                            self.logger.info(f"Successfully initialized with alternative model: {alt_model}")
                            break
                        except Exception:
                            continue
                    else:
                        # If all alternatives failed, raise the original error
                        raise model_error
                else:
                    raise
        except Exception as e:
            error_msg = str(e)
            # Provide helpful error messages for common issues
            if "quota" in error_msg.lower() or "429" in error_msg:
                self.logger.error(
                    f"Error initializing Google Gemini model: {e}\n"
                    f"NOTE: If you're seeing quota errors with 'gemini-2.0-flash', "
                    f"this model requires Google Cloud billing enabled. "
                    f"Try using 'gemini-1.5-flash' or 'gemini-1.5-pro' for free tier access."
                )
            else:
                self.logger.error(f"Error initializing Google Gemini model: {e}")
            raise
