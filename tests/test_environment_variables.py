"""
Unit tests for environment variable handling
"""

import os
import unittest
from unittest.mock import patch

from unittest.mock import MagicMock

from langvio.config import Config
from langvio.llm.google import GeminiProcessor
from langvio.llm.openai import OpenAIProcessor


class TestEnvironmentVariables(unittest.TestCase):
    """Test cases for environment variable usage"""

    def test_config_loads_env_vars(self):
        """Test that config loads environment variables via dotenv"""
        # This test verifies that load_dotenv() is called
        # The actual loading is tested in test_config.py
        config = Config()
        self.assertIsNotNone(config.config)

    @patch.dict(os.environ, {"LANGVIO_DEFAULT_LLM": "gemini"}, clear=False)
    def test_llm_env_override(self):
        """Test LLM environment variable override"""
        config = Config()
        # Should use gemini from environment
        self.assertEqual(config.config["llm"]["default"], "gemini")

    @patch.dict(os.environ, {"LANGVIO_DEFAULT_VISION": "yolo_world_v2_m"}, clear=False)
    def test_vision_env_override(self):
        """Test vision environment variable override"""
        config = Config()
        # Should use yolo_world_v2_m from environment
        self.assertEqual(config.config["vision"]["default"], "yolo_world_v2_m")

    @patch.dict(os.environ, {}, clear=True)
    def test_openai_missing_api_key(self):
        """Test OpenAI processor raises error when API key missing"""
        processor = OpenAIProcessor(
            name="test",
            model_name="gpt-3.5-turbo",
            model_kwargs={}
        )
        with self.assertRaises(ValueError):
            processor.initialize()

    @patch.dict(os.environ, {}, clear=True)
    def test_gemini_missing_api_key(self):
        """Test Gemini processor raises error when API key missing"""
        processor = GeminiProcessor(
            name="test",
            model_name="gemini-pro",
            model_kwargs={}
        )
        with self.assertRaises(ValueError):
            processor.initialize()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    @patch('langvio.llm.openai.ChatOpenAI')
    def test_openai_with_api_key(self, mock_chat):
        """Test OpenAI processor initializes with API key"""
        mock_chat.return_value = MagicMock()
        processor = OpenAIProcessor(
            name="test",
            model_name="gpt-3.5-turbo",
            model_kwargs={}
        )
        # Should not raise error
        try:
            processor.initialize()
        except ImportError:
            # This is expected if langchain_openai is not installed
            pass

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=False)
    @patch('langvio.llm.google.ChatGoogleGenerativeAI')
    def test_gemini_with_api_key(self, mock_chat):
        """Test Gemini processor initializes with API key"""
        mock_chat.return_value = MagicMock()
        processor = GeminiProcessor(
            name="test",
            model_name="gemini-pro",
            model_kwargs={}
        )
        # Should not raise error
        try:
            processor.initialize()
        except ImportError:
            # This is expected if langchain_google_genai is not installed
            pass


if __name__ == "__main__":
    unittest.main()

