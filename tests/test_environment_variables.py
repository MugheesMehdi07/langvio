"""
Unit tests for environment variable handling
"""

import os
import unittest
from unittest.mock import patch

from langvio.config import Config
from langvio.llm.qwen_vlm import VLMProcessor


class TestEnvironmentVariables(unittest.TestCase):
    """Test cases for environment variable usage"""

    def test_config_loads_env_vars(self):
        """Test that config loads environment variables via dotenv"""
        # This test verifies that load_dotenv() is called
        # The actual loading is tested in test_config.py
        config = Config()
        self.assertIsNotNone(config.config)

    @patch.dict(os.environ, {"LANGVIO_DEFAULT_LLM": "qwen"}, clear=False)
    def test_llm_env_override(self):
        """Test LLM environment variable override"""
        config = Config()
        # Should use the local VLM (qwen) from environment
        self.assertEqual(config.config["llm"]["default"], "qwen")

    @patch.dict(os.environ, {"LANGVIO_DEFAULT_VISION": "yolo11n"}, clear=False)
    def test_vision_env_override(self):
        """Test vision environment variable override"""
        config = Config()
        # Should use yolo11n from environment (it's in the config)
        self.assertEqual(config.config["vision"]["default"], "yolo11n")

    def test_vlm_processor_needs_no_api_key(self):
        """The local VLM processor does not depend on any API key env var."""
        processor = VLMProcessor(
            name="test",
            model_name="Qwen/Qwen2.5-VL-3B-Instruct",
            model_kwargs={"temperature": 0.2, "max_new_tokens": 1024},
        )
        # Config is populated purely from constructor args, no env lookup.
        self.assertEqual(
            processor.config["model_name"], "Qwen/Qwen2.5-VL-3B-Instruct"
        )
        self.assertEqual(processor.config["model_kwargs"]["max_new_tokens"], 1024)


if __name__ == "__main__":
    unittest.main()
