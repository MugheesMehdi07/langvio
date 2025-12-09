"""
Unit tests for configuration management
"""

import os
import tempfile
import unittest
from unittest.mock import patch

from langvio.config import Config


class TestConfig(unittest.TestCase):
    """Test cases for Config class"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = Config()

    def test_default_config_loaded(self):
        """Test that default config is loaded"""
        self.assertIn("llm", self.config.config)
        self.assertIn("vision", self.config.config)
        self.assertIn("media", self.config.config)
        self.assertIn("logging", self.config.config)

    def test_get_llm_config(self):
        """Test getting LLM configuration"""
        llm_config = self.config.get_llm_config()
        self.assertIsInstance(llm_config, dict)
        self.assertIn("model_name", llm_config)

    def test_get_vision_config(self):
        """Test getting vision configuration"""
        vision_config = self.config.get_vision_config()
        self.assertIsInstance(vision_config, dict)

    def test_get_media_config(self):
        """Test getting media configuration"""
        media_config = self.config.get_media_config()
        self.assertIsInstance(media_config, dict)
        self.assertIn("output_dir", media_config)

    def test_get_logging_config(self):
        """Test getting logging configuration"""
        logging_config = self.config.get_logging_config()
        self.assertIsInstance(logging_config, dict)
        self.assertIn("level", logging_config)

    def test_load_config_from_file(self):
        """Test loading configuration from YAML file"""
        # Create a temporary config file
        config_content = """
llm:
  default: "gemini"
  models:
    gemini:
      model_name: "gemini-pro"
      model_kwargs:
        temperature: 0.3
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = f.name

        try:
            config = Config(config_path=temp_path)
            llm_config = config.get_llm_config("gemini")
            self.assertEqual(llm_config["model_kwargs"]["temperature"], 0.3)
        finally:
            os.unlink(temp_path)

    @patch.dict(os.environ, {"LANGVIO_DEFAULT_LLM": "gemini"})
    def test_environment_override_llm(self):
        """Test that environment variable overrides default LLM"""
        config = Config()
        self.assertEqual(config.config["llm"]["default"], "gemini")

    @patch.dict(os.environ, {"LANGVIO_DEFAULT_VISION": "yolo_world_v2_m"})
    def test_environment_override_vision(self):
        """Test that environment variable overrides default vision model"""
        config = Config()
        self.assertEqual(config.config["vision"]["default"], "yolo_world_v2_m")

    def test_invalid_llm_model(self):
        """Test that invalid LLM model raises ValueError"""
        with self.assertRaises(ValueError):
            self.config.get_llm_config("invalid_model")

    def test_invalid_vision_model(self):
        """Test that invalid vision model raises ValueError"""
        with self.assertRaises(ValueError):
            self.config.get_vision_config("invalid_model")

    def test_save_config(self):
        """Test saving configuration to file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name

        try:
            self.config.save_config(temp_path)
            self.assertTrue(os.path.exists(temp_path))
            
            # Load it back and verify
            loaded_config = Config(config_path=temp_path)
            self.assertEqual(loaded_config.config["llm"]["default"], 
                           self.config.config["llm"]["default"])
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == "__main__":
    unittest.main()

