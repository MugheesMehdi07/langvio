"""
Unit tests for Google LLM processor
"""

import unittest
from unittest.mock import patch, MagicMock
from langvio.llm.google import GeminiProcessor


class TestGeminiProcessor(unittest.TestCase):
    """Test cases for GeminiProcessor"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            "model_name": "gemini-pro"
        }

    def test_initialization(self):
        """Test GeminiProcessor initialization"""
        processor = GeminiProcessor("test", **self.config)
        self.assertEqual(processor.name, "test")
        self.assertEqual(processor.config["model_name"], "gemini-pro")

    def test_initialization_with_model_kwargs(self):
        """Test GeminiProcessor initialization with model_kwargs"""
        config = {
            "model_name": "gemini-pro",
            "model_kwargs": {"temperature": 0.7}
        }
        processor = GeminiProcessor("test", **config)
        self.assertIn("model_kwargs", processor.config)



if __name__ == "__main__":
    unittest.main()

