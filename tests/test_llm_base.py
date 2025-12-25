"""
Unit tests for LLM base processor
"""

import unittest
from unittest.mock import MagicMock, patch

from langvio.llm.base import BaseLLMProcessor


class MockLLMProcessor(BaseLLMProcessor):
    """Mock LLM processor for testing"""
    
    def _initialize_llm(self):
        """Mock LLM initialization"""
        self.llm = MagicMock()


class TestBaseLLMProcessor(unittest.TestCase):
    """Test cases for BaseLLMProcessor class"""

    def setUp(self):
        """Set up test fixtures"""
        self.processor = MockLLMProcessor("test_processor", {
            "model_name": "test-model",
            "model_kwargs": {}
        })

    def test_initialization(self):
        """Test processor initialization"""
        self.assertEqual(self.processor.name, "test_processor")
        self.assertIsNotNone(self.processor.config)

    def test_initialize(self):
        """Test processor initialization"""
        result = self.processor.initialize()
        self.assertTrue(result)
        self.assertIsNotNone(self.processor.llm)

    def test_get_highlighted_objects_default(self):
        """Test getting highlighted objects when none set"""
        result = self.processor.get_highlighted_objects()
        self.assertEqual(result, [])

    def test_get_highlighted_objects(self):
        """Test getting highlighted objects"""
        self.processor._highlighted_objects = [{"id": 1}, {"id": 2}]
        result = self.processor.get_highlighted_objects()
        self.assertEqual(len(result), 2)

    def test_is_package_installed(self):
        """Test checking if package is installed"""
        # Should return True for standard library packages
        self.assertTrue(self.processor.is_package_installed("os"))
        # Should return False for non-existent packages
        self.assertFalse(self.processor.is_package_installed("nonexistent_package_xyz123"))


if __name__ == "__main__":
    unittest.main()

