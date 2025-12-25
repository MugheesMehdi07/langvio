"""
Unit tests for model registry
"""

import unittest
from unittest.mock import MagicMock

from langvio.core.registry import ModelRegistry


class TestModelRegistry(unittest.TestCase):
    """Test cases for ModelRegistry class"""

    def setUp(self):
        """Set up test fixtures"""
        self.registry = ModelRegistry()

    def test_register_llm_processor(self):
        """Test registering LLM processor"""
        mock_processor = MagicMock()
        self.registry.register_llm_processor("test_llm", mock_processor, param1="value1")
        
        processors = self.registry.list_llm_processors()
        self.assertIn("test_llm", processors)

    def test_register_vision_processor(self):
        """Test registering vision processor"""
        mock_processor = MagicMock()
        self.registry.register_vision_processor("test_vision", mock_processor, param1="value1")
        
        processors = self.registry.list_vision_processors()
        self.assertIn("test_vision", processors)

    def test_get_llm_processor(self):
        """Test getting LLM processor instance"""
        mock_processor_class = MagicMock()
        mock_instance = MagicMock()
        mock_processor_class.return_value = mock_instance
        
        self.registry.register_llm_processor("test_llm", mock_processor_class, param1="value1")
        
        instance = self.registry.get_llm_processor("test_llm", param2="value2")
        self.assertEqual(instance, mock_instance)
        mock_processor_class.assert_called_once()

    def test_get_vision_processor(self):
        """Test getting vision processor instance"""
        mock_processor_class = MagicMock()
        mock_instance = MagicMock()
        mock_processor_class.return_value = mock_instance
        
        self.registry.register_vision_processor("test_vision", mock_processor_class, param1="value1")
        
        instance = self.registry.get_vision_processor("test_vision", param2="value2")
        self.assertEqual(instance, mock_instance)
        mock_processor_class.assert_called_once()

    def test_get_nonexistent_llm_processor(self):
        """Test getting non-existent LLM processor raises ValueError"""
        with self.assertRaises(ValueError):
            self.registry.get_llm_processor("nonexistent")

    def test_get_nonexistent_vision_processor(self):
        """Test getting non-existent vision processor raises ValueError"""
        with self.assertRaises(ValueError):
            self.registry.get_vision_processor("nonexistent")

    def test_list_llm_processors(self):
        """Test listing LLM processors"""
        mock_processor1 = MagicMock()
        mock_processor2 = MagicMock()
        
        self.registry.register_llm_processor("llm1", mock_processor1)
        self.registry.register_llm_processor("llm2", mock_processor2)
        
        processors = self.registry.list_llm_processors()
        self.assertEqual(len(processors), 2)
        self.assertIn("llm1", processors)
        self.assertIn("llm2", processors)

    def test_list_vision_processors(self):
        """Test listing vision processors"""
        mock_processor1 = MagicMock()
        mock_processor2 = MagicMock()
        
        self.registry.register_vision_processor("vision1", mock_processor1)
        self.registry.register_vision_processor("vision2", mock_processor2)
        
        processors = self.registry.list_vision_processors()
        self.assertEqual(len(processors), 2)
        self.assertIn("vision1", processors)
        self.assertIn("vision2", processors)


if __name__ == "__main__":
    unittest.main()

