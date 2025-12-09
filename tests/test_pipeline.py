"""
Unit tests for pipeline
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from langvio.config import Config
from langvio.core.pipeline import Pipeline


class TestPipeline(unittest.TestCase):
    """Test cases for Pipeline class"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = Config()
        self.pipeline = Pipeline()

    def test_initialization(self):
        """Test pipeline initialization"""
        self.assertIsNotNone(self.pipeline.config)
        self.assertIsNotNone(self.pipeline.processor_manager)
        self.assertIsNotNone(self.pipeline.visualization_manager)

    def test_load_config(self):
        """Test loading configuration from file"""
        config_content = """
llm:
  default: "gemini"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = f.name

        try:
            self.pipeline.load_config(temp_path)
            self.assertEqual(self.pipeline.config.config["llm"]["default"], "gemini")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @patch('langvio.core.processor_manager.registry')
    def test_set_llm_processor(self, mock_registry):
        """Test setting LLM processor"""
        mock_registry.list_llm_processors.return_value = {"test_llm": MagicMock()}
        mock_processor = MagicMock()
        mock_registry.get_llm_processor.return_value = mock_processor
        
        self.pipeline.set_llm_processor("test_llm")
        self.assertEqual(self.pipeline.processor_manager.llm_processor, mock_processor)

    @patch('langvio.core.processor_manager.registry')
    def test_set_vision_processor(self, mock_registry):
        """Test setting vision processor"""
        mock_registry.list_vision_processors.return_value = {"test_vision": MagicMock()}
        mock_processor = MagicMock()
        mock_registry.get_vision_processor.return_value = mock_processor
        
        self.pipeline.set_vision_processor("test_vision")
        self.assertEqual(self.pipeline.processor_manager.vision_processor, mock_processor)

    def test_process_file_not_found(self):
        """Test process raises FileNotFoundError for non-existent file"""
        with self.assertRaises(FileNotFoundError):
            self.pipeline.process("test query", "nonexistent.jpg")

    @patch('langvio.core.processor_manager.registry')
    def test_process_without_processors(self, mock_registry):
        """Test process raises ValueError when processors not set"""
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jpg', delete=False) as f:
            temp_path = f.name

        try:
            with self.assertRaises(ValueError):
                self.pipeline.process("test query", temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @patch('langvio.core.processor_manager.registry')
    @patch('langvio.utils.file_utils.is_video_file')
    def test_process_image(self, mock_is_video, mock_registry):
        """Test processing an image"""
        mock_is_video.return_value = False
        
        # Set up mocks
        mock_llm = MagicMock()
        mock_llm.parse_query.return_value = {"task_type": "identification"}
        mock_llm.generate_explanation.return_value = "Test explanation"
        mock_llm.get_highlighted_objects.return_value = []
        
        mock_vision = MagicMock()
        mock_vision.process_image.return_value = {"objects": []}
        
        mock_registry.list_llm_processors.return_value = {"test_llm": MagicMock()}
        mock_registry.list_vision_processors.return_value = {"test_vision": MagicMock()}
        mock_registry.get_llm_processor.return_value = mock_llm
        mock_registry.get_vision_processor.return_value = mock_vision
        
        self.pipeline.set_llm_processor("test_llm")
        self.pipeline.set_vision_processor("test_vision")
        
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jpg', delete=False) as f:
            temp_path = f.name

        try:
            result = self.pipeline.process("test query", temp_path)
            self.assertIn("query", result)
            self.assertIn("explanation", result)
            self.assertIn("output_path", result)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == "__main__":
    unittest.main()

