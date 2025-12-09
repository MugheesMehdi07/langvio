"""
Unit tests for processor manager
"""

import unittest
from unittest.mock import MagicMock, patch

from langvio.config import Config
from langvio.core.processor_manager import ProcessorManager


class TestProcessorManager(unittest.TestCase):
    """Test cases for ProcessorManager class"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = Config()
        self.manager = ProcessorManager(self.config)

    def test_initialization(self):
        """Test processor manager initialization"""
        self.assertIsNone(self.manager.llm_processor)
        self.assertIsNone(self.manager.vision_processor)
        self.assertEqual(self.manager.config, self.config)

    def test_has_processors_false(self):
        """Test has_processors returns False when processors not set"""
        self.assertFalse(self.manager.has_processors())

    def test_has_processors_true(self):
        """Test has_processors returns True when processors are set"""
        self.manager.llm_processor = MagicMock()
        self.manager.vision_processor = MagicMock()
        self.assertTrue(self.manager.has_processors())

    @patch('langvio.core.processor_manager.registry')
    def test_set_llm_processor(self, mock_registry):
        """Test setting LLM processor"""
        mock_processor = MagicMock()
        mock_registry.list_llm_processors.return_value = {"test_llm": MagicMock()}
        mock_registry.get_llm_processor.return_value = mock_processor
        
        self.manager.set_llm_processor("test_llm")
        self.assertEqual(self.manager.llm_processor, mock_processor)
        mock_processor.initialize.assert_called_once()

    @patch('langvio.core.processor_manager.registry')
    def test_set_vision_processor(self, mock_registry):
        """Test setting vision processor"""
        mock_processor = MagicMock()
        mock_registry.list_vision_processors.return_value = {"test_vision": MagicMock()}
        mock_registry.get_vision_processor.return_value = mock_processor
        
        self.manager.set_vision_processor("test_vision")
        self.assertEqual(self.manager.vision_processor, mock_processor)

    def test_parse_query_without_processor(self):
        """Test parse_query raises ValueError when LLM processor not set"""
        with self.assertRaises(ValueError):
            self.manager.parse_query("test query")

    def test_parse_query_with_processor(self):
        """Test parse_query calls LLM processor"""
        mock_processor = MagicMock()
        mock_processor.parse_query.return_value = {"task_type": "identification"}
        self.manager.llm_processor = mock_processor
        
        result = self.manager.parse_query("test query")
        self.assertEqual(result["task_type"], "identification")
        mock_processor.parse_query.assert_called_once_with("test query")

    def test_process_media_without_processor(self):
        """Test process_media raises ValueError when vision processor not set"""
        with self.assertRaises(ValueError):
            self.manager.process_media("test.jpg", {})

    @patch('langvio.utils.file_utils.is_video_file')
    def test_process_media_image(self, mock_is_video):
        """Test process_media for image"""
        mock_is_video.return_value = False
        mock_processor = MagicMock()
        mock_processor.process_image.return_value = {"objects": []}
        self.manager.vision_processor = mock_processor
        
        result = self.manager.process_media("test.jpg", {})
        mock_processor.process_image.assert_called_once()
        self.assertIn("objects", result)

    @patch('langvio.utils.file_utils.is_video_file')
    def test_process_media_video(self, mock_is_video):
        """Test process_media for video"""
        mock_is_video.return_value = True
        mock_processor = MagicMock()
        mock_processor.process_video.return_value = {"frame_detections": {}}
        self.manager.vision_processor = mock_processor
        
        result = self.manager.process_media("test.mp4", {})
        mock_processor.process_video.assert_called_once()
        self.assertIn("frame_detections", result)

    def test_generate_explanation_without_processor(self):
        """Test generate_explanation raises ValueError when LLM processor not set"""
        with self.assertRaises(ValueError):
            self.manager.generate_explanation("test query", {})

    def test_generate_explanation_with_processor(self):
        """Test generate_explanation calls LLM processor"""
        mock_processor = MagicMock()
        mock_processor.generate_explanation.return_value = "Test explanation"
        self.manager.llm_processor = mock_processor
        
        result = self.manager.generate_explanation("test query", {})
        self.assertEqual(result, "Test explanation")
        mock_processor.generate_explanation.assert_called_once()

    def test_get_highlighted_objects_without_processor(self):
        """Test get_highlighted_objects returns empty list when LLM processor not set"""
        result = self.manager.get_highlighted_objects()
        self.assertEqual(result, [])

    def test_get_highlighted_objects_with_processor(self):
        """Test get_highlighted_objects calls LLM processor"""
        mock_processor = MagicMock()
        mock_processor.get_highlighted_objects.return_value = [{"id": 1}]
        self.manager.llm_processor = mock_processor
        
        result = self.manager.get_highlighted_objects()
        self.assertEqual(result, [{"id": 1}])
        mock_processor.get_highlighted_objects.assert_called_once()


if __name__ == "__main__":
    unittest.main()

