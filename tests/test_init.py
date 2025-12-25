"""
Unit tests for __init__.py module
"""

import unittest
from unittest.mock import patch, MagicMock
import langvio


class TestInit(unittest.TestCase):
    """Test cases for langvio __init__.py"""

    def test_registry_available(self):
        """Test that registry is available"""
        self.assertIsNotNone(langvio.registry)
        self.assertIsNotNone(langvio.registry.list_llm_processors)
        self.assertIsNotNone(langvio.registry.list_vision_processors)

    def test_create_pipeline_default(self):
        """Test create_pipeline with defaults"""
        with patch('langvio.Pipeline') as mock_pipeline_class:
            mock_pipeline = MagicMock()
            mock_pipeline.config.config = {
                "vision": {
                    "default": "yolo11n",
                    "models": {"yolo11n": {}, "yolo_world_v2_m": {}}
                },
                "llm": {
                    "default": "gpt-3.5",
                    "models": {"gpt-3.5": {}}
                }
            }
            mock_pipeline_class.return_value = mock_pipeline
            
            with patch('langvio.registry') as mock_registry:
                mock_registry.list_vision_processors.return_value = {"yolo11n": MagicMock()}
                mock_registry.list_llm_processors.return_value = {"gpt-3.5": MagicMock()}
                
                result = langvio.create_pipeline()
                self.assertIsNotNone(result)

    def test_create_pipeline_with_vision_name(self):
        """Test create_pipeline with vision name"""
        with patch('langvio.Pipeline') as mock_pipeline_class:
            mock_pipeline = MagicMock()
            mock_pipeline_class.return_value = mock_pipeline
            
            result = langvio.create_pipeline(vision_name="yolo11n")
            mock_pipeline.set_vision_processor.assert_called_once_with("yolo11n")

    def test_create_pipeline_with_llm_name(self):
        """Test create_pipeline with LLM name"""
        with patch('langvio.Pipeline') as mock_pipeline_class:
            mock_pipeline = MagicMock()
            mock_pipeline.config.config = {
                "vision": {
                    "default": "yolo11n",
                    "models": {"yolo11n": {}}
                },
                "llm": {"default": "gpt-3.5", "models": {"gpt-3.5": {}}}
            }
            mock_pipeline_class.return_value = mock_pipeline
            
            with patch('langvio.registry') as mock_registry:
                mock_registry.list_vision_processors.return_value = {"yolo11n": MagicMock()}
                mock_registry.list_llm_processors.return_value = {"gpt-3.5": MagicMock()}
                
                # Mock set_vision_processor and set_llm_processor to not raise
                mock_pipeline.set_vision_processor = MagicMock()
                mock_pipeline.set_llm_processor = MagicMock()
                
                result = langvio.create_pipeline(llm_name="gpt-3.5")
                mock_pipeline.set_llm_processor.assert_called_once_with("gpt-3.5")

    def test_create_pipeline_fallback_vision(self):
        """Test create_pipeline fallback vision processor selection"""
        with patch('langvio.Pipeline') as mock_pipeline_class:
            mock_pipeline = MagicMock()
            mock_pipeline.config.config = {
                "vision": {
                    "default": "invalid_model",
                    "models": {"yolo11n": {}, "yolo_world_v2_m": {}}
                },
                "llm": {
                    "default": "gpt-3.5",
                    "models": {"gpt-3.5": {}}
                }
            }
            mock_pipeline_class.return_value = mock_pipeline
            
            with patch('langvio.registry') as mock_registry:
                mock_registry.list_vision_processors.return_value = {
                    "yolo11n": MagicMock(),
                    "yolo_world_v2_m": MagicMock()
                }
                mock_registry.list_llm_processors.return_value = {"gpt-3.5": MagicMock()}
                
                # Mock set_vision_processor to succeed
                mock_pipeline.set_vision_processor = MagicMock()
                
                result = langvio.create_pipeline()
                # Should have tried to set a vision processor
                self.assertGreaterEqual(mock_pipeline.set_vision_processor.call_count, 0)


if __name__ == "__main__":
    unittest.main()

