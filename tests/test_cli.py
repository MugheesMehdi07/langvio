"""
Unit tests for CLI interface
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock, mock_open
import argparse

from langvio.cli import main, list_available_models


class TestCLI(unittest.TestCase):
    """Test cases for CLI interface"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_image = None
        self.test_video = None
        
    def tearDown(self):
        """Clean up test fixtures"""
        if self.test_image and os.path.exists(self.test_image):
            os.unlink(self.test_image)
        if self.test_video and os.path.exists(self.test_video):
            os.unlink(self.test_video)

    @patch('langvio.cli.sys.argv', ['langvio', '--list-models'])
    @patch('langvio.cli.list_available_models')
    @patch('langvio.cli.setup_logging')
    def test_list_models_flag(self, mock_logging, mock_list):
        """Test --list-models flag"""
        try:
            result = main()
            mock_list.assert_called_once()
            self.assertEqual(result, 0)
        except SystemExit:
            # argparse may call sys.exit, which is fine
            pass

    @patch('langvio.cli.sys.argv', ['langvio', '--list-models'])
    @patch('langvio.registry')
    def test_list_available_models(self, mock_registry):
        """Test list_available_models function"""
        mock_registry.list_llm_processors.return_value = {"gpt-3.5": MagicMock(), "gemini": MagicMock()}
        mock_registry.list_vision_processors.return_value = {"yolo11n": MagicMock()}
        
        # Capture stdout
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            list_available_models()
        output = f.getvalue()
        
        self.assertIn("Available LLM Processors", output)
        self.assertIn("Available Vision Processors", output)
        self.assertIn("gpt-3.5", output)
        self.assertIn("yolo11n", output)

    @patch('langvio.cli.sys.argv', ['langvio', '--query', 'test', '--media', 'nonexistent.jpg'])
    @patch('langvio.cli.setup_logging')
    def test_media_file_not_found(self, mock_logging):
        """Test error when media file doesn't exist"""
        with patch('langvio.cli.os.path.exists', return_value=False):
            result = main()
            self.assertEqual(result, 1)

    @patch('langvio.cli.sys.argv', ['langvio', '--query', 'test', '--media', 'test.txt'])
    @patch('langvio.cli.setup_logging')
    def test_unsupported_media_format(self, mock_logging):
        """Test error for unsupported media format"""
        with patch('langvio.cli.os.path.exists', return_value=True):
            with patch('langvio.cli.is_image_file', return_value=False):
                with patch('langvio.cli.is_video_file', return_value=False):
                    result = main()
                    self.assertEqual(result, 1)

    @patch('langvio.cli.sys.argv', ['langvio', '--query', 'Count people', '--media', 'test.jpg'])
    @patch('langvio.cli.setup_logging')
    @patch('langvio.cli.create_pipeline')
    @patch('langvio.cli.is_image_file', return_value=True)
    @patch('langvio.cli.os.path.exists', return_value=True)
    def test_successful_processing(self, mock_exists, mock_is_image, mock_create_pipeline, mock_logging):
        """Test successful processing"""
        # Create a temporary image file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jpg', delete=False) as f:
            self.test_image = f.name
        
        mock_pipeline = MagicMock()
        mock_pipeline.process.return_value = {
            "query": "Count people",
            "media_path": self.test_image,
            "media_type": "image",
            "output_path": "output.jpg",
            "explanation": "Found 3 people",
            "detections": {
                "frame_0": [
                    {"label": "person", "confidence": 0.9},
                    {"label": "person", "confidence": 0.8},
                ]
            }
        }
        mock_pipeline.config.config = {"media": {"output_dir": "./output"}}
        mock_create_pipeline.return_value = mock_pipeline
        
        with patch('langvio.cli.sys.argv', ['langvio', '--query', 'Count people', '--media', self.test_image]):
            with patch('langvio.cli.print') as mock_print:
                result = main()
                self.assertEqual(result, 0)
                mock_print.assert_called()

    @patch('langvio.cli.sys.argv', ['langvio', '--query', 'test', '--media', 'test.jpg', '--output', '/custom/output'])
    @patch('langvio.cli.setup_logging')
    @patch('langvio.cli.create_pipeline')
    @patch('langvio.cli.is_image_file', return_value=True)
    @patch('langvio.cli.os.path.exists', return_value=True)
    @patch('langvio.cli.os.makedirs')
    def test_custom_output_directory(self, mock_makedirs, mock_exists, mock_is_image, mock_create_pipeline, mock_logging):
        """Test custom output directory"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jpg', delete=False) as f:
            self.test_image = f.name
        
        mock_pipeline = MagicMock()
        mock_pipeline.process.return_value = {
            "query": "test",
            "media_path": self.test_image,
            "media_type": "image",
            "output_path": "output.jpg",
            "explanation": "Test",
            "detections": {}
        }
        mock_pipeline.config.config = {"media": {"output_dir": "./output"}}
        mock_create_pipeline.return_value = mock_pipeline
        
        with patch('langvio.cli.sys.argv', ['langvio', '--query', 'test', '--media', self.test_image, '--output', '/custom/output']):
            result = main()
            mock_makedirs.assert_called_once_with('/custom/output', exist_ok=True)
            self.assertEqual(mock_pipeline.config.config["media"]["output_dir"], '/custom/output')

    @patch('langvio.cli.sys.argv', ['langvio', '--query', 'test', '--media', 'test.jpg'])
    @patch('langvio.cli.setup_logging')
    @patch('langvio.cli.create_pipeline')
    @patch('langvio.cli.is_image_file', return_value=True)
    @patch('langvio.cli.os.path.exists', return_value=True)
    def test_processing_exception(self, mock_exists, mock_is_image, mock_create_pipeline, mock_logging):
        """Test exception handling during processing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jpg', delete=False) as f:
            self.test_image = f.name
        
        mock_pipeline = MagicMock()
        mock_pipeline.process.side_effect = Exception("Processing error")
        mock_create_pipeline.return_value = mock_pipeline
        
        with patch('langvio.cli.sys.argv', ['langvio', '--query', 'test', '--media', self.test_image]):
            result = main()
            self.assertEqual(result, 1)

    @patch('langvio.cli.sys.argv', ['langvio', '--query', 'test', '--media', 'test.jpg', '--config', 'config.yaml'])
    @patch('langvio.cli.setup_logging')
    @patch('langvio.cli.create_pipeline')
    @patch('langvio.cli.is_image_file', return_value=True)
    @patch('langvio.cli.os.path.exists', return_value=True)
    def test_config_file_argument(self, mock_exists, mock_is_image, mock_create_pipeline, mock_logging):
        """Test config file argument"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jpg', delete=False) as f:
            self.test_image = f.name
        
        mock_pipeline = MagicMock()
        mock_pipeline.process.return_value = {
            "query": "test",
            "media_path": self.test_image,
            "media_type": "image",
            "output_path": "output.jpg",
            "explanation": "Test",
            "detections": {}
        }
        mock_pipeline.config.config = {"media": {"output_dir": "./output"}}
        mock_create_pipeline.return_value = mock_pipeline
        
        with patch('langvio.cli.sys.argv', ['langvio', '--query', 'test', '--media', self.test_image, '--config', 'config.yaml']):
            result = main()
            mock_create_pipeline.assert_called_once_with(
                config_path='config.yaml',
                llm_name=None,
                vision_name=None
            )

    @patch('langvio.cli.sys.argv', ['langvio', '--query', 'test', '--media', 'test.jpg', '--llm', 'gpt-3.5', '--vision', 'yolo11n'])
    @patch('langvio.cli.setup_logging')
    @patch('langvio.cli.create_pipeline')
    @patch('langvio.cli.is_image_file', return_value=True)
    @patch('langvio.cli.os.path.exists', return_value=True)
    def test_llm_and_vision_arguments(self, mock_exists, mock_is_image, mock_create_pipeline, mock_logging):
        """Test LLM and vision processor arguments"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jpg', delete=False) as f:
            self.test_image = f.name
        
        mock_pipeline = MagicMock()
        mock_pipeline.process.return_value = {
            "query": "test",
            "media_path": self.test_image,
            "media_type": "image",
            "output_path": "output.jpg",
            "explanation": "Test",
            "detections": {}
        }
        mock_pipeline.config.config = {"media": {"output_dir": "./output"}}
        mock_create_pipeline.return_value = mock_pipeline
        
        with patch('langvio.cli.sys.argv', ['langvio', '--query', 'test', '--media', self.test_image, '--llm', 'gpt-3.5', '--vision', 'yolo11n']):
            result = main()
            mock_create_pipeline.assert_called_once_with(
                config_path=None,
                llm_name='gpt-3.5',
                vision_name='yolo11n'
            )


if __name__ == "__main__":
    unittest.main()

