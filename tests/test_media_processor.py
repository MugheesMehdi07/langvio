"""
Unit tests for media processor
"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from langvio.media.processor import MediaProcessor


class TestMediaProcessor(unittest.TestCase):
    """Test cases for MediaProcessor"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            "output_dir": "./test_output",
            "temp_dir": "./test_temp",
            "visualization": {
                "box_color": [0, 255, 0],
                "text_color": [255, 255, 255],
                "line_thickness": 2
            }
        }
        self.processor = MediaProcessor(self.config)

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        if os.path.exists("./test_output"):
            shutil.rmtree("./test_output")
        if os.path.exists("./test_temp"):
            shutil.rmtree("./test_temp")

    def test_initialization(self):
        """Test MediaProcessor initialization"""
        self.assertEqual(self.processor.config, self.config)
        self.assertIsNotNone(self.processor.image_visualizer)
        self.assertIsNotNone(self.processor.video_visualizer)

    def test_is_video_true(self):
        """Test is_video returns True for video files"""
        self.assertTrue(self.processor.is_video("test.mp4"))
        self.assertTrue(self.processor.is_video("test.avi"))
        self.assertTrue(self.processor.is_video("test.mov"))

    def test_is_video_false(self):
        """Test is_video returns False for non-video files"""
        self.assertFalse(self.processor.is_video("test.jpg"))
        self.assertFalse(self.processor.is_video("test.png"))
        self.assertFalse(self.processor.is_video("test.txt"))

    def test_get_output_path(self):
        """Test generating output path"""
        input_path = "/path/to/video.mp4"
        output_path = self.processor.get_output_path(input_path)
        
        self.assertIn("video_processed.mp4", output_path)
        self.assertIn("test_output", output_path)

    def test_get_output_path_custom_suffix(self):
        """Test generating output path with custom suffix"""
        input_path = "/path/to/image.jpg"
        output_path = self.processor.get_output_path(input_path, suffix="_annotated")
        
        self.assertIn("image_annotated.jpg", output_path)

    def test_update_config(self):
        """Test updating configuration"""
        new_config = {"output_dir": "./new_output"}
        self.processor.update_config(new_config)
        
        self.assertEqual(self.processor.config["output_dir"], "./new_output")

    def test_visualize_image_with_highlights(self):
        """Test delegating image visualization"""
        with patch.object(self.processor.image_visualizer, 'visualize_with_highlights') as mock_viz:
            self.processor.visualize_image_with_highlights(
                "input.jpg", "output.jpg", [], []
            )
            mock_viz.assert_called_once()

    def test_visualize_video_with_highlights(self):
        """Test delegating video visualization"""
        with patch.object(self.processor.video_visualizer, 'visualize_with_highlights') as mock_viz:
            self.processor.visualize_video_with_highlights(
                "input.mp4", "output.mp4", {}, []
            )
            mock_viz.assert_called_once()


if __name__ == "__main__":
    unittest.main()

