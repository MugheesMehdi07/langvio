"""
Unit tests for visualization manager
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch, Mock
import cv2
import numpy as np

from langvio.config import Config
from langvio.core.visualization_manager import VisualizationManager

# Import cv2 constants for mocking
try:
    CAP_PROP_FRAME_WIDTH = cv2.CAP_PROP_FRAME_WIDTH
    CAP_PROP_FRAME_HEIGHT = cv2.CAP_PROP_FRAME_HEIGHT
    CAP_PROP_FPS = cv2.CAP_PROP_FPS
except AttributeError:
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5


class TestVisualizationManager(unittest.TestCase):
    """Test cases for VisualizationManager"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = Config()
        self.manager = VisualizationManager(self.config)
        self.test_image = None
        self.test_video = None

    def tearDown(self):
        """Clean up test fixtures"""
        if self.test_image and os.path.exists(self.test_image):
            os.unlink(self.test_image)
        if self.test_video and os.path.exists(self.test_video):
            os.unlink(self.test_video)

    def test_initialization(self):
        """Test VisualizationManager initialization"""
        self.assertIsNotNone(self.manager.config)
        self.assertIsNotNone(self.manager.media_processor)
        self.assertIsNotNone(self.manager.yolo_world_visualizer)

    def test_get_visualization_config_default(self):
        """Test getting default visualization config"""
        query_params = {"task_type": "identification"}
        config = self.manager._get_visualization_config(query_params)
        
        self.assertIn("box_color", config)
        self.assertIn("text_color", config)
        self.assertIn("line_thickness", config)

    def test_get_visualization_config_counting(self):
        """Test visualization config for counting task"""
        query_params = {"task_type": "counting"}
        config = self.manager._get_visualization_config(query_params)
        
        self.assertEqual(config["box_color"], [255, 0, 0])  # Red

    def test_get_visualization_config_verification(self):
        """Test visualization config for verification task"""
        query_params = {"task_type": "verification"}
        config = self.manager._get_visualization_config(query_params)
        
        self.assertEqual(config["box_color"], [0, 0, 255])  # Blue

    def test_get_visualization_config_tracking(self):
        """Test visualization config for tracking task"""
        query_params = {"task_type": "tracking"}
        config = self.manager._get_visualization_config(query_params)
        
        self.assertEqual(config["box_color"], [255, 165, 0])  # Orange
        self.assertEqual(config["line_thickness"], 3)

    def test_get_visualization_config_with_attributes(self):
        """Test visualization config with attributes"""
        query_params = {
            "task_type": "identification",
            "attributes": ["color", "size"]
        }
        config = self.manager._get_visualization_config(query_params)
        
        # Line thickness should be increased
        self.assertGreater(config["line_thickness"], 2)

    @patch('langvio.core.visualization_manager.is_video_file', return_value=False)
    def test_create_image_visualization(self, mock_is_video):
        """Test creating image visualization"""
        # Create a test image
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jpg', delete=False) as f:
            self.test_image = f.name
        
        cv2.imwrite(self.test_image, test_img)
        
        detections = {"objects": [{"bbox": [10, 10, 50, 50], "label": "person"}]}
        highlighted = []
        query_params = {"task_type": "identification"}
        
        with patch.object(self.manager, '_create_image_visualization') as mock_create:
            mock_create.return_value = "output.jpg"
            with patch.object(self.manager.media_processor, 'get_output_path', return_value="output.jpg"):
                result = self.manager.create_visualization(
                    self.test_image, detections, highlighted, query_params
                )
                # Result should be the output path (may have full path)
                self.assertIsNotNone(result)
                mock_create.assert_called_once()

    @patch('langvio.core.visualization_manager.is_video_file', return_value=True)
    def test_create_video_visualization_with_tracker(self, mock_is_video):
        """Test creating video visualization with tracker data"""
        detections = {
            "tracker_file_path": "tracker.json",
            "frame_detections": {}
        }
        highlighted = []
        query_params = {"task_type": "tracking"}
        
        with patch.object(self.manager, '_create_yolo_world_video_visualization') as mock_create:
            mock_create.return_value = "output.mp4"
            with patch.object(self.manager.media_processor, 'get_output_path', return_value="output.mp4"):
                result = self.manager.create_visualization(
                    "test.mp4", detections, highlighted, query_params
                )
                mock_create.assert_called_once()

    @patch('langvio.core.visualization_manager.is_video_file', return_value=True)
    def test_create_video_visualization_without_tracker(self, mock_is_video):
        """Test creating video visualization without tracker data"""
        detections = {"frame_detections": {"0": []}}
        highlighted = []
        query_params = {"task_type": "identification"}
        
        with patch.object(self.manager, '_create_video_visualization') as mock_create:
            mock_create.return_value = "output.mp4"
            with patch.object(self.manager.media_processor, 'get_output_path', return_value="output.mp4"):
                result = self.manager.create_visualization(
                    "test.mp4", detections, highlighted, query_params
                )
                mock_create.assert_called_once()

    def test_create_image_visualization_implementation(self):
        """Test _create_image_visualization implementation"""
        # Create a test image
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jpg', delete=False) as f:
            self.test_image = f.name
        
        cv2.imwrite(self.test_image, test_img)
        
        detections = {"objects": [{"bbox": [10, 10, 50, 50], "label": "person"}]}
        highlighted = [{"detection": {"bbox": [10, 10, 50, 50], "label": "person"}}]  # Match expected format
        config = {"box_color": [0, 255, 0], "line_thickness": 2, "text_color": [255, 255, 255], "show_attributes": True, "show_confidence": True}
        
        # Mock the media processor method to avoid actual file I/O
        with patch.object(self.manager.media_processor, 'visualize_image_with_highlights') as mock_viz:
            self.manager._create_image_visualization(
                self.test_image, "output.jpg", detections, highlighted, config
            )
            # Should have called visualize_image_with_highlights on media_processor
            mock_viz.assert_called_once()
            # Verify it was called with correct arguments
            call_args = mock_viz.call_args
            self.assertEqual(call_args[0][0], self.test_image)  # image_path
            self.assertEqual(call_args[0][1], "output.jpg")  # output_path

    @patch('langvio.core.visualization_manager.cv2.VideoWriter')
    @patch('langvio.core.visualization_manager.cv2.VideoCapture')
    @patch('os.path.exists')
    def test_create_video_visualization_implementation(self, mock_exists, mock_video_capture, mock_video_writer):
        """Test _create_video_visualization implementation"""
        mock_exists.return_value = True
        
        # Mock VideoCapture - CRITICAL: isOpened() must return False after first frame to break loop
        mock_cap = MagicMock()
        mock_cap.isOpened.side_effect = [True, False]  # First True, then False to break loop
        mock_cap.read.side_effect = [(True, np.zeros((100, 100, 3), dtype=np.uint8)), (False, None)]  # First frame, then break
        mock_cap.get.side_effect = lambda prop: {CAP_PROP_FRAME_WIDTH: 100, CAP_PROP_FRAME_HEIGHT: 100, CAP_PROP_FPS: 30.0}.get(prop, 0)
        mock_cap.set.return_value = True
        mock_cap.release = MagicMock()
        mock_video_capture.return_value = mock_cap
        
        # Mock VideoWriter
        mock_writer = MagicMock()
        mock_writer.write = MagicMock()
        mock_writer.release = MagicMock()
        mock_video_writer.return_value = mock_writer
        
        detections = {"frame_detections": {"0": [{"bbox": [10, 10, 50, 50]}]}, "summary": {}}
        highlighted = []
        config = {"box_color": [0, 255, 0], "text_color": [255, 255, 255], "line_thickness": 2}
        
        # Create a dummy video file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mp4', delete=False) as f:
            video_file = f.name
        
        try:
            self.manager._create_video_visualization(
                video_file, "output.mp4", detections, highlighted, config
            )
            # Should have attempted to process video
            self.assertTrue(mock_video_capture.called)
            mock_cap.release.assert_called()
            mock_writer.release.assert_called()
        finally:
            if os.path.exists(video_file):
                os.unlink(video_file)

    @patch('langvio.media.yolo_world_video_visualizer.YOLOWorldVideoVisualizer')
    @patch('os.path.exists')
    def test_create_yolo_world_video_visualization(self, mock_exists, mock_visualizer_class):
        """Test _create_yolo_world_video_visualization"""
        mock_exists.return_value = True
        
        mock_visualizer = MagicMock()
        mock_visualizer.visualize_from_tracker_file = MagicMock()
        # Replace the actual visualizer instance
        self.manager.yolo_world_visualizer = mock_visualizer
        
        # Create a dummy tracker file
        import json
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            tracker_file = f.name
            json.dump({}, f)
        
        # Create a dummy video file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mp4', delete=False) as f:
            video_file = f.name
        
        try:
            detections = {
                "tracker_file_path": tracker_file,
                "frame_detections": {}
            }
            highlighted = []
            config = {"box_color": [0, 255, 0], "text_color": [255, 255, 255], "line_thickness": 2, "show_attributes": True, "show_confidence": True}
            
            self.manager._create_yolo_world_video_visualization(
                video_file, "output.mp4", detections, highlighted, config
            )
            # Should have called visualize_from_tracker_file
            mock_visualizer.visualize_from_tracker_file.assert_called_once()
        finally:
            if os.path.exists(tracker_file):
                os.unlink(tracker_file)
            if os.path.exists(video_file):
                os.unlink(video_file)


if __name__ == "__main__":
    unittest.main()

