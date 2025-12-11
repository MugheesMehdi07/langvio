"""
Unit tests for YOLO-World video visualizer
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import cv2
import tempfile
import os

from langvio.media.yolo_world_video_visualizer import YOLOWorldVideoVisualizer


class TestYOLOWorldVideoVisualizer(unittest.TestCase):
    """Test cases for YOLOWorldVideoVisualizer"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {}
        self.visualizer = YOLOWorldVideoVisualizer(self.config)
        self.test_video = None

    def tearDown(self):
        """Clean up test fixtures"""
        if self.test_video and os.path.exists(self.test_video):
            os.unlink(self.test_video)

    def test_initialization(self):
        """Test YOLOWorldVideoVisualizer initialization"""
        self.assertIsNotNone(self.visualizer)

    @patch('cv2.VideoCapture')
    @patch('cv2.VideoWriter')
    def test_visualize_with_tracker_data(self, mock_writer, mock_capture):
        """Test visualize_with_tracker_data"""
        # Create a dummy video file
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mp4', delete=False) as f:
            self.test_video = f.name
        
        # Mock video capture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [
            (True, test_img),
            (False, None)
        ]
        mock_cap.get.side_effect = lambda x: {
            cv2.CAP_PROP_FRAME_WIDTH: 100,
            cv2.CAP_PROP_FRAME_HEIGHT: 100,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 1
        }.get(x, 0)
        mock_capture.return_value = mock_cap
        
        # Mock video writer
        mock_writer_instance = MagicMock()
        mock_writer.return_value = mock_writer_instance
        
        tracker_data = {
            "detections": [
                {
                    "frame_id": 0,
                    "objects": [
                        {
                            "bbox": [10, 10, 50, 50],
                            "label": "person",
                            "confidence": 0.9,
                            "track_id": 1
                        }
                    ]
                }
            ],
            "tracks": [{"track_id": 1}],
            "metadata": {}
        }
        
        self.visualizer.visualize_with_tracker_data(
            self.test_video, "output.mp4", tracker_data
        )
        
        # Should complete without error
        mock_writer_instance.write.assert_called()

    @patch('cv2.VideoCapture')
    def test_visualize_with_tracker_data_no_video(self, mock_capture):
        """Test visualize_with_tracker_data with invalid video"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_capture.return_value = mock_cap
        
        tracker_data = {"detections": [], "tracks": [], "metadata": {}}
        
        with self.assertRaises(ValueError):
            self.visualizer.visualize_with_tracker_data(
                "nonexistent.mp4", "output.mp4", tracker_data
            )


if __name__ == "__main__":
    unittest.main()

