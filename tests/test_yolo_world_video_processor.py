"""
Unit tests for YOLO-World video processor
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import tempfile
import os

from langvio.vision.yolo_world.video_processor import YOLOWorldVideoProcessor


class TestYOLOWorldVideoProcessor(unittest.TestCase):
    """Test cases for YOLOWorldVideoProcessor"""

    def setUp(self):
        """Set up test fixtures"""
        self.model = MagicMock()
        self.config = {"confidence": 0.5}
        self.model_name = "yolov8m-worldv2"
        self.processor = YOLOWorldVideoProcessor(self.model, self.config, self.model_name)
        self.test_video = None

    def tearDown(self):
        """Clean up test fixtures"""
        if self.test_video and os.path.exists(self.test_video):
            os.unlink(self.test_video)

    def test_initialization(self):
        """Test YOLOWorldVideoProcessor initialization"""
        self.assertEqual(self.processor.model, self.model)
        self.assertEqual(self.processor.config, self.config)
        self.assertEqual(self.processor.model_name, self.model_name)
        self.assertIsNotNone(self.processor.tracker_file_manager)
        self.assertIsNotNone(self.processor.byte_tracker)

    @patch('cv2.VideoCapture')
    def test_initialize_video_capture(self, mock_video_capture):
        """Test _initialize_video_capture"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0  # fps
        mock_cap.get.side_effect = lambda x: {
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 300
        }.get(x, 0)
        mock_video_capture.return_value = mock_cap
        
        import cv2
        cap, props = self.processor._initialize_video_capture("test.mp4")
        self.assertIsNotNone(cap)
        self.assertEqual(len(props), 4)  # width, height, fps, total_frames

    @patch('cv2.VideoCapture')
    def test_initialize_video_capture_failure(self, mock_video_capture):
        """Test _initialize_video_capture with failure"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap
        
        import cv2
        with self.assertRaises(ValueError):
            self.processor._initialize_video_capture("nonexistent.mp4")

    def test_run_detection(self):
        """Test _run_detection"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Mock YOLO model result
        mock_result = MagicMock()
        mock_result.boxes = MagicMock()
        mock_result.boxes.xyxy = MagicMock()
        mock_result.boxes.xyxy.cpu.return_value = MagicMock()
        mock_result.boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[10, 10, 50, 50]])
        mock_result.boxes.conf = MagicMock()
        mock_result.boxes.conf.cpu.return_value = MagicMock()
        mock_result.boxes.conf.cpu.return_value.numpy.return_value = np.array([0.9])
        mock_result.boxes.cls = MagicMock()
        mock_result.boxes.cls.cpu.return_value = MagicMock()
        mock_result.boxes.cls.cpu.return_value.numpy.return_value = np.array([0])
        mock_result.names = {0: "person"}
        
        self.model.return_value = [mock_result]
        
        detections = self.processor._run_detection(frame, 100, 100)
        self.assertIsInstance(detections, list)

    @patch('langvio.utils.tracking.tracker_file_manager.TrackerFileManager')
    @patch('cv2.VideoCapture')
    def test_process_with_existing_tracker_file(self, mock_video_capture, mock_tracker_mgr):
        """Test process when tracker file exists"""
        mock_tracker_instance = MagicMock()
        mock_tracker_instance.get_tracker_file_if_exists.return_value = "existing_tracker.json"
        mock_tracker_instance.load_tracker_data.return_value = {
            "detections": [],
            "metadata": {}
        }
        mock_tracker_instance.convert_to_legacy_format.return_value = {
            "frame_detections": {}
        }
        mock_tracker_mgr.return_value = mock_tracker_instance
        
        self.processor.tracker_file_manager = mock_tracker_instance
        
        result = self.processor.process("test.mp4", {}, sample_rate=3)
        self.assertIn("frame_detections", result)

    @patch('langvio.utils.tracking.tracker_file_manager.TrackerFileManager')
    @patch('cv2.VideoCapture')
    def test_process_video_with_tracking(self, mock_video_capture, mock_tracker_mgr):
        """Test _process_video_with_tracking"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [
            (True, np.zeros((100, 100, 3), dtype=np.uint8)),
            (False, None)
        ]
        mock_cap.get.side_effect = lambda x: {
            cv2.CAP_PROP_FRAME_WIDTH: 100,
            cv2.CAP_PROP_FRAME_HEIGHT: 100,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 1
        }.get(x, 0)
        mock_video_capture.return_value = mock_cap
        
        import cv2
        import torch
        
        # Mock model result
        mock_result = MagicMock()
        mock_result.boxes = None
        self.model.return_value = [mock_result]
        
        mock_tracker_instance = MagicMock()
        mock_tracker_instance.save_tracker_data.return_value = "tracker.json"
        mock_tracker_instance.convert_to_legacy_format.return_value = {
            "frame_detections": {}
        }
        mock_tracker_mgr.return_value = mock_tracker_instance
        
        self.processor.tracker_file_manager = mock_tracker_instance
        
        with patch('torch.no_grad'):
            with patch('langvio.utils.detection.extraction.optimize_for_memory'):
                result = self.processor._process_video_with_tracking("test.mp4", {}, 1)
                self.assertIn("frame_detections", result)


if __name__ == "__main__":
    unittest.main()

