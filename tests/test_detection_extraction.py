"""
Unit tests for detection extraction utilities
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import cv2
import torch

from langvio.utils.detection.extraction import (
    optimize_for_memory,
    force_cpu_mode,
    extract_detections,
    add_unified_attributes,
    add_tracking_info,
    add_color_attributes,
    add_size_and_position_attributes,
)


class TestDetectionExtraction(unittest.TestCase):
    """Test cases for detection extraction utilities"""

    @patch('torch.cuda.is_available', return_value=False)
    @patch('torch.set_num_threads')
    @patch('torch.set_grad_enabled')
    def test_optimize_for_memory_cpu(self, mock_grad, mock_threads, mock_cuda):
        """Test optimize_for_memory in CPU mode"""
        optimize_for_memory()
        mock_threads.assert_called()
        mock_grad.assert_called()

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.empty_cache')
    @patch('torch.cuda.synchronize')
    @patch('torch.set_grad_enabled')
    def test_optimize_for_memory_cuda(self, mock_grad, mock_sync, mock_cache, mock_cuda):
        """Test optimize_for_memory in CUDA mode"""
        optimize_for_memory()
        mock_cache.assert_called()
        mock_sync.assert_called()

    @patch('os.environ')
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.empty_cache')
    def test_force_cpu_mode(self, mock_cache, mock_cuda, mock_env):
        """Test force_cpu_mode"""
        force_cpu_mode()
        mock_env.__setitem__.assert_called()

    def test_extract_detections(self):
        """Test extract_detections from YOLO results"""
        # Create mock YOLO result
        mock_result = MagicMock()
        mock_box = MagicMock()
        mock_box.xyxy = MagicMock()
        mock_box.xyxy.__getitem__.return_value = MagicMock()
        mock_box.xyxy.__getitem__.return_value.tolist.return_value = [10, 20, 50, 60]
        mock_box.conf = MagicMock()
        mock_box.conf.__getitem__.return_value = 0.9
        mock_box.cls = MagicMock()
        mock_box.cls.__getitem__.return_value = 0
        
        mock_result.boxes = [mock_box]
        mock_result.names = {0: "person"}
        
        results = [mock_result]
        detections = extract_detections(results)
        
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["label"], "person")
        self.assertEqual(detections[0]["confidence"], 0.9)

    def test_add_unified_attributes_empty(self):
        """Test add_unified_attributes with empty detections"""
        result = add_unified_attributes([], 100, 100, None, False, False, False, False)
        self.assertEqual(result, [])

    def test_add_unified_attributes_basic(self):
        """Test add_unified_attributes with basic detections"""
        detections = [
            {"bbox": [10, 10, 50, 50], "label": "person", "confidence": 0.9}
        ]
        result = add_unified_attributes(
            detections, 100, 100, None, False, False, True, False
        )
        self.assertEqual(len(result), 1)
        self.assertIn("center", result[0])
        self.assertIn("object_id", result[0])

    def test_add_unified_attributes_with_size(self):
        """Test add_unified_attributes with size attributes"""
        detections = [
            {"bbox": [10, 10, 50, 50], "label": "person"}
        ]
        result = add_unified_attributes(
            detections, 100, 100, None, False, False, True, False
        )
        self.assertIn("attributes", result[0])
        self.assertIn("size", result[0]["attributes"])

    def test_add_unified_attributes_with_spatial(self):
        """Test add_unified_attributes with spatial attributes"""
        detections = [
            {"bbox": [10, 10, 50, 50], "label": "person"}
        ]
        result = add_unified_attributes(
            detections, 100, 100, None, False, True, False, False
        )
        self.assertIn("attributes", result[0])
        self.assertIn("position", result[0]["attributes"])

    def test_add_unified_attributes_invalid_bbox(self):
        """Test add_unified_attributes with invalid bbox"""
        detections = [
            {"bbox": [50, 50, 10, 10], "label": "person"}  # Invalid: x1 > x2
        ]
        result = add_unified_attributes(
            detections, 100, 100, None, False, False, False, False
        )
        # Should still return detection but skip enhancement
        self.assertEqual(len(result), 1)

    @patch('cv2.imread')
    @patch('langvio.vision.color_detection.ColorDetector')
    def test_add_unified_attributes_with_color(self, mock_color_detector, mock_imread):
        """Test add_unified_attributes with color detection"""
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = test_image
        
        mock_color_detector.get_color_profile.return_value = {
            "dominant_color": "red",
            "is_multicolored": False
        }
        
        detections = [
            {"bbox": [10, 10, 50, 50], "label": "person"}
        ]
        result = add_unified_attributes(
            detections, 100, 100, "test.jpg", True, False, False, False
        )
        self.assertIn("attributes", result[0])
        # Color detection may be disabled for video frames
        if "color" in result[0]["attributes"]:
            self.assertEqual(result[0]["attributes"]["color"], "red")

    def test_add_tracking_info(self):
        """Test add_tracking_info"""
        detections = [
            {"label": "person", "confidence": 0.9}
        ]
        result = add_tracking_info(detections, 0)
        self.assertIn("track_id", result[0])
        self.assertIn("object_id", result[0])

    def test_add_tracking_info_existing(self):
        """Test add_tracking_info with existing track_id"""
        detections = [
            {"label": "person", "track_id": "existing_track"}
        ]
        result = add_tracking_info(detections, 0)
        self.assertEqual(result[0]["track_id"], "existing_track")

    def test_add_color_attributes(self):
        """Test add_color_attributes"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[25:75, 25:75] = [0, 0, 255]  # Red square
        
        detections = [
            {"bbox": [25, 25, 75, 75], "label": "person"}
        ]
        
        with patch('langvio.vision.color_detection.ColorDetector') as mock_detector:
            mock_detector.detect_color.return_value = "red"
            result = add_color_attributes(detections, frame, True)
            self.assertIn("attributes", result[0])
            self.assertIn("color", result[0]["attributes"])

    def test_add_color_attributes_no_color(self):
        """Test add_color_attributes when color not needed"""
        detections = [{"bbox": [10, 10, 50, 50], "label": "person"}]
        result = add_color_attributes(detections, None, False)
        self.assertEqual(result, detections)

    def test_add_size_and_position_attributes(self):
        """Test add_size_and_position_attributes"""
        detections = [
            {"bbox": [10, 10, 50, 50], "label": "person"}
        ]
        result = add_size_and_position_attributes(detections, 100, 100)
        self.assertIn("center", result[0])
        self.assertIn("attributes", result[0])
        self.assertIn("size", result[0]["attributes"])
        self.assertIn("position", result[0]["attributes"])

    def test_add_size_and_position_attributes_no_bbox(self):
        """Test add_size_and_position_attributes without bbox"""
        detections = [{"label": "person"}]
        result = add_size_and_position_attributes(detections, 100, 100)
        self.assertEqual(result, detections)


if __name__ == "__main__":
    unittest.main()

