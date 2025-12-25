"""
Unit tests for YOLO-World image processor
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import cv2
import tempfile
import os

from langvio.vision.yolo_world.image_processor import YOLOWorldImageProcessor


class TestYOLOWorldImageProcessor(unittest.TestCase):
    """Test cases for YOLOWorldImageProcessor"""

    def setUp(self):
        """Set up test fixtures"""
        self.model = MagicMock()
        self.config = {"confidence": 0.5}
        self.processor = YOLOWorldImageProcessor(self.model, self.config)
        self.test_image = None

    def tearDown(self):
        """Clean up test fixtures"""
        if self.test_image and os.path.exists(self.test_image):
            os.unlink(self.test_image)

    def test_initialization(self):
        """Test YOLOWorldImageProcessor initialization"""
        self.assertEqual(self.processor.model, self.model)
        self.assertEqual(self.processor.config, self.config)

    def test_get_image_dimensions_success(self):
        """Test _get_image_dimensions with valid image"""
        test_img = np.zeros((100, 200, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jpg', delete=False) as f:
            self.test_image = f.name
        cv2.imwrite(self.test_image, test_img)
        
        dimensions = self.processor._get_image_dimensions(self.test_image)
        self.assertEqual(dimensions, (200, 100))  # (width, height)

    def test_get_image_dimensions_invalid(self):
        """Test _get_image_dimensions with invalid image"""
        dimensions = self.processor._get_image_dimensions("nonexistent.jpg")
        self.assertIsNone(dimensions)

    @patch('langvio.utils.detection.extraction.add_unified_attributes')
    @patch('langvio.utils.detection.compress_detections_for_output')
    def test_process_success(self, mock_compress, mock_add_attrs):
        """Test process with successful detection"""
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jpg', delete=False) as f:
            self.test_image = f.name
        cv2.imwrite(self.test_image, test_img)
        
        # Mock model results
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
        
        mock_add_attrs.return_value = [
            {"bbox": [10, 10, 50, 50], "label": "person", "confidence": 0.9}
        ]
        mock_compress.return_value = [{"id": "obj_1", "type": "person"}]
        
        result = self.processor.process(self.test_image, {"task_type": "identification"})
        
        self.assertIn("objects", result)
        self.assertIn("summary", result)

    def test_extract_detections(self):
        """Test _extract_detections"""
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
        
        detections = self.processor._extract_detections(mock_result)
        
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["label"], "person")
        self.assertEqual(detections[0]["confidence"], 0.9)

    def test_extract_detections_empty(self):
        """Test _extract_detections with no boxes"""
        mock_result = MagicMock()
        mock_result.boxes = None
        
        detections = self.processor._extract_detections(mock_result)
        self.assertEqual(detections, [])

    def test_create_image_summary(self):
        """Test _create_image_summary"""
        detections = [
            {"label": "person", "confidence": 0.9},
            {"label": "person", "confidence": 0.8},
            {"label": "car", "confidence": 0.7}
        ]
        query_params = {"task_type": "counting"}
        
        summary = self.processor._create_image_summary(detections, 100, 100, query_params)
        
        self.assertEqual(summary["total_objects"], 3)
        self.assertEqual(summary["object_counts"]["person"], 2)
        self.assertEqual(summary["object_counts"]["car"], 1)
        self.assertEqual(summary["query_type"], "counting")


if __name__ == "__main__":
    unittest.main()

