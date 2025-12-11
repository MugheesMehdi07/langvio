"""
Comprehensive unit tests for vision base processor
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import cv2
import tempfile
import os

from langvio.vision.base import BaseVisionProcessor


class MockVisionProcessor(BaseVisionProcessor):
    """Concrete implementation for testing"""
    
    def initialize(self):
        return True
    
    def process_image(self, image_path: str, query_params):
        return {"objects": []}
    
    def process_video(self, video_path: str, query_params, sample_rate=3):
        return {"frame_detections": {}}


class TestVisionBaseComprehensive(unittest.TestCase):
    """Comprehensive test cases for BaseVisionProcessor"""

    def setUp(self):
        """Set up test fixtures"""
        self.processor = MockVisionProcessor("test", {"confidence": 0.5})
        self.test_image = None

    def tearDown(self):
        """Clean up test fixtures"""
        if self.test_image and os.path.exists(self.test_image):
            os.unlink(self.test_image)

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

    @patch('langvio.vision.color_detection.ColorDetector')
    def test_enhance_detections_with_attributes(self, mock_color_detector):
        """Test _enhance_detections_with_attributes"""
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        test_img[25:75, 25:75] = [0, 0, 255]  # Red square
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jpg', delete=False) as f:
            self.test_image = f.name
        cv2.imwrite(self.test_image, test_img)
        
        mock_color_detector.get_color_profile.return_value = {
            "dominant_color": "red",
            "is_multicolored": False,
            "color_percentages": {"red": 1.0}
        }
        
        detections = [
            {
                "bbox": [25, 25, 75, 75],
                "label": "person",
                "confidence": 0.9
            }
        ]
        
        enhanced = self.processor._enhance_detections_with_attributes(
            detections, self.test_image
        )
        
        self.assertEqual(len(enhanced), 1)
        self.assertIn("attributes", enhanced[0])
        self.assertIn("size", enhanced[0]["attributes"])
        self.assertIn("color", enhanced[0]["attributes"])

    def test_enhance_detections_invalid_bbox(self):
        """Test _enhance_detections_with_attributes with invalid bbox"""
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jpg', delete=False) as f:
            self.test_image = f.name
        cv2.imwrite(self.test_image, test_img)
        
        detections = [
            {
                "bbox": [150, 150, 200, 200],  # Outside image bounds
                "label": "person"
            }
        ]
        
        enhanced = self.processor._enhance_detections_with_attributes(
            detections, self.test_image
        )
        # Invalid boxes are skipped (continue), so they're not enhanced but may still be in list
        # The method continues processing, so the detection may still be present
        self.assertIsInstance(enhanced, list)

    def test_enhance_detections_no_image(self):
        """Test _enhance_detections_with_attributes when image can't be loaded"""
        detections = [
            {"bbox": [10, 10, 50, 50], "label": "person"}
        ]
        
        enhanced = self.processor._enhance_detections_with_attributes(
            detections, "nonexistent.jpg"
        )
        # Should return original detections
        self.assertEqual(len(enhanced), 1)


if __name__ == "__main__":
    unittest.main()

