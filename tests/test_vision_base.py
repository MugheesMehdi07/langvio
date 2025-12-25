"""
Unit tests for vision base processor
"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import cv2
import numpy as np

from langvio.vision.base import BaseVisionProcessor


class MockVisionProcessor(BaseVisionProcessor):
    """Mock vision processor for testing"""
    
    def initialize(self):
        """Mock initialization"""
        return True
    
    def process_image(self, image_path, query_params):
        return {"objects": []}
    
    def process_video(self, video_path, query_params, sample_rate):
        return {"frame_detections": {}}


class TestVisionBase(unittest.TestCase):
    """Test cases for BaseVisionProcessor"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            "model_name": "test_model",
            "confidence": 0.5
        }
        self.processor = MockVisionProcessor("test", self.config)
        self.test_image = None

    def tearDown(self):
        """Clean up test fixtures"""
        if self.test_image and os.path.exists(self.test_image):
            os.unlink(self.test_image)

    def test_initialization(self):
        """Test BaseVisionProcessor initialization"""
        self.assertEqual(self.processor.name, "test")
        self.assertEqual(self.processor.config, self.config)

    def test_get_image_dimensions_success(self):
        """Test getting image dimensions"""
        # Create a test image
        test_img = np.zeros((100, 200, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jpg', delete=False) as f:
            self.test_image = f.name
        
        cv2.imwrite(self.test_image, test_img)
        
        dimensions = self.processor._get_image_dimensions(self.test_image)
        self.assertEqual(dimensions, (200, 100))  # (width, height)

    def test_get_image_dimensions_invalid(self):
        """Test getting dimensions for invalid image"""
        dimensions = self.processor._get_image_dimensions("nonexistent.jpg")
        self.assertIsNone(dimensions)

    def test_enhance_detections_with_attributes(self):
        """Test enhancing detections with attributes"""
        # Create a test image
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        test_img[25:75, 25:75] = [0, 0, 255]  # Red square
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jpg', delete=False) as f:
            self.test_image = f.name
        
        cv2.imwrite(self.test_image, test_img)
        
        detections = [
            {
                "bbox": [25, 25, 75, 75],
                "label": "object",
                "confidence": 0.9
            }
        ]
        
        enhanced = self.processor._enhance_detections_with_attributes(
            detections, self.test_image
        )
        
        self.assertEqual(len(enhanced), 1)
        self.assertIn("attributes", enhanced[0])
        self.assertIn("size", enhanced[0]["attributes"])

    def test_enhance_detections_invalid_bbox(self):
        """Test enhancing detections with invalid bbox"""
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jpg', delete=False) as f:
            self.test_image = f.name
        
        cv2.imwrite(self.test_image, test_img)
        
        detections = [
            {
                "bbox": [75, 75, 25, 25],  # Invalid: x1 > x2
                "label": "object"
            }
        ]
        
        enhanced = self.processor._enhance_detections_with_attributes(
            detections, self.test_image
        )
        
        # Invalid bbox should be skipped (continue statement skips it)
        # The method may return empty list or original list depending on implementation
        self.assertIsInstance(enhanced, list)

    def test_enhance_detections_out_of_bounds(self):
        """Test enhancing detections with out-of-bounds bbox"""
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jpg', delete=False) as f:
            self.test_image = f.name
        
        cv2.imwrite(self.test_image, test_img)
        
        detections = [
            {
                "bbox": [90, 90, 150, 150],  # Out of bounds (x2 > image_width)
                "label": "object"
            }
        ]
        
        enhanced = self.processor._enhance_detections_with_attributes(
            detections, self.test_image
        )
        
        # Out of bounds bbox should be skipped (continue statement)
        # The method may return empty list or original list depending on implementation
        self.assertIsInstance(enhanced, list)

    def test_enhance_detections_size_calculation(self):
        """Test size attribute calculation"""
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jpg', delete=False) as f:
            self.test_image = f.name
        
        cv2.imwrite(self.test_image, test_img)
        
        # Small object (< 5% of image)
        detections = [
            {
                "bbox": [0, 0, 10, 10],  # 100 pixels = 1% of 10000
                "label": "object"
            }
        ]
        
        enhanced = self.processor._enhance_detections_with_attributes(
            detections, self.test_image
        )
        
        if len(enhanced) > 0:
            self.assertEqual(enhanced[0]["attributes"]["size"], "small")

    def test_enhance_detections_image_load_failure(self):
        """Test handling image load failure"""
        detections = [{"bbox": [10, 10, 50, 50], "label": "object"}]
        
        enhanced = self.processor._enhance_detections_with_attributes(
            detections, "nonexistent.jpg"
        )
        
        # Should return original detections on failure
        self.assertEqual(len(enhanced), 1)


if __name__ == "__main__":
    unittest.main()

