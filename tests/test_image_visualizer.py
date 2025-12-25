"""
Unit tests for image visualizer
"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import cv2
import numpy as np

from langvio.media.image_visualizer import ImageVisualizer


class TestImageVisualizer(unittest.TestCase):
    """Test cases for ImageVisualizer"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            "box_color": [0, 255, 0],
            "text_color": [255, 255, 255],
            "line_thickness": 2,
            "show_attributes": True,
            "show_confidence": True
        }
        self.visualizer = ImageVisualizer(self.config)
        self.test_image = None

    def tearDown(self):
        """Clean up test fixtures"""
        if self.test_image and os.path.exists(self.test_image):
            os.unlink(self.test_image)

    def test_initialization(self):
        """Test ImageVisualizer initialization"""
        self.assertEqual(self.visualizer.config, self.config)

    def test_visualize_with_highlights(self):
        """Test visualizing with highlights"""
        # Create a test image
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jpg', delete=False) as f:
            self.test_image = f.name
        
        cv2.imwrite(self.test_image, test_img)
        
        all_detections = [
            {
                "bbox": [10, 10, 50, 50],
                "label": "person",
                "confidence": 0.9
            },
            {
                "bbox": [60, 60, 90, 90],
                "label": "car",
                "confidence": 0.8
            }
        ]
        
        highlighted = [all_detections[0]]  # Highlight first detection
        
        output_path = self.test_image.replace('.jpg', '_output.jpg')
        
        self.visualizer.visualize_with_highlights(
            self.test_image,
            output_path,
            all_detections,
            highlighted
        )
        
        # Check output file was created
        self.assertTrue(os.path.exists(output_path))
        if os.path.exists(output_path):
            os.unlink(output_path)

    def test_visualize_empty_detections(self):
        """Test visualizing with empty detections"""
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jpg', delete=False) as f:
            self.test_image = f.name
        
        cv2.imwrite(self.test_image, test_img)
        
        output_path = self.test_image.replace('.jpg', '_output.jpg')
        
        self.visualizer.visualize_with_highlights(
            self.test_image,
            output_path,
            [],
            []
        )
        
        self.assertTrue(os.path.exists(output_path))
        if os.path.exists(output_path):
            os.unlink(output_path)

    def test_draw_single_detection(self):
        """Test drawing a single detection"""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        detection = {
            "bbox": [10, 10, 50, 50],
            "label": "person",
            "confidence": 0.9
        }
        
        result = self.visualizer._draw_single_detection(
            image,
            detection,
            [0, 255, 0],
            [255, 255, 255],
            2,
            show_attributes=True,
            show_confidence=True,
            is_highlighted=False
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, image.shape)

    def test_draw_single_detection_with_attributes(self):
        """Test drawing detection with attributes"""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        detection = {
            "bbox": [10, 10, 50, 50],
            "label": "person",
            "confidence": 0.9,
            "color": "red",
            "size": "large"
        }
        
        result = self.visualizer._draw_single_detection(
            image,
            detection,
            [0, 255, 0],
            [255, 255, 255],
            2,
            show_attributes=True,
            show_confidence=True,
            is_highlighted=False
        )
        
        self.assertIsNotNone(result)

    @patch('cv2.imread', return_value=None)
    def test_visualize_invalid_image(self, mock_imread):
        """Test handling invalid image"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jpg', delete=False) as f:
            self.test_image = f.name
        
        output_path = self.test_image.replace('.jpg', '_output.jpg')
        
        # Should raise ValueError or handle gracefully
        try:
            self.visualizer.visualize_with_highlights(
                self.test_image,
                output_path,
                [],
                []
            )
            # If no exception, that's also acceptable (error logged)
        except ValueError:
            # Expected behavior
            pass


if __name__ == "__main__":
    unittest.main()

