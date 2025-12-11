"""
Unit tests for color detection
"""

import unittest
import numpy as np
import cv2
from langvio.vision.color_detection import ColorDetector


class TestColorDetection(unittest.TestCase):
    """Test cases for ColorDetector"""

    def test_detect_color_red(self):
        """Test detecting red color"""
        # Create a red image region
        red_image = np.zeros((50, 50, 3), dtype=np.uint8)
        red_image[:, :] = [0, 0, 255]  # BGR red
        
        color = ColorDetector.detect_color(red_image)
        self.assertIn("red", color.lower())

    def test_detect_color_blue(self):
        """Test detecting blue color"""
        # Create a blue image region
        blue_image = np.zeros((50, 50, 3), dtype=np.uint8)
        blue_image[:, :] = [255, 0, 0]  # BGR blue
        
        color = ColorDetector.detect_color(blue_image)
        self.assertIn("blue", color.lower())

    def test_detect_color_green(self):
        """Test detecting green color"""
        # Create a green image region
        green_image = np.zeros((50, 50, 3), dtype=np.uint8)
        green_image[:, :] = [0, 255, 0]  # BGR green
        
        color = ColorDetector.detect_color(green_image)
        self.assertIn("green", color.lower())

    def test_detect_color_empty_region(self):
        """Test detecting color in empty region"""
        empty_image = np.array([])
        color = ColorDetector.detect_color(empty_image)
        self.assertEqual(color, "unknown")

    def test_detect_color_invalid_region(self):
        """Test detecting color in invalid region"""
        invalid_image = None
        color = ColorDetector.detect_color(invalid_image)
        self.assertEqual(color, "unknown")

    def test_detect_color_return_all(self):
        """Test detecting all colors"""
        # Create a mixed color image
        mixed_image = np.zeros((50, 50, 3), dtype=np.uint8)
        mixed_image[:25, :] = [0, 0, 255]  # Red
        mixed_image[25:, :] = [255, 0, 0]  # Blue
        
        colors = ColorDetector.detect_color(mixed_image, return_all=True)
        self.assertIsInstance(colors, dict)
        self.assertGreater(len(colors), 0)

    def test_detect_color_grayscale(self):
        """Test detecting grayscale colors"""
        # Create a gray image
        gray_image = np.zeros((50, 50, 3), dtype=np.uint8)
        gray_image[:, :] = [128, 128, 128]  # Gray
        
        color = ColorDetector.detect_color(gray_image)
        self.assertIn("gray", color.lower())


if __name__ == "__main__":
    unittest.main()

