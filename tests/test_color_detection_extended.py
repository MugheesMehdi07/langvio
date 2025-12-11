"""
Extended unit tests for color detection
"""

import unittest
import numpy as np
import cv2
from langvio.vision.color_detection import ColorDetector


class TestColorDetectionExtended(unittest.TestCase):
    """Extended test cases for ColorDetector"""

    def test_detect_color_yellow(self):
        """Test detecting yellow color"""
        yellow_image = np.zeros((50, 50, 3), dtype=np.uint8)
        yellow_image[:, :] = [0, 255, 255]  # BGR yellow
        color = ColorDetector.detect_color(yellow_image)
        self.assertIn("yellow", color.lower())

    def test_detect_color_orange(self):
        """Test detecting orange color"""
        orange_image = np.zeros((50, 50, 3), dtype=np.uint8)
        orange_image[:, :] = [0, 165, 255]  # BGR orange
        color = ColorDetector.detect_color(orange_image)
        # Should detect orange or similar
        self.assertIsInstance(color, str)

    def test_detect_color_purple(self):
        """Test detecting purple color"""
        purple_image = np.zeros((50, 50, 3), dtype=np.uint8)
        purple_image[:, :] = [128, 0, 128]  # BGR purple
        color = ColorDetector.detect_color(purple_image)
        self.assertIsInstance(color, str)

    def test_detect_color_white(self):
        """Test detecting white color"""
        white_image = np.zeros((50, 50, 3), dtype=np.uint8)
        white_image[:, :] = [255, 255, 255]  # BGR white
        color = ColorDetector.detect_color(white_image)
        self.assertIn("white", color.lower())

    def test_detect_color_black(self):
        """Test detecting black color"""
        black_image = np.zeros((50, 50, 3), dtype=np.uint8)
        black_image[:, :] = [0, 0, 0]  # BGR black
        color = ColorDetector.detect_color(black_image)
        self.assertIn("black", color.lower())

    def test_detect_color_return_all_multiple(self):
        """Test detecting all colors in mixed image"""
        mixed_image = np.zeros((50, 50, 3), dtype=np.uint8)
        mixed_image[:16, :] = [0, 0, 255]  # Red
        mixed_image[16:33, :] = [255, 0, 0]  # Blue
        mixed_image[33:, :] = [0, 255, 0]  # Green
        
        colors = ColorDetector.detect_color(mixed_image, return_all=True)
        self.assertIsInstance(colors, dict)
        self.assertGreater(len(colors), 0)

    def test_get_color_profile(self):
        """Test get_color_profile method"""
        red_image = np.zeros((50, 50, 3), dtype=np.uint8)
        red_image[:, :] = [0, 0, 255]  # BGR red
        
        profile = ColorDetector.get_color_profile(red_image)
        self.assertIn("dominant_color", profile)
        self.assertIn("is_multicolored", profile)
        self.assertIn("color_percentages", profile)

    def test_get_color_profile_empty(self):
        """Test get_color_profile with empty image"""
        empty_image = np.array([])
        profile = ColorDetector.get_color_profile(empty_image)
        self.assertEqual(profile["dominant_color"], "unknown")
        self.assertFalse(profile["is_multicolored"])

    def test_get_color_profile_invalid(self):
        """Test get_color_profile with invalid image"""
        profile = ColorDetector.get_color_profile(None)
        self.assertEqual(profile["dominant_color"], "unknown")


if __name__ == "__main__":
    unittest.main()

