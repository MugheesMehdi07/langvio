"""
Comprehensive unit tests for color detection
"""

import unittest
import numpy as np
import cv2
from langvio.vision.color_detection import ColorDetector


class TestColorDetectionComprehensive(unittest.TestCase):
    """Comprehensive test cases for ColorDetector"""

    def test_detect_colors_layered(self):
        """Test detect_colors_layered"""
        # Create a mixed color image
        mixed_image = np.zeros((50, 50, 3), dtype=np.uint8)
        mixed_image[:25, :] = [0, 0, 255]  # Red
        mixed_image[25:, :] = [255, 0, 0]  # Blue
        
        colors = ColorDetector.detect_colors_layered(mixed_image, max_colors=2)
        self.assertIsInstance(colors, list)
        self.assertLessEqual(len(colors), 2)

    def test_get_color_profile_comprehensive(self):
        """Test get_color_profile comprehensively"""
        red_image = np.zeros((50, 50, 3), dtype=np.uint8)
        red_image[:, :] = [0, 0, 255]  # BGR red
        
        profile = ColorDetector.get_color_profile(red_image)
        self.assertIn("dominant_color", profile)
        self.assertIn("color_percentages", profile)
        self.assertIn("is_multicolored", profile)
        self.assertIn("brightness", profile)
        self.assertIn("saturation", profile)
        self.assertIsInstance(profile["brightness"], (int, float))
        self.assertIsInstance(profile["saturation"], (int, float))

    def test_get_color_profile_multicolored(self):
        """Test get_color_profile with multicolored image"""
        multicolored = np.zeros((50, 50, 3), dtype=np.uint8)
        multicolored[:16, :] = [0, 0, 255]  # Red
        multicolored[16:33, :] = [255, 0, 0]  # Blue
        multicolored[33:, :] = [0, 255, 0]  # Green
        
        profile = ColorDetector.get_color_profile(multicolored)
        self.assertTrue(profile["is_multicolored"])

    def test_detect_color_threshold(self):
        """Test detect_color with threshold parameter"""
        # Create image with low color coverage
        mostly_black = np.zeros((50, 50, 3), dtype=np.uint8)
        mostly_black[10:15, 10:15] = [0, 0, 255]  # Small red region
        
        # With high threshold, should return multicolored or unknown
        color = ColorDetector.detect_color(mostly_black, threshold=0.5)
        self.assertIsInstance(color, str)

    def test_detect_color_cyan(self):
        """Test detecting cyan color"""
        cyan_image = np.zeros((50, 50, 3), dtype=np.uint8)
        cyan_image[:, :] = [255, 255, 0]  # BGR cyan
        color = ColorDetector.detect_color(cyan_image)
        self.assertIsInstance(color, str)

    def test_detect_color_pink(self):
        """Test detecting pink color"""
        pink_image = np.zeros((50, 50, 3), dtype=np.uint8)
        pink_image[:, :] = [203, 192, 255]  # BGR pink
        color = ColorDetector.detect_color(pink_image)
        self.assertIsInstance(color, str)


if __name__ == "__main__":
    unittest.main()

