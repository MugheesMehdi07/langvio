"""
Unit tests for file utilities
"""

import os
import tempfile
import unittest

from langvio.utils.file_utils import (
    ensure_directory,
    get_file_extension,
    is_image_file,
    is_video_file,
    create_temp_copy,
    get_files_in_directory,
)


class TestFileUtils(unittest.TestCase):
    """Test cases for file utility functions"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_ensure_directory(self):
        """Test directory creation"""
        test_dir = os.path.join(self.temp_dir, "test_dir")
        ensure_directory(test_dir)
        self.assertTrue(os.path.isdir(test_dir))

    def test_get_file_extension(self):
        """Test getting file extension"""
        self.assertEqual(get_file_extension("test.jpg"), ".jpg")
        self.assertEqual(get_file_extension("test.PNG"), ".png")
        self.assertEqual(get_file_extension("no_ext"), "")

    def test_is_image_file(self):
        """Test image file detection"""
        self.assertTrue(is_image_file("test.jpg"))
        self.assertTrue(is_image_file("test.png"))
        self.assertTrue(is_image_file("test.JPEG"))
        self.assertFalse(is_image_file("test.mp4"))
        self.assertFalse(is_image_file("test.txt"))

    def test_is_video_file(self):
        """Test video file detection"""
        self.assertTrue(is_video_file("test.mp4"))
        self.assertTrue(is_video_file("test.avi"))
        self.assertTrue(is_video_file("test.MOV"))
        self.assertFalse(is_video_file("test.jpg"))
        self.assertFalse(is_video_file("test.txt"))

    def test_create_temp_copy(self):
        """Test creating temporary copy of file"""
        # Create a test file
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test content")

        temp_copy = create_temp_copy(test_file)
        self.assertTrue(os.path.exists(temp_copy))
        
        # Verify content
        with open(temp_copy, "r") as f:
            content = f.read()
        self.assertEqual(content, "test content")
        
        # Clean up
        if os.path.exists(temp_copy):
            os.unlink(temp_copy)

    def test_get_files_in_directory(self):
        """Test getting files in directory"""
        # Create test files
        test_files = ["test1.txt", "test2.jpg", "test3.mp4"]
        for filename in test_files:
            filepath = os.path.join(self.temp_dir, filename)
            with open(filepath, "w") as f:
                f.write("test")

        # Get all files
        all_files = get_files_in_directory(self.temp_dir)
        self.assertEqual(len(all_files), 3)

        # Get only image files
        image_files = get_files_in_directory(self.temp_dir, [".jpg"])
        self.assertEqual(len(image_files), 1)
        self.assertTrue(image_files[0].endswith(".jpg"))

        # Get files from non-existent directory
        non_existent_files = get_files_in_directory("/non/existent/dir")
        self.assertEqual(len(non_existent_files), 0)


if __name__ == "__main__":
    unittest.main()

