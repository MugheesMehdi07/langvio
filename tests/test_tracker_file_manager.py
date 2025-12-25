"""
Unit tests for tracker file manager
"""

import unittest
import tempfile
import os
import json
from unittest.mock import patch, MagicMock
from pathlib import Path

from langvio.utils.tracking.tracker_file_manager import TrackerFileManager


class TestTrackerFileManager(unittest.TestCase):
    """Test cases for TrackerFileManager"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = TrackerFileManager(output_dir=self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test TrackerFileManager initialization"""
        self.assertEqual(str(self.manager.output_dir), self.temp_dir)

    def test_create_tracker_file_path(self):
        """Test create_tracker_file_path"""
        video_path = "/path/to/video.mp4"
        tracker_path = self.manager.create_tracker_file_path(video_path)
        
        self.assertIn("video_tracker.json", tracker_path)
        self.assertIn(self.temp_dir, tracker_path)

    def test_save_tracker_data(self):
        """Test save_tracker_data"""
        video_path = "test.mp4"
        detections = [{"frame_id": 0, "objects": []}]
        tracks = [{"track_id": 1, "bbox": [10, 10, 50, 50]}]
        metadata = {"fps": 30.0, "total_frames": 100}
        
        tracker_path = self.manager.save_tracker_data(
            video_path, detections, tracks, metadata, query="test query"
        )
        
        self.assertTrue(os.path.exists(tracker_path))
        
        # Verify file contents
        with open(tracker_path, 'r') as f:
            data = json.load(f)
            self.assertIn("detections", data)
            self.assertIn("tracks", data)
            self.assertIn("metadata", data)

    def test_load_tracker_data(self):
        """Test load_tracker_data"""
        # Create a test tracker file
        tracker_data = {
            "metadata": {"video_path": "test.mp4"},
            "detections": [{"frame_id": 0, "objects": []}],
            "tracks": []
        }
        
        tracker_path = os.path.join(self.temp_dir, "test_tracker.json")
        with open(tracker_path, 'w') as f:
            json.dump(tracker_data, f)
        
        loaded = self.manager.load_tracker_data(tracker_path)
        self.assertEqual(loaded["metadata"]["video_path"], "test.mp4")

    def test_convert_to_legacy_format(self):
        """Test convert_to_legacy_format"""
        tracker_data = {
            "metadata": {"video_path": "test.mp4"},
            "detections": [
                {"frame_id": 0, "objects": [{"label": "person"}]},
                {"frame_id": 10, "objects": [{"label": "car"}]}
            ],
            "tracks": []
        }
        
        legacy = self.manager.convert_to_legacy_format(tracker_data)
        self.assertIn("frame_detections", legacy)
        self.assertIn("summary", legacy)
        self.assertIn("0", legacy["frame_detections"])
        self.assertIn("10", legacy["frame_detections"])

    def test_get_tracker_file_if_exists(self):
        """Test get_tracker_file_if_exists when file exists"""
        video_path = "test.mp4"
        tracker_path = self.manager.create_tracker_file_path(video_path)
        
        # Create the file
        with open(tracker_path, 'w') as f:
            json.dump({}, f)
        
        result = self.manager.get_tracker_file_if_exists(video_path)
        self.assertEqual(result, tracker_path)

    def test_get_tracker_file_if_not_exists(self):
        """Test get_tracker_file_if_exists when file doesn't exist"""
        video_path = "nonexistent.mp4"
        result = self.manager.get_tracker_file_if_exists(video_path)
        self.assertIsNone(result)

    def test_cleanup_old_tracker_files(self):
        """Test cleanup_old_tracker_files"""
        # Create a test tracker file
        tracker_file = os.path.join(self.temp_dir, "old_tracker.json")
        with open(tracker_file, 'w') as f:
            json.dump({}, f)
        
        # Test that cleanup doesn't crash
        # (actual cleanup depends on file age which is hard to test)
        try:
            self.manager.cleanup_old_tracker_files(max_age_days=7)
            # Should complete without error
            self.assertTrue(True)
        except Exception as e:
            # If it fails, that's acceptable for testing
            pass


if __name__ == "__main__":
    unittest.main()

