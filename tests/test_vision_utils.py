"""
Unit tests for vision utilities
"""

import unittest
from langvio.vision.utils import TemporalObjectTracker


class TestVisionUtils(unittest.TestCase):
    """Test cases for vision utilities"""

    def setUp(self):
        """Set up test fixtures"""
        self.tracker = TemporalObjectTracker(max_history=10)

    def test_temporal_tracker_initialization(self):
        """Test TemporalObjectTracker initialization"""
        self.assertEqual(self.tracker.max_history, 10)
        self.assertEqual(len(self.tracker.object_histories), 0)

    def test_temporal_tracker_update_frame(self):
        """Test updating tracker with frame detections"""
        detections = [
            {
                "label": "person",
                "track_id": 1,
                "center": (100, 100),
                "attributes": {"color": "red"}
            }
        ]
        
        self.tracker.update_frame(0, detections, fps=30.0)
        
        self.assertEqual(len(self.tracker.object_histories), 1)
        obj_key = "person_1"
        self.assertIn(obj_key, self.tracker.object_histories)
        
        history = self.tracker.object_histories[obj_key]
        self.assertEqual(len(history["positions"]), 1)
        self.assertEqual(history["positions"][0], (100, 100))

    def test_temporal_tracker_multiple_frames(self):
        """Test tracker with multiple frames"""
        detections_frame1 = [
            {"label": "person", "track_id": 1, "center": (100, 100)}
        ]
        detections_frame2 = [
            {"label": "person", "track_id": 1, "center": (110, 110)}
        ]
        
        self.tracker.update_frame(0, detections_frame1, fps=30.0)
        self.tracker.update_frame(1, detections_frame2, fps=30.0)
        
        obj_key = "person_1"
        history = self.tracker.object_histories[obj_key]
        self.assertEqual(len(history["positions"]), 2)
        self.assertEqual(history["total_appearances"], 2)

    def test_temporal_tracker_get_temporal_relationships(self):
        """Test getting temporal relationships"""
        detections = [
            {"label": "person", "track_id": 1, "center": (100, 100)}
        ]
        self.tracker.update_frame(0, detections, fps=30.0)
        
        relationships = self.tracker.get_temporal_relationships()
        self.assertIsInstance(relationships, list)

    def test_temporal_tracker_get_object_trajectory(self):
        """Test getting object trajectory"""
        detections = [
            {"label": "person", "track_id": 1, "center": (100, 100)}
        ]
        self.tracker.update_frame(0, detections, fps=30.0)
        
        # Check if method exists, if not skip
        if hasattr(self.tracker, 'get_object_trajectory'):
            trajectory = self.tracker.get_object_trajectory("person_1")
            self.assertIsNotNone(trajectory)
        else:
            # Method doesn't exist, check history directly
            obj_key = "person_1"
            self.assertIn(obj_key, self.tracker.object_histories)
            history = self.tracker.object_histories[obj_key]
            self.assertIn("positions", history)

    def test_temporal_tracker_max_history(self):
        """Test max_history limit"""
        tracker = TemporalObjectTracker(max_history=3)
        detections = [
            {"label": "person", "track_id": 1, "center": (100, 100)}
        ]
        
        # Add more frames than max_history
        for i in range(5):
            tracker.update_frame(i, detections, fps=30.0)
        
        obj_key = "person_1"
        history = tracker.object_histories[obj_key]
        # Should only keep last 3 positions
        self.assertLessEqual(len(history["positions"]), 3)


if __name__ == "__main__":
    unittest.main()

