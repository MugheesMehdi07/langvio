"""
Extended unit tests for vision utilities
"""

import unittest
from langvio.vision.utils import TemporalObjectTracker


class TestVisionUtilsExtended(unittest.TestCase):
    """Extended test cases for vision utilities"""

    def setUp(self):
        """Set up test fixtures"""
        self.tracker = TemporalObjectTracker(max_history=10)

    def test_calculate_temporal_overlap(self):
        """Test _calculate_temporal_overlap"""
        hist1 = {
            "first_seen": 0.0,
            "last_seen": 10.0,
            "timestamps": [0.0, 5.0, 10.0]
        }
        hist2 = {
            "first_seen": 5.0,
            "last_seen": 15.0,
            "timestamps": [5.0, 10.0, 15.0]
        }
        
        overlap = self.tracker._calculate_temporal_overlap(hist1, hist2)
        # Overlap should be between 0 and 1
        self.assertGreaterEqual(overlap, 0)
        self.assertLessEqual(overlap, 1.0)

    def test_calculate_total_movement(self):
        """Test _calculate_total_movement"""
        positions = [(0, 0), (10, 0), (20, 0), (30, 0)]
        total = self.tracker._calculate_total_movement(positions)
        self.assertEqual(total, 30.0)

    def test_calculate_total_movement_empty(self):
        """Test _calculate_total_movement with empty positions"""
        total = self.tracker._calculate_total_movement([])
        self.assertEqual(total, 0.0)

    def test_calculate_average_speed(self):
        """Test _calculate_average_speed"""
        positions = [(0, 0), (10, 0), (20, 0)]
        timestamps = [0.0, 1.0, 2.0]
        speed = self.tracker._calculate_average_speed(positions, timestamps)
        self.assertGreater(speed, 0)

    def test_calculate_average_speed_zero_time(self):
        """Test _calculate_average_speed with zero time"""
        positions = [(0, 0), (10, 0)]
        timestamps = [0.0, 0.0]
        speed = self.tracker._calculate_average_speed(positions, timestamps)
        self.assertEqual(speed, 0.0)

    def test_get_primary_direction_right(self):
        """Test _get_primary_direction for right movement"""
        positions = [(0, 0), (50, 0)]
        direction = self.tracker._get_primary_direction(positions)
        self.assertEqual(direction, "right")

    def test_get_primary_direction_left(self):
        """Test _get_primary_direction for left movement"""
        positions = [(50, 0), (0, 0)]
        direction = self.tracker._get_primary_direction(positions)
        self.assertEqual(direction, "left")

    def test_get_primary_direction_stationary(self):
        """Test _get_primary_direction for stationary object"""
        positions = [(10, 10), (11, 11)]  # Minimal movement
        direction = self.tracker._get_primary_direction(positions)
        self.assertEqual(direction, "stationary")

    def test_get_temporal_relationships_with_overlap(self):
        """Test get_temporal_relationships with overlapping objects"""
        detections1 = [{"label": "person", "track_id": 1, "center": (100, 100)}]
        detections2 = [{"label": "car", "track_id": 2, "center": (110, 110)}]
        
        self.tracker.update_frame(0, detections1, fps=30.0)
        self.tracker.update_frame(5, detections2, fps=30.0)
        self.tracker.update_frame(10, detections1 + detections2, fps=30.0)
        
        relationships = self.tracker.get_temporal_relationships()
        self.assertIsInstance(relationships, list)

    def test_get_movement_patterns(self):
        """Test get_movement_patterns"""
        detections = [
            {"label": "person", "track_id": 1, "center": (100, 100)},
            {"label": "person", "track_id": 1, "center": (150, 100)},  # Moving right
            {"label": "person", "track_id": 1, "center": (200, 100)}   # More movement
        ]
        
        self.tracker.update_frame(0, [detections[0]], fps=30.0)
        self.tracker.update_frame(10, [detections[1]], fps=30.0)
        self.tracker.update_frame(20, [detections[2]], fps=30.0)
        
        patterns = self.tracker.get_movement_patterns()
        # get_movement_patterns returns the dict directly, not nested
        self.assertIn("directional_movements", patterns)
        self.assertIn("moving_objects", patterns)
        self.assertIn("stationary_objects", patterns)


if __name__ == "__main__":
    unittest.main()

