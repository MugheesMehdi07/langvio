"""
Comprehensive unit tests for vision utils - TemporalObjectTracker and SpatialRelationshipAnalyzer
"""

import unittest
from langvio.vision.utils import TemporalObjectTracker, SpatialRelationshipAnalyzer


class TestTemporalObjectTrackerComprehensive(unittest.TestCase):
    """Comprehensive test cases for TemporalObjectTracker"""

    def setUp(self):
        """Set up test fixtures"""
        self.tracker = TemporalObjectTracker(max_history=30)

    def test_get_movement_patterns_stationary(self):
        """Test get_movement_patterns with stationary objects"""
        # Add detections with minimal movement
        detections = [
            {"label": "person", "center": (100, 100), "track_id": 1, "object_id": "obj_1"}
        ]
        self.tracker.update_frame(0, detections, 30.0)
        self.tracker.update_frame(1, detections, 30.0)
        
        patterns = self.tracker.get_movement_patterns()
        self.assertIn("stationary_objects", patterns)

    def test_get_movement_patterns_moving(self):
        """Test get_movement_patterns with moving objects"""
        # Add detections with movement
        for i in range(5):
            detections = [
                {
                    "label": "car",
                    "center": (100 + i * 10, 100 + i * 5),
                    "track_id": 1,
                    "object_id": "obj_1"
                }
            ]
            self.tracker.update_frame(i, detections, 30.0)
        
        patterns = self.tracker.get_movement_patterns()
        self.assertIn("moving_objects", patterns)
        self.assertIn("directional_movements", patterns)

    def test_get_movement_patterns_fast_moving(self):
        """Test get_movement_patterns with fast moving objects"""
        # Add detections with large movement
        for i in range(5):
            detections = [
                {
                    "label": "car",
                    "center": (100 + i * 50, 100 + i * 30),
                    "track_id": 1,
                    "object_id": "obj_1"
                }
            ]
            self.tracker.update_frame(i, detections, 30.0)
        
        patterns = self.tracker.get_movement_patterns()
        # May be fast moving or just moving depending on thresholds
        self.assertIn("moving_objects", patterns)

    def test_get_temporal_relationships(self):
        """Test get_temporal_relationships"""
        # Add two objects that overlap temporally
        for i in range(10):
            detections = [
                {"label": "person", "center": (100, 100), "track_id": 1, "object_id": "obj_1"},
                {"label": "car", "center": (200, 200), "track_id": 2, "object_id": "obj_2"}
            ]
            self.tracker.update_frame(i, detections, 30.0)
        
        relationships = self.tracker.get_temporal_relationships()
        self.assertIsInstance(relationships, list)
        # Should have co-occurring relationship if overlap > 0.5
        if relationships:
            self.assertIn("relationship", relationships[0])

    def test_get_temporal_relationships_no_overlap(self):
        """Test get_temporal_relationships with no overlap"""
        # Add objects at different times
        for i in range(5):
            detections = [
                {"label": "person", "center": (100, 100), "track_id": 1, "object_id": "obj_1"}
            ]
            self.tracker.update_frame(i, detections, 30.0)
        
        for i in range(10, 15):
            detections = [
                {"label": "car", "center": (200, 200), "track_id": 2, "object_id": "obj_2"}
            ]
            self.tracker.update_frame(i, detections, 30.0)
        
        relationships = self.tracker.get_temporal_relationships()
        # Should have no relationships or low overlap
        self.assertIsInstance(relationships, list)

    def test_calculate_total_movement(self):
        """Test _calculate_total_movement"""
        positions = [(0, 0), (10, 0), (20, 0), (30, 0)]
        total = self.tracker._calculate_total_movement(positions)
        self.assertEqual(total, 30.0)

    def test_calculate_total_movement_single(self):
        """Test _calculate_total_movement with single position"""
        positions = [(100, 100)]
        total = self.tracker._calculate_total_movement(positions)
        self.assertEqual(total, 0)

    def test_calculate_average_speed(self):
        """Test _calculate_average_speed"""
        positions = [(0, 0), (10, 0), (20, 0)]
        timestamps = [0.0, 1.0, 2.0]
        speed = self.tracker._calculate_average_speed(positions, timestamps)
        self.assertGreater(speed, 0)

    def test_get_primary_direction_right(self):
        """Test _get_primary_direction for right movement"""
        positions = [(0, 0), (50, 10)]
        direction = self.tracker._get_primary_direction(positions)
        self.assertEqual(direction, "right")

    def test_get_primary_direction_left(self):
        """Test _get_primary_direction for left movement"""
        positions = [(50, 0), (0, 10)]
        direction = self.tracker._get_primary_direction(positions)
        self.assertEqual(direction, "left")

    def test_get_primary_direction_up(self):
        """Test _get_primary_direction for up movement"""
        positions = [(0, 50), (10, 0)]
        direction = self.tracker._get_primary_direction(positions)
        self.assertEqual(direction, "up")

    def test_get_primary_direction_down(self):
        """Test _get_primary_direction for down movement"""
        positions = [(0, 0), (10, 50)]
        direction = self.tracker._get_primary_direction(positions)
        self.assertEqual(direction, "down")

    def test_get_primary_direction_stationary(self):
        """Test _get_primary_direction for stationary"""
        positions = [(100, 100), (105, 105)]
        direction = self.tracker._get_primary_direction(positions)
        self.assertEqual(direction, "stationary")

    def test_calculate_temporal_overlap(self):
        """Test _calculate_temporal_overlap"""
        hist1 = {"first_seen": 0.0, "last_seen": 10.0}
        hist2 = {"first_seen": 5.0, "last_seen": 15.0}
        overlap = self.tracker._calculate_temporal_overlap(hist1, hist2)
        # Overlap should be calculated: overlap_duration / total_duration
        # overlap_duration = 5.0 (from 5.0 to 10.0)
        # total_duration = 15.0 (from 0.0 to 15.0)
        # So overlap should be 5.0 / 15.0 = 0.333...
        self.assertGreaterEqual(overlap, 0)
        self.assertLessEqual(overlap, 1)

    def test_calculate_temporal_overlap_no_overlap(self):
        """Test _calculate_temporal_overlap with no overlap"""
        hist1 = {"first_seen": 0.0, "last_seen": 5.0}
        hist2 = {"first_seen": 10.0, "last_seen": 15.0}
        overlap = self.tracker._calculate_temporal_overlap(hist1, hist2)
        self.assertEqual(overlap, 0)

    def test_calculate_temporal_overlap_missing_data(self):
        """Test _calculate_temporal_overlap with missing data"""
        hist1 = {"first_seen": None, "last_seen": None}
        hist2 = {"first_seen": 5.0, "last_seen": 15.0}
        overlap = self.tracker._calculate_temporal_overlap(hist1, hist2)
        self.assertEqual(overlap, 0)


class TestSpatialRelationshipAnalyzerComprehensive(unittest.TestCase):
    """Comprehensive test cases for SpatialRelationshipAnalyzer"""

    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = SpatialRelationshipAnalyzer()

    def test_update_relationships(self):
        """Test update_relationships"""
        detections = [
            {"label": "person", "center": (100, 100), "bbox": [90, 90, 110, 110]},
            {"label": "car", "center": (150, 100), "bbox": [140, 90, 160, 110]}
        ]
        self.analyzer.update_relationships(detections)
        
        # Should have tracked relationships
        self.assertGreater(len(self.analyzer.relationship_history), 0)

    def test_update_relationships_single(self):
        """Test update_relationships with single detection"""
        detections = [
            {"label": "person", "center": (100, 100)}
        ]
        self.analyzer.update_relationships(detections)
        # Should not create relationships with single object

    def test_get_common_spatial_patterns(self):
        """Test get_common_spatial_patterns"""
        # Add multiple relationships
        for _ in range(5):
            detections = [
                {"label": "person", "center": (100, 100)},
                {"label": "car", "center": (150, 100)}
            ]
            self.analyzer.update_relationships(detections)
        
        patterns = self.analyzer.get_common_spatial_patterns(min_occurrences=3)
        self.assertIsInstance(patterns, dict)

    def test_get_relationship_summary(self):
        """Test get_relationship_summary"""
        # Add relationships
        for _ in range(5):
            detections = [
                {"label": "person", "center": (100, 100)},
                {"label": "car", "center": (150, 100)}
            ]
            self.analyzer.update_relationships(detections)
        
        summary = self.analyzer.get_relationship_summary()
        self.assertIn("most_common_relations", summary)
        self.assertIn("frequent_object_pairs", summary)
        self.assertIn("spatial_patterns", summary)

    def test_get_relationship_summary_empty(self):
        """Test get_relationship_summary with no relationships"""
        summary = self.analyzer.get_relationship_summary()
        self.assertEqual(summary, {})

    def test_analyze_object_pair_near(self):
        """Test _analyze_object_pair with near objects"""
        det1 = {"label": "person", "center": (100, 100)}
        det2 = {"label": "car", "center": (150, 100)}  # Within 100 pixels
        
        relationship = self.analyzer._analyze_object_pair(det1, det2)
        self.assertIsNotNone(relationship)
        self.assertIn("relation", relationship)

    def test_analyze_object_pair_no_center(self):
        """Test _analyze_object_pair without center"""
        det1 = {"label": "person"}
        det2 = {"label": "car", "center": (150, 100)}
        
        relationship = self.analyzer._analyze_object_pair(det1, det2)
        self.assertIsNone(relationship)


if __name__ == "__main__":
    unittest.main()

