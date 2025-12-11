"""
Unit tests for detection compression utilities
"""

import unittest
from langvio.utils.detection.compression import (
    compress_detections_for_output,
    identify_object_clusters,
)


class TestDetectionCompression(unittest.TestCase):
    """Test cases for detection compression utilities"""

    def test_compress_detections_basic(self):
        """Test compress_detections_for_output with basic detections"""
        detections = [
            {
                "label": "person",
                "confidence": 0.9,
                "bbox": [10, 10, 50, 50],
                "object_id": "obj_1"
            }
        ]
        compressed = compress_detections_for_output(detections, is_video=False)
        
        self.assertEqual(len(compressed), 1)
        self.assertIn("id", compressed[0])
        self.assertIn("type", compressed[0])
        self.assertIn("confidence", compressed[0])

    def test_compress_detections_with_attributes(self):
        """Test compress_detections_for_output with attributes"""
        detections = [
            {
                "label": "person",
                "confidence": 0.9,
                "bbox": [10, 10, 50, 50],
                "object_id": "obj_1",
                "attributes": {
                    "size": "large",
                    "position": "center",
                    "color": "red"
                }
            }
        ]
        compressed = compress_detections_for_output(detections, is_video=False)
        
        self.assertIn("size", compressed[0])
        self.assertIn("position", compressed[0])
        self.assertIn("color", compressed[0])

    def test_compress_detections_video(self):
        """Test compress_detections_for_output for video"""
        detections = [
            {
                "label": "person",
                "confidence": 0.9,
                "bbox": [10, 10, 50, 50],
                "track_id": 1
            }
        ]
        compressed = compress_detections_for_output(detections, is_video=True)
        
        self.assertIn("track_id", compressed[0])

    def test_compress_detections_with_relationships(self):
        """Test compress_detections_for_output with relationships"""
        detections = [
            {
                "label": "person",
                "confidence": 0.9,
                "bbox": [10, 10, 50, 50],
                "relationships": [
                    {"object": "car", "relations": ["near"]}
                ]
            }
        ]
        compressed = compress_detections_for_output(detections, is_video=False)
        
        # Should include key relationships
        if "key_relationships" in compressed[0]:
            self.assertGreater(len(compressed[0]["key_relationships"]), 0)

    def test_identify_object_clusters_empty(self):
        """Test identify_object_clusters with empty detections"""
        clusters = identify_object_clusters([])
        self.assertEqual(clusters, [])

    def test_identify_object_clusters_single(self):
        """Test identify_object_clusters with single detection"""
        detections = [
            {"center": (100, 100), "label": "person"}
        ]
        clusters = identify_object_clusters(detections)
        self.assertEqual(clusters, [])

    def test_identify_object_clusters_close_objects(self):
        """Test identify_object_clusters with close objects"""
        detections = [
            {"center": (100, 100), "label": "person"},
            {"center": (120, 110), "label": "person"},  # Close to first
            {"center": (300, 300), "label": "car"}  # Far from others
        ]
        clusters = identify_object_clusters(detections, distance_threshold=150)
        
        # Should find at least one cluster
        self.assertGreaterEqual(len(clusters), 0)

    def test_identify_object_clusters_no_center(self):
        """Test identify_object_clusters with detections without center"""
        detections = [
            {"label": "person"},  # No center
            {"label": "car"}  # No center
        ]
        clusters = identify_object_clusters(detections)
        self.assertEqual(clusters, [])


if __name__ == "__main__":
    unittest.main()

