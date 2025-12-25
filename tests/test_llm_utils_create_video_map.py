"""
Unit tests for create_video_detection_map_for_highlighting function
"""

import unittest
from langvio.utils.llm_utils import create_video_detection_map_for_highlighting


class TestCreateVideoDetectionMap(unittest.TestCase):
    """Test cases for create_video_detection_map_for_highlighting"""

    def test_create_video_detection_map_empty(self):
        """Test create_video_detection_map_for_highlighting with empty data"""
        frame_detections = {}
        result = create_video_detection_map_for_highlighting(frame_detections)
        self.assertEqual(result, {})

    def test_create_video_detection_map_with_detections(self):
        """Test create_video_detection_map_for_highlighting with detections"""
        video_results = {
            "frame_detections": {
                "0": [
                    {
                        "object_id": "obj_1",
                        "label": "person",
                        "bbox": [10, 10, 50, 50],
                        "confidence": 0.9
                    }
                ],
                "10": [
                    {
                        "object_id": "obj_2",
                        "label": "car",
                        "bbox": [60, 60, 90, 90],
                        "confidence": 0.8
                    }
                ]
            }
        }
        result = create_video_detection_map_for_highlighting(video_results)
        
        # Should map from latest frame (10)
        self.assertIn("obj_2", result)
        self.assertEqual(result["obj_2"]["frame_key"], "10")

    def test_create_video_detection_map_with_object_id(self):
        """Test create_video_detection_map_for_highlighting with object_id field"""
        video_results = {
            "frame_detections": {
                "0": [
                    {
                        "object_id": "obj_1",
                        "label": "person",
                        "bbox": [10, 10, 50, 50]
                    }
                ]
            }
        }
        result = create_video_detection_map_for_highlighting(video_results)
        
        self.assertIn("obj_1", result)

    def test_create_video_detection_map_multiple_frames(self):
        """Test create_video_detection_map_for_highlighting with multiple frames"""
        video_results = {
            "frame_detections": {
                "0": [
                    {"object_id": "obj_1", "label": "person"},
                    {"object_id": "obj_2", "label": "car"}
                ],
                "10": [
                    {"object_id": "obj_1", "label": "person"}  # Same object in different frame
                ]
            }
        }
        result = create_video_detection_map_for_highlighting(video_results)
        
        # Should map from latest frame (10)
        self.assertIn("obj_1", result)
        self.assertEqual(result["obj_1"]["frame_key"], "10")


if __name__ == "__main__":
    unittest.main()

