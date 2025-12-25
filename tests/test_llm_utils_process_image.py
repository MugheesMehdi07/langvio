"""
Unit tests for process_image_detections_and_format_summary function
"""

import unittest
from langvio.utils.llm_utils import process_image_detections_and_format_summary


class TestProcessImageDetections(unittest.TestCase):
    """Test cases for process_image_detections_and_format_summary"""

    def test_process_image_detections_minimal(self):
        """Test process_image_detections_and_format_summary with minimal data"""
        detections = {"objects": [], "summary": {}}
        query_params = {}
        
        summary, detection_map = process_image_detections_and_format_summary(
            detections, query_params
        )
        self.assertIsInstance(summary, str)
        self.assertIsInstance(detection_map, dict)

    def test_process_image_detections_with_objects(self):
        """Test process_image_detections_and_format_summary with objects"""
        detections = {
            "objects": [
                {
                    "id": "obj_1",
                    "type": "person",
                    "label": "person",
                    "confidence": 0.9,
                    "bbox": [10, 10, 50, 50],
                    "size": "medium",
                    "position": "center"
                }
            ],
            "summary": {}
        }
        query_params = {"task_type": "identification"}
        
        summary, detection_map = process_image_detections_and_format_summary(
            detections, query_params
        )
        self.assertIn("person", summary.lower())
        self.assertIn("obj_1", detection_map)

    def test_process_image_detections_with_attributes(self):
        """Test process_image_detections_and_format_summary with attributes"""
        detections = {
            "objects": [
                {
                    "id": "obj_1",
                    "type": "car",
                    "label": "car",
                    "confidence": 0.8,
                    "bbox": [10, 10, 50, 50],
                    "color": "red",
                    "size": "large"
                }
            ],
            "summary": {}
        }
        query_params = {"task_type": "identification"}
        
        summary, detection_map = process_image_detections_and_format_summary(
            detections, query_params
        )
        self.assertIn("red", summary.lower())

    def test_process_image_detections_with_relationships(self):
        """Test process_image_detections_and_format_summary with relationships"""
        detections = {
            "objects": [
                {
                    "id": "obj_1",
                    "type": "person",
                    "label": "person",
                    "confidence": 0.9,
                    "bbox": [10, 10, 50, 50],
                    "key_relationships": [
                        {"to": "car", "relation": "near"}
                ]
                }
            ],
            "summary": {}
        }
        query_params = {"task_type": "identification"}
        
        summary, detection_map = process_image_detections_and_format_summary(
            detections, query_params
        )
        self.assertIsInstance(summary, str)


if __name__ == "__main__":
    unittest.main()

