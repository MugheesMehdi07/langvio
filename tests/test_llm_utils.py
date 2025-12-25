"""
Unit tests for LLM utility functions
"""

import unittest
from langvio.utils.llm_utils import (
    process_image_detections_and_format_summary,
    format_video_summary,
    analyze_frame_activity,
)


class TestLLMUtils(unittest.TestCase):
    """Test cases for LLM utility functions"""

    def test_process_image_detections_empty(self):
        """Test processing empty image detections"""
        detections = {"objects": []}
        query_params = {"task_type": "identification"}
        
        summary, detection_map = process_image_detections_and_format_summary(
            detections, query_params
        )
        
        self.assertIn("No objects detected", summary)
        self.assertEqual(detection_map, {})

    def test_process_image_detections_with_objects(self):
        """Test processing image detections with objects"""
        detections = {
            "objects": [
                {
                    "id": "obj_1",
                    "type": "person",
                    "confidence": 0.9,
                    "bbox": [10, 10, 50, 50]
                },
                {
                    "id": "obj_2",
                    "type": "car",
                    "confidence": 0.8,
                    "bbox": [60, 60, 100, 100]
                }
            ],
            "summary": {
                "image_info": {
                    "resolution": "1920x1080",
                    "total_objects": 2,
                    "unique_types": 2
                },
                "object_distribution": {
                    "by_type": {"person": 1, "car": 1}
                }
            }
        }
        query_params = {"task_type": "identification"}
        
        summary, detection_map = process_image_detections_and_format_summary(
            detections, query_params
        )
        
        self.assertIn("Image Analysis Summary", summary)
        self.assertIn("person", summary)
        self.assertIn("car", summary)
        self.assertEqual(len(detection_map), 2)
        self.assertIn("obj_1", detection_map)
        self.assertIn("obj_2", detection_map)

    def test_process_image_detections_with_attributes(self):
        """Test processing image detections with attributes"""
        detections = {
            "objects": [
                {
                    "id": "obj_1",
                    "type": "person",
                    "confidence": 0.9,
                    "color": "red",
                    "size": "large",
                    "position": "center"
                }
            ],
            "summary": {}
        }
        query_params = {"task_type": "identification"}
        
        summary, detection_map = process_image_detections_and_format_summary(
            detections, query_params
        )
        
        self.assertIn("color:red", summary)
        self.assertIn("size:large", summary)

    def test_format_video_summary(self):
        """Test formatting video summary"""
        video_results = {
            "summary": {
                "video_info": {
                    "duration_seconds": 10.0,
                    "resolution": "1920x1080",
                    "fps": 30.0,
                    "activity_level": "medium",
                    "primary_objects": ["person", "car"]
                },
                "counting_analysis": {
                    "total_crossings": 5,
                    "flow_direction": "in",
                    "net_flow": 3,
                    "objects_entered": 5,
                    "objects_exited": 2
                }
            },
            "frame_detections": {
                "0": [{"type": "person"}],
                "10": [{"type": "person"}]
            },
            "processing_info": {
                "total_frames": 300
            }
        }
        parsed_query = {"task_type": "counting"}
        
        formatted = format_video_summary(video_results, parsed_query)
        
        self.assertIsInstance(formatted, str)
        self.assertIn("VIDEO ANALYSIS", formatted)
        self.assertIn("person", formatted)

    def test_analyze_frame_activity(self):
        """Test analyzing frame activity"""
        frame_detections = {
            "0": [{"type": "person"}, {"type": "car"}],
            "10": [{"type": "person"}],
            "20": [{"type": "person"}, {"type": "person"}, {"type": "car"}]
        }
        
        analysis = analyze_frame_activity(frame_detections)
        
        self.assertIn("peak_frame", analysis)
        self.assertIn("peak_count", analysis)
        self.assertIn("avg_objects", analysis)
        self.assertGreaterEqual(analysis["peak_count"], 2)


if __name__ == "__main__":
    unittest.main()

