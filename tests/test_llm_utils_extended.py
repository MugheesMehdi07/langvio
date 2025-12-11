"""
Extended unit tests for LLM utility functions
"""

import unittest
from langvio.utils.llm_utils import (
    process_image_detections_and_format_summary,
    format_video_summary,
    analyze_frame_activity,
    create_frame_summary_for_llm,
    extract_object_ids,
    get_objects_by_ids,
    parse_explanation_response,
    format_enhanced_video_summary,
    create_video_detection_map_for_highlighting,
)


class TestLLMUtilsExtended(unittest.TestCase):
    """Extended test cases for LLM utility functions"""

    def test_create_frame_summary_for_llm_empty(self):
        """Test create_frame_summary_for_llm with empty detections"""
        result = create_frame_summary_for_llm({})
        self.assertIn("No frame data available", result)

    def test_create_frame_summary_for_llm_with_frames(self):
        """Test create_frame_summary_for_llm with frame detections"""
        frame_detections = {
            "0": [{"label": "person", "confidence": 0.9}],
            "10": [{"label": "car", "confidence": 0.8}],
            "20": [{"label": "person", "confidence": 0.9}, {"label": "car", "confidence": 0.7}]
        }
        result = create_frame_summary_for_llm(frame_detections)
        self.assertIn("DETAILED FRAME ANALYSIS", result)
        self.assertIn("person", result)
        self.assertIn("car", result)

    def test_create_frame_summary_with_attributes(self):
        """Test create_frame_summary_for_llm with attributes"""
        frame_detections = {
            "0": [{"label": "person", "attributes": {"color": "red"}}]
        }
        result = create_frame_summary_for_llm(frame_detections)
        self.assertIn("red", result)

    def test_extract_object_ids_simple(self):
        """Test extracting object IDs from simple text"""
        text = "Highlight objects obj_1, obj_2, and obj_3"
        ids = extract_object_ids(text)
        self.assertIn("obj_1", ids)
        self.assertIn("obj_2", ids)

    def test_extract_object_ids_brackets(self):
        """Test extracting object IDs from bracketed text"""
        text = "Objects [obj_1] and [obj_2] are highlighted"
        ids = extract_object_ids(text)
        self.assertIn("obj_1", ids)
        self.assertIn("obj_2", ids)

    def test_extract_object_ids_empty(self):
        """Test extracting object IDs from empty text"""
        ids = extract_object_ids("")
        self.assertEqual(ids, [])

    def test_get_objects_by_ids(self):
        """Test getting objects by IDs"""
        detection_map = {
            "obj_1": {"frame_key": "0", "detection": {"label": "person"}},
            "obj_2": {"frame_key": "0", "detection": {"label": "car"}}
        }
        ids = ["obj_1", "obj_2"]
        objects = get_objects_by_ids(ids, detection_map)
        self.assertEqual(len(objects), 2)

    def test_get_objects_by_ids_missing(self):
        """Test getting objects by IDs with missing IDs"""
        detection_map = {
            "obj_1": {"frame_key": "0", "detection": {"label": "person"}}
        }
        ids = ["obj_1", "obj_nonexistent"]
        objects = get_objects_by_ids(ids, detection_map)
        self.assertEqual(len(objects), 1)

    def test_parse_explanation_response_simple(self):
        """Test parsing simple explanation response"""
        response = "Found 3 people in the video."
        detection_map = {}
        explanation, highlights = parse_explanation_response(response, detection_map)
        self.assertIsInstance(explanation, str)
        self.assertIn("people", explanation)

    def test_parse_explanation_response_with_ids(self):
        """Test parsing explanation response with object IDs"""
        response = "Found [obj_1] and [obj_2] in the video. HIGHLIGHT_OBJECTS: obj_1, obj_2"
        detection_map = {
            "obj_1": {"frame_key": "0", "detection": {"label": "person"}},
            "obj_2": {"frame_key": "0", "detection": {"label": "car"}}
        }
        explanation, highlights = parse_explanation_response(response, detection_map)
        self.assertIsInstance(explanation, str)
        self.assertEqual(len(highlights), 2)

    def test_format_enhanced_video_summary(self):
        """Test formatting enhanced video summary"""
        video_results = {
            "summary": {
                "video_info": {"duration_seconds": 10.0},
                "counting_analysis": {"total_crossings": 5}
            },
            "frame_detections": {"0": []},
            "processing_info": {"total_frames": 300}
        }
        parsed_query = {"task_type": "counting"}
        result = format_enhanced_video_summary(video_results, parsed_query)
        self.assertIsInstance(result, str)
        self.assertIn("Enhanced Video Analysis Summary", result)

    def test_create_video_detection_map_for_highlighting(self):
        """Test creating video detection map for highlighting"""
        video_results = {
            "frame_detections": {
                "0": [{"object_id": "obj_1", "label": "person"}],
                "10": [{"object_id": "obj_2", "label": "car"}]
            }
        }
        result = create_video_detection_map_for_highlighting(video_results)
        self.assertIsInstance(result, dict)
        # May be empty if no detections meet criteria

    def test_analyze_frame_activity_empty(self):
        """Test analyze_frame_activity with empty detections"""
        result = analyze_frame_activity({})
        self.assertEqual(result["peak_frame"], 0)
        self.assertEqual(result["peak_count"], 0)
        self.assertEqual(result["avg_objects"], 0.0)

    def test_analyze_frame_activity_with_peak(self):
        """Test analyze_frame_activity finding peak frame"""
        frame_detections = {
            "0": [{"label": "person"}],
            "10": [{"label": "person"}, {"label": "car"}, {"label": "person"}],  # Peak
            "20": [{"label": "car"}]
        }
        result = analyze_frame_activity(frame_detections)
        self.assertEqual(result["peak_frame"], 10)
        self.assertEqual(result["peak_count"], 3)
        self.assertGreater(result["avg_objects"], 0)

    def test_format_video_summary_comprehensive(self):
        """Test format_video_summary with comprehensive data"""
        video_results = {
            "summary": {
                "video_info": {
                    "duration_seconds": 10.0,
                    "resolution": "1920x1080",
                    "fps": 30.0,
                    "activity_level": "high",
                    "primary_objects": ["person", "car"]
                },
                "counting_analysis": {
                    "total_crossings": 10,
                    "flow_direction": "in",
                    "net_flow": 5,
                    "objects_entered": 10,
                    "objects_exited": 5,
                    "by_object_type": {
                        "person": {"entered": 7, "exited": 3, "net_flow": 4, "dominance": "in"},
                        "car": {"entered": 3, "exited": 2, "net_flow": 1, "dominance": "in"}
                    },
                    "most_active_type": "person"
                },
                "speed_analysis": {
                    "speed_available": True,
                    "objects_with_speed": 5,
                    "average_speed_kmh": 15.5,
                    "speed_category": "moderate",
                    "by_object_type": {
                        "person": {"average_speed": 5.0, "sample_count": 3, "speed_category": "slow"},
                        "car": {"average_speed": 30.0, "sample_count": 2, "speed_category": "fast"}
                    }
                }
            },
            "frame_detections": {
                "0": [{"label": "person"}],
                "10": [{"label": "person"}, {"label": "car"}],
                "20": [{"label": "person"}]
            },
            "processing_info": {
                "total_frames": 300,
                "frames_analyzed": 3,
                "yolo_world_enabled": True,
                "analysis_type": "comprehensive"
            }
        }
        parsed_query = {
            "task_type": "counting",
            "target_objects": ["person"],
            "attributes": [{"attribute": "color", "value": "red"}]
        }
        result = format_video_summary(video_results, parsed_query)
        self.assertIn("VIDEO ANALYSIS", result)
        self.assertIn("person", result)
        self.assertIn("crossings", result.lower())


if __name__ == "__main__":
    unittest.main()

