"""
Comprehensive unit tests for LLM utility functions
"""

import unittest
from langvio.utils.llm_utils import (
    extract_object_ids,
    get_objects_by_ids,
    parse_explanation_response,
    create_frame_summary_for_llm,
    analyze_frame_activity,
)


class TestLLMUtilsComprehensive(unittest.TestCase):
    """Comprehensive test cases for LLM utility functions"""

    def test_extract_object_ids_various_formats(self):
        """Test extracting object IDs from various text formats"""
        # Test with brackets
        text1 = "Objects [obj_1] and [obj_2] are visible"
        ids1 = extract_object_ids(text1)
        self.assertIn("obj_1", ids1)
        self.assertIn("obj_2", ids1)
        
        # Test with commas
        text2 = "Highlight obj_1, obj_2, obj_3"
        ids2 = extract_object_ids(text2)
        self.assertGreaterEqual(len(ids2), 2)
        
        # Test with "and"
        text3 = "Objects obj_1 and obj_2"
        ids3 = extract_object_ids(text3)
        self.assertGreaterEqual(len(ids3), 1)

    def test_get_objects_by_ids_comprehensive(self):
        """Test get_objects_by_ids comprehensively"""
        detection_map = {
            "obj_1": {"frame_key": "0", "detection": {"label": "person", "id": "obj_1"}},
            "obj_2": {"frame_key": "0", "detection": {"label": "car", "id": "obj_2"}},
            "obj_3": {"frame_key": "10", "detection": {"label": "person", "id": "obj_3"}}
        }
        
        ids = ["obj_1", "obj_2", "obj_nonexistent"]
        objects = get_objects_by_ids(ids, detection_map)
        
        self.assertEqual(len(objects), 2)
        self.assertEqual(objects[0]["frame_key"], "0")
        self.assertIn("detection", objects[0])

    def test_parse_explanation_response_with_highlight_section(self):
        """Test parse_explanation_response with highlight section"""
        response = "EXPLANATION: Found 3 people. HIGHLIGHT_OBJECTS: obj_1, obj_2"
        detection_map = {
            "obj_1": {"frame_key": "0", "detection": {"label": "person"}},
            "obj_2": {"frame_key": "0", "detection": {"label": "person"}}
        }
        
        explanation, highlights = parse_explanation_response(response, detection_map)
        self.assertIn("Found", explanation)
        self.assertEqual(len(highlights), 2)

    def test_parse_explanation_response_without_highlight(self):
        """Test parse_explanation_response without highlight section"""
        response = "Found 3 people in the video."
        detection_map = {}
        
        explanation, highlights = parse_explanation_response(response, detection_map)
        self.assertIn("Found", explanation)
        self.assertEqual(len(highlights), 0)

    def test_parse_explanation_response_explanation_prefix(self):
        """Test parse_explanation_response with EXPLANATION: prefix"""
        response = "EXPLANATION: Found 3 people."
        detection_map = {}
        
        explanation, highlights = parse_explanation_response(response, detection_map)
        self.assertIn("Found", explanation)
        self.assertNotIn("EXPLANATION:", explanation)

    def test_create_frame_summary_max_frames(self):
        """Test create_frame_summary_for_llm with max_frames limit"""
        frame_detections = {}
        for i in range(50):
            frame_detections[str(i)] = [{"label": "person"}]
        
        result = create_frame_summary_for_llm(frame_detections, max_frames=20)
        # Should limit to max_frames
        self.assertIn("DETAILED FRAME ANALYSIS", result)

    def test_create_frame_summary_with_special_attributes(self):
        """Test create_frame_summary_for_llm with special attributes"""
        frame_detections = {
            "0": [
                {"label": "person", "attributes": {"color": "red"}},
                {"label": "car", "attributes": {"color": "blue"}}
            ]
        }
        
        result = create_frame_summary_for_llm(frame_detections)
        self.assertIn("red", result)
        self.assertIn("blue", result)

    def test_analyze_frame_activity_comprehensive(self):
        """Test analyze_frame_activity comprehensively"""
        frame_detections = {
            "0": [{"label": "person"}],
            "10": [{"label": "person"}, {"label": "car"}],
            "20": [{"label": "person"}, {"label": "person"}, {"label": "car"}],  # Peak
            "30": [{"label": "car"}]
        }
        
        result = analyze_frame_activity(frame_detections)
        
        self.assertEqual(result["peak_frame"], 20)
        self.assertEqual(result["peak_count"], 3)
        self.assertGreater(result["avg_objects"], 0)
        self.assertIn("activity_timeline", result)
        self.assertIn("total_frames_with_activity", result)

    def test_analyze_frame_activity_single_frame(self):
        """Test analyze_frame_activity with single frame"""
        frame_detections = {
            "0": [{"label": "person"}]
        }
        
        result = analyze_frame_activity(frame_detections)
        self.assertEqual(result["peak_frame"], 0)
        self.assertEqual(result["peak_count"], 1)
        self.assertEqual(result["avg_objects"], 1.0)


if __name__ == "__main__":
    unittest.main()

