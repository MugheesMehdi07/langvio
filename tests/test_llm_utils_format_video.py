"""
Unit tests for format_video_summary function
"""

import unittest
from langvio.utils.llm_utils import format_video_summary


class TestFormatVideoSummary(unittest.TestCase):
    """Test cases for format_video_summary"""

    def test_format_video_summary_minimal(self):
        """Test format_video_summary with minimal data"""
        video_results = {
            "summary": {},
            "frame_detections": {},
            "processing_info": {}
        }
        parsed_query = {}
        
        result = format_video_summary(video_results, parsed_query)
        self.assertIsInstance(result, str)

    def test_format_video_summary_with_video_info(self):
        """Test format_video_summary with video info"""
        video_results = {
            "summary": {
                "video_info": {
                    "duration_seconds": 10.0,
                    "resolution": "1920x1080",
                    "fps": 30.0,
                    "activity_level": "high",
                    "primary_objects": ["person", "car"]
                }
            },
            "frame_detections": {},
            "processing_info": {}
        }
        parsed_query = {}
        
        result = format_video_summary(video_results, parsed_query)
        self.assertIn("VIDEO", result)
        self.assertIn("person", result)

    def test_format_video_summary_with_counting(self):
        """Test format_video_summary with counting analysis"""
        video_results = {
            "summary": {
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
                }
            },
            "frame_detections": {},
            "processing_info": {}
        }
        parsed_query = {}
        
        result = format_video_summary(video_results, parsed_query)
        self.assertIn("crossings", result.lower())
        # Check for person in any case (may be uppercase in output)
        self.assertIn("PERSON", result.upper())

    def test_format_video_summary_with_speed(self):
        """Test format_video_summary with speed analysis"""
        video_results = {
            "summary": {
                "speed_analysis": {
                    "speed_available": True,
                    "objects_with_speed": 5,
                    "average_speed_kmh": 20.5,
                    "speed_category": "moderate",
                    "by_object_type": {
                        "person": {"average_speed": 5.0, "sample_count": 3, "speed_category": "slow"},
                        "car": {"average_speed": 30.0, "sample_count": 2, "speed_category": "fast"}
                    },
                    "fastest_type": "car"
                }
            },
            "frame_detections": {},
            "processing_info": {}
        }
        parsed_query = {}
        
        result = format_video_summary(video_results, parsed_query)
        self.assertIn("speed", result.lower())

    def test_format_video_summary_with_spatial(self):
        """Test format_video_summary with spatial relationships"""
        video_results = {
            "summary": {
                "spatial_relationships": {
                    "common_relations": {"near": 10, "above": 5},
                    "frequent_pairs": {"person-car": 8},
                    "spatial_patterns": {"left-right": 3}
                }
            },
            "frame_detections": {},
            "processing_info": {}
        }
        parsed_query = {}
        
        result = format_video_summary(video_results, parsed_query)
        self.assertIsInstance(result, str)

    def test_format_video_summary_with_object_analysis(self):
        """Test format_video_summary with object analysis"""
        video_results = {
            "summary": {
                "object_analysis": {
                    "object_characteristics": {"person": {"avg_size": "medium"}},
                    "most_common_types": ["person", "car"]
                }
            },
            "frame_detections": {},
            "processing_info": {}
        }
        parsed_query = {}
        
        result = format_video_summary(video_results, parsed_query)
        self.assertIsInstance(result, str)

    def test_format_video_summary_with_temporal(self):
        """Test format_video_summary with temporal relationships"""
        video_results = {
            "summary": {
                "temporal_relationships": {
                    "movement_patterns": {
                        "moving_count": 3,
                        "stationary_count": 1
                    }
                }
            },
            "frame_detections": {},
            "processing_info": {}
        }
        parsed_query = {}
        
        result = format_video_summary(video_results, parsed_query)
        self.assertIsInstance(result, str)

    def test_format_video_summary_with_frame_analysis(self):
        """Test format_video_summary with frame analysis"""
        video_results = {
            "summary": {
                "video_info": {"fps": 30.0}
            },
            "frame_detections": {
                "0": [{"label": "person"}],
                "10": [{"label": "person"}, {"label": "car"}],
                "20": [{"label": "person"}]
            },
            "processing_info": {"total_frames": 300}
        }
        parsed_query = {}
        
        result = format_video_summary(video_results, parsed_query)
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()

