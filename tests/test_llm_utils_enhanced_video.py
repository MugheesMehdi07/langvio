"""
Unit tests for format_enhanced_video_summary function
"""

import unittest
from langvio.utils.llm_utils import format_enhanced_video_summary


class TestFormatEnhancedVideoSummary(unittest.TestCase):
    """Test cases for format_enhanced_video_summary"""

    def test_format_enhanced_video_summary_minimal(self):
        """Test format_enhanced_video_summary with minimal data"""
        video_results = {
            "summary": {},
            "processing_info": {}
        }
        parsed_query = {}
        
        result = format_enhanced_video_summary(video_results, parsed_query)
        self.assertIsInstance(result, str)

    def test_format_enhanced_video_summary_with_video_info(self):
        """Test format_enhanced_video_summary with video info"""
        video_results = {
            "summary": {
                "video_info": {
                    "duration_seconds": 10.0,
                    "resolution": "1920x1080",
                    "activity_level": "high",
                    "primary_objects": ["person", "car"]
                }
            },
            "processing_info": {}
        }
        parsed_query = {}
        
        result = format_enhanced_video_summary(video_results, parsed_query)
        self.assertIn("Duration", result)
        self.assertIn("person", result)

    def test_format_enhanced_video_summary_with_counting(self):
        """Test format_enhanced_video_summary with counting analysis"""
        video_results = {
            "summary": {
                "counting_analysis": {
                    "objects_entered": 10,
                    "objects_exited": 5,
                    "net_flow": 5,
                    "total_crossings": 15,
                    "by_object_type": {
                        "person": {"entered": 7, "exited": 3, "net_flow": 4},
                        "car": {"entered": 3, "exited": 2, "net_flow": 1}
                    },
                    "most_active_type": "person"
                }
            },
            "processing_info": {}
        }
        parsed_query = {}
        
        result = format_enhanced_video_summary(video_results, parsed_query)
        self.assertIn("Entered", result)
        self.assertIn("person", result.lower())

    def test_format_enhanced_video_summary_with_speed(self):
        """Test format_enhanced_video_summary with speed analysis"""
        video_results = {
            "summary": {
                "speed_analysis": {
                    "speed_available": True,
                    "objects_with_speed": 5,
                    "average_speed_kmh": 20.5,
                    "speed_category": "moderate",
                    "by_object_type": {
                        "person": {"average_speed": 5.0, "speed_category": "slow"},
                        "car": {"average_speed": 30.0, "speed_category": "fast"}
                    },
                    "fastest_type": "car"
                }
            },
            "processing_info": {}
        }
        parsed_query = {}
        
        result = format_enhanced_video_summary(video_results, parsed_query)
        self.assertIn("Speed", result)

    def test_format_enhanced_video_summary_with_temporal(self):
        """Test format_enhanced_video_summary with temporal relationships"""
        video_results = {
            "summary": {
                "temporal_relationships": {
                    "movement_patterns": {
                        "stationary_count": 1,
                        "moving_count": 3,
                        "fast_moving_count": 1,
                        "primary_directions": {
                            "right": ["obj_1", "obj_2"],
                            "left": ["obj_3"]
                        }
                    },
                    "co_occurrence_events": 5,
                    "interaction_summary": [
                        {"object1": "person", "object2": "car", "relationship": "near"}
                    ]
                }
            },
            "processing_info": {}
        }
        parsed_query = {}
        
        result = format_enhanced_video_summary(video_results, parsed_query)
        self.assertIn("Stationary", result)
        self.assertIn("Moving", result)

    def test_format_enhanced_video_summary_with_spatial(self):
        """Test format_enhanced_video_summary with spatial relationships"""
        video_results = {
            "summary": {
                "spatial_relationships": {
                    "common_relations": {"near": 10, "above": 5},
                    "frequent_pairs": {"person-car": 8}
                }
            },
            "processing_info": {}
        }
        parsed_query = {}
        
        result = format_enhanced_video_summary(video_results, parsed_query)
        self.assertIsInstance(result, str)

    def test_format_enhanced_video_summary_with_object_analysis(self):
        """Test format_enhanced_video_summary with object analysis"""
        video_results = {
            "summary": {
                "object_analysis": {
                    "object_characteristics": {
                        "person": {
                            "total_instances": 5,
                            "movement_behavior": "walking",
                            "common_attributes": {"size": "medium", "color": "red"}
                        }
                    }
                }
            },
            "processing_info": {}
        }
        parsed_query = {}
        
        result = format_enhanced_video_summary(video_results, parsed_query)
        self.assertIn("person", result.lower())

    def test_format_enhanced_video_summary_with_insights(self):
        """Test format_enhanced_video_summary with insights"""
        video_results = {
            "summary": {
                "primary_insights": [
                    "High activity detected",
                    "Multiple object types"
                ]
            },
            "processing_info": {}
        }
        parsed_query = {}
        
        result = format_enhanced_video_summary(video_results, parsed_query)
        self.assertIn("Insights", result)

    def test_format_enhanced_video_summary_with_processing_info(self):
        """Test format_enhanced_video_summary with processing info"""
        video_results = {
            "summary": {},
            "processing_info": {
                "frames_analyzed": 100,
                "total_frames": 300,
                "yolo_world_enabled": True
            }
        }
        parsed_query = {}
        
        result = format_enhanced_video_summary(video_results, parsed_query)
        self.assertIn("Coverage", result)


if __name__ == "__main__":
    unittest.main()

