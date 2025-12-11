"""
Comprehensive unit tests for visualization manager
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from langvio.config import Config
from langvio.core.visualization_manager import VisualizationManager


class TestVisualizationManagerComprehensive(unittest.TestCase):
    """Comprehensive test cases for VisualizationManager"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = Config()
        self.manager = VisualizationManager(self.config)

    def test_prepare_overlay_information_comprehensive(self):
        """Test _prepare_overlay_information with comprehensive data"""
        summary = {
            "video_info": {
                "primary_objects": ["person", "car", "bicycle"]
            },
            "counting_analysis": {
                "total_crossings": 10,
                "net_flow": 5,
                "flow_direction": "in"
            },
            "speed_analysis": {
                "speed_available": True,
                "average_speed_kmh": 20.5
            },
            "temporal_relationships": {
                "movement_patterns": {
                    "moving_count": 3,
                    "stationary_count": 1
                }
            },
            "primary_insights": [
                "High activity detected",
                "Multiple object types"
            ]
        }
        
        overlay = self.manager._prepare_overlay_information(summary)
        self.assertIn("lines", overlay)
        self.assertIn("insights", overlay)
        self.assertGreater(len(overlay["lines"]), 0)

    def test_draw_detections_on_frame_multiple(self):
        """Test _draw_detections_on_frame with multiple detections"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = [
            {"bbox": [10, 10, 50, 50], "label": "person", "confidence": 0.9, "object_id": "obj_1"},
            {"bbox": [60, 60, 90, 90], "label": "car", "confidence": 0.8, "object_id": "obj_2"}
        ]
        highlighted = {"obj_1"}
        config = {"box_color": [0, 255, 0], "text_color": [255, 255, 255], "line_thickness": 2}
        
        result = self.manager._draw_detections_on_frame(frame, detections, highlighted, config)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, frame.shape)

    def test_draw_detections_on_frame_no_bbox(self):
        """Test _draw_detections_on_frame with detection without bbox"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = [
            {"label": "person", "confidence": 0.9}  # No bbox
        ]
        highlighted = set()
        config = {"box_color": [0, 255, 0], "text_color": [255, 255, 255], "line_thickness": 2}
        
        result = self.manager._draw_detections_on_frame(frame, detections, highlighted, config)
        # Should handle gracefully
        self.assertIsNotNone(result)

    def test_draw_detections_on_frame_with_attributes(self):
        """Test _draw_detections_on_frame with attributes"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = [
            {
                "bbox": [10, 10, 50, 50],
                "label": "person",
                "confidence": 0.9,
                "attributes": {"color": "red", "size": "large"}
            }
        ]
        highlighted = set()
        config = {"box_color": [0, 255, 0], "text_color": [255, 255, 255], "line_thickness": 2}
        
        result = self.manager._draw_detections_on_frame(frame, detections, highlighted, config)
        self.assertIsNotNone(result)

    def test_add_comprehensive_overlay_with_insights(self):
        """Test _add_comprehensive_overlay with insights"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        overlay_info = {
            "lines": ["Objects: person, car", "Crossings: 5"],
            "stats": {"total": 5},
            "insights": ["High activity", "Multiple types"]
        }
        
        result = self.manager._add_comprehensive_overlay(frame, overlay_info, 10, 30.0)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, frame.shape)

    def test_add_comprehensive_overlay_frame_info(self):
        """Test _add_comprehensive_overlay adds frame information"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        overlay_info = {
            "lines": [],
            "stats": {},
            "insights": []
        }
        
        result = self.manager._add_comprehensive_overlay(frame, overlay_info, 100, 30.0)
        # Should add frame/time info
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()

