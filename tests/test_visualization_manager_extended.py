"""
Extended unit tests for visualization manager
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from langvio.config import Config
from langvio.core.visualization_manager import VisualizationManager


class TestVisualizationManagerExtended(unittest.TestCase):
    """Extended test cases for VisualizationManager"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = Config()
        self.manager = VisualizationManager(self.config)

    def test_prepare_overlay_information(self):
        """Test _prepare_overlay_information"""
        summary = {
            "video_info": {
                "primary_objects": ["person", "car"]
            },
            "counting_analysis": {
                "total_crossings": 5,
                "net_flow": 3
            },
            "speed_analysis": {
                "speed_available": True,
                "average_speed_kmh": 15.0
            }
        }
        overlay = self.manager._prepare_overlay_information(summary)
        self.assertIn("lines", overlay)
        self.assertIn("stats", overlay)

    def test_prepare_overlay_information_empty(self):
        """Test _prepare_overlay_information with empty summary"""
        overlay = self.manager._prepare_overlay_information({})
        self.assertIsInstance(overlay, dict)

    def test_draw_detections_on_frame(self):
        """Test _draw_detections_on_frame"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = [
            {"bbox": [10, 10, 50, 50], "label": "person", "confidence": 0.9}
        ]
        highlighted = set()
        config = {"box_color": [0, 255, 0], "text_color": [255, 255, 255], "line_thickness": 2}
        
        result = self.manager._draw_detections_on_frame(frame, detections, highlighted, config)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, frame.shape)

    def test_draw_detections_on_frame_highlighted(self):
        """Test _draw_detections_on_frame with highlighted objects"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = [
            {"bbox": [10, 10, 50, 50], "label": "person", "object_id": "obj_1"}
        ]
        highlighted = {"obj_1"}
        config = {"box_color": [0, 255, 0], "text_color": [255, 255, 255], "line_thickness": 2}
        
        result = self.manager._draw_detections_on_frame(frame, detections, highlighted, config)
        self.assertIsNotNone(result)

    def test_add_comprehensive_overlay(self):
        """Test _add_comprehensive_overlay"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        overlay_info = {
            "lines": ["Objects: person, car"],
            "stats": {"total": 5},
            "insights": []
        }
        
        result = self.manager._add_comprehensive_overlay(frame, overlay_info, 0, 30.0)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, frame.shape)

    def test_add_comprehensive_overlay_empty(self):
        """Test _add_comprehensive_overlay with empty overlay"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        overlay_info = {"lines": [], "stats": {}, "insights": []}
        
        result = self.manager._add_comprehensive_overlay(frame, overlay_info, 0, 30.0)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()

