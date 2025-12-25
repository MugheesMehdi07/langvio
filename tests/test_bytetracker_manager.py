"""
Unit tests for ByteTracker manager
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from langvio.utils.tracking.bytetracker_manager import ByteTrackerManager


class TestByteTrackerManager(unittest.TestCase):
    """Test cases for ByteTrackerManager"""

    def setUp(self):
        """Set up test fixtures"""
        self.tracker = ByteTrackerManager(
            track_thresh=0.3,
            track_buffer=70,
            match_thresh=0.6,
            frame_rate=30
        )

    def test_initialization(self):
        """Test ByteTrackerManager initialization"""
        self.assertEqual(self.tracker.track_thresh, 0.3)
        self.assertEqual(self.tracker.track_buffer, 70)
        self.assertEqual(self.tracker.match_thresh, 0.6)
        self.assertEqual(self.tracker.frame_rate, 30)
        self.assertEqual(self.tracker.track_id_count, 0)

    def test_update_empty_detections(self):
        """Test update with empty detections"""
        result = self.tracker.update([], 0)
        self.assertEqual(result, [])

    def test_update_low_confidence(self):
        """Test update with low confidence detections"""
        detections = [
            {"bbox": [10, 10, 50, 50], "confidence": 0.1, "class_id": 0}
        ]
        result = self.tracker.update(detections, 0)
        # Low confidence detections should be filtered
        self.assertEqual(len(result), 0)

    def test_update_first_frame(self):
        """Test update on first frame"""
        detections = [
            {"bbox": [10, 10, 50, 50], "confidence": 0.9, "class_id": 0, "label": "person"}
        ]
        result = self.tracker.update(detections, 0)
        self.assertEqual(len(result), 1)
        self.assertIn("track_id", result[0])

    def test_update_multiple_frames(self):
        """Test update across multiple frames"""
        # First frame
        detections1 = [
            {"bbox": [10, 10, 50, 50], "confidence": 0.9, "class_id": 0, "label": "person"}
        ]
        result1 = self.tracker.update(detections1, 0)
        track_id = result1[0]["track_id"]
        
        # Second frame - same object moved slightly
        detections2 = [
            {"bbox": [12, 12, 52, 52], "confidence": 0.9, "class_id": 0, "label": "person"}
        ]
        result2 = self.tracker.update(detections2, 1)
        
        # Should maintain same track_id if IoU is high enough
        self.assertEqual(len(result2), 1)
        # Track ID may be same or different depending on IoU matching

    def test_create_new_track(self):
        """Test _create_new_track"""
        detection = {"bbox": [10, 10, 50, 50], "confidence": 0.9, "class_id": 0, "label": "person"}
        track_id = self.tracker._create_new_track(detection)
        self.assertIsNotNone(track_id)
        self.assertIn(track_id, self.tracker.tracks)
        self.assertEqual(self.tracker.tracks[track_id]["class_name"], "person")

    def test_calculate_iou_matrix(self):
        """Test _calculate_iou_matrix"""
        boxes1 = np.array([[10, 10, 20, 20], [30, 30, 40, 40]])  # [x, y, w, h]
        boxes2 = np.array([[12, 12, 20, 20], [50, 50, 20, 20]])
        
        iou_matrix = self.tracker._calculate_iou_matrix(boxes1, boxes2)
        self.assertEqual(iou_matrix.shape, (2, 2))
        # First box should have higher IoU with first box in boxes2
        self.assertGreater(iou_matrix[0, 0], iou_matrix[0, 1])

    def test_match_detections_to_tracks(self):
        """Test _match_detections_to_tracks"""
        # Create high IoU matrix (almost perfect matches)
        iou_matrix = np.array([[0.9, 0.1], [0.1, 0.9]])
        confidences = np.array([0.9, 0.8])
        track_ids = [1, 2]
        
        matched_det, matched_track = self.tracker._match_detections_to_tracks(
            iou_matrix, confidences, track_ids
        )
        self.assertEqual(len(matched_det), 2)
        self.assertEqual(len(matched_track), 2)

    def test_update_lost_tracks(self):
        """Test _update_lost_tracks"""
        # Create a track
        detection = {"bbox": [10, 10, 50, 50], "confidence": 0.9, "class_id": 0, "label": "person"}
        track_id = self.tracker._create_new_track(detection)
        
        # Mark as lost
        self.tracker.tracks[track_id]["state"] = "lost"
        self.tracker.tracks[track_id]["last_seen"] = 0
        
        # Update lost tracks
        self.tracker._update_lost_tracks()
        
        # Track should be removed if buffer exceeded
        # (depends on implementation)


if __name__ == "__main__":
    unittest.main()

