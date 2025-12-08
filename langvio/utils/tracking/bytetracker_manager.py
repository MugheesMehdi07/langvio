"""
ByteTracker integration for multi-object tracking
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ByteTrackerManager:
    """ByteTracker implementation for multi-object tracking"""

    def __init__(
        self,
        track_thresh: float = 0.3,
        track_buffer: int = 70,
        match_thresh: float = 0.6,
        frame_rate: int = 30
    ):
        """
        Initialize ByteTracker

        Args:
            track_thresh: Detection confidence threshold for tracking
            track_buffer: Number of frames to keep lost tracks
            match_thresh: IoU threshold for track matching
            frame_rate: Video frame rate
        """
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate

        # Tracking state
        self.track_id_count = 0
        self.tracks = {}  # track_id -> track_data
        self.lost_tracks = {}  # track_id -> track_data
        self.frame_count = 0

        self.logger = logging.getLogger(__name__)

    def update(self, detections: List[Dict[str, Any]], frame_id: int) -> List[Dict[str, Any]]:
        """
        Update tracker with new detections

        Args:
            detections: List of detections from YOLO-World
            frame_id: Current frame ID

        Returns:
            List of detections with track IDs assigned
        """
        self.frame_count = frame_id

        # Filter detections by confidence
        valid_detections = [
            det for det in detections
            if det.get("confidence", 0) >= self.track_thresh
        ]

        if not valid_detections:
            # No valid detections, update lost tracks
            self._update_lost_tracks()
            return []

        # Convert detections to tracking format
        det_boxes = []
        det_confidences = []
        det_classes = []
        for det in valid_detections:
            bbox = det.get("bbox", [0, 0, 0, 0])
            # Convert to [x, y, w, h] format
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            det_boxes.append([x1, y1, w, h])
            det_confidences.append(det.get("confidence", 0))
            det_classes.append(det.get("class_id", 0))

        det_boxes = np.array(det_boxes)
        det_confidences = np.array(det_confidences)
        det_classes = np.array(det_classes)

        # Perform tracking
        tracked_detections = self._associate_detections_to_tracks(
            det_boxes, det_confidences, det_classes, valid_detections
        )

        # Update lost tracks
        self._update_lost_tracks()

        return tracked_detections

    def _associate_detections_to_tracks(
        self,
        det_boxes: np.ndarray,
        det_confidences: np.ndarray,
        det_classes: np.ndarray,
        original_detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Associate detections with existing tracks"""

        if len(self.tracks) == 0:
            # First frame or no existing tracks, create new tracks
            return self._create_new_tracks(original_detections)

        for track_id, track_data in self.tracks.items():
            if "prev_bbox" in track_data:
                old = np.array(track_data["prev_bbox"])
                new = np.array(track_data["bbox"])
                smoothed = 0.7 * old + 0.3 * new
                track_data["bbox"] = smoothed.tolist()
            track_data["prev_bbox"] = track_data["bbox"]

        # Calculate IoU between detections and existing tracks
        track_boxes = []
        track_ids = []

        for track_id, track_data in self.tracks.items():
            if track_data["state"] == "tracked":
                bbox = track_data["bbox"]
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1
                track_boxes.append([x1, y1, w, h])
                track_ids.append(track_id)

        if not track_boxes:
            return self._create_new_tracks(original_detections)

        track_boxes = np.array(track_boxes)

        # Calculate IoU matrix
        iou_matrix = self._calculate_iou_matrix(det_boxes, track_boxes)

        # Match detections to tracks
        matched_det_indices, matched_track_indices = self._match_detections_to_tracks(
            iou_matrix, det_confidences, track_ids
        )

        # Update matched tracks
        tracked_detections = []

        for det_idx, track_idx in zip(matched_det_indices, matched_track_indices):
            track_id = track_ids[track_idx]
            det = original_detections[det_idx].copy()

            # Update track data
            self.tracks[track_id]["bbox"] = det["bbox"]
            self.tracks[track_id]["confidence"] = det["confidence"]
            self.tracks[track_id]["class_id"] = det["class_id"]
            self.tracks[track_id]["state"] = "tracked"
            self.tracks[track_id]["last_seen"] = self.frame_count

            # Add track ID to detection
            det["track_id"] = track_id
            tracked_detections.append(det)

        # Create new tracks for unmatched detections
        unmatched_det_indices = [
            i for i in range(len(original_detections))
            if i not in matched_det_indices
        ]

        for det_idx in unmatched_det_indices:
            det = original_detections[det_idx].copy()
            track_id = self._create_new_track(det)
            det["track_id"] = track_id
            tracked_detections.append(det)

        return tracked_detections

    def _create_new_tracks(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create new tracks for all detections"""
        tracked_detections = []

        for det in detections:
            det_copy = det.copy()
            track_id = self._create_new_track(det)
            det_copy["track_id"] = track_id
            tracked_detections.append(det_copy)

        return tracked_detections

    def _create_new_track(self, detection: Dict[str, Any]) -> int:
        """Create a new track for a detection"""

        self.track_id_count += 1
        track_id = self.track_id_count

        self.tracks[track_id] = {
            "track_id": track_id,
            "bbox": detection["bbox"],
            "confidence": detection["confidence"],
            "class_id": detection["class_id"],
            "class_name": detection["label"],
            "state": "tracked",
            "first_seen": self.frame_count,
            "last_seen": self.frame_count,
            "total_frames": 1
        }

        return track_id

    def _calculate_iou_matrix(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """Calculate IoU matrix between two sets of boxes"""
        # Convert to [x1, y1, x2, y2] format
        boxes1_xyxy = np.column_stack([
            boxes1[:, 0], boxes1[:, 1],
            boxes1[:, 0] + boxes1[:, 2], boxes1[:, 1] + boxes1[:, 3]
        ])
        boxes2_xyxy = np.column_stack([
            boxes2[:, 0], boxes2[:, 1],
            boxes2[:, 0] + boxes2[:, 2], boxes2[:, 1] + boxes2[:, 3]
        ])

        # Calculate IoU
        iou_matrix = np.zeros((len(boxes1), len(boxes2)))

        for i, box1 in enumerate(boxes1_xyxy):
            for j, box2 in enumerate(boxes2_xyxy):
                iou_matrix[i, j] = self._calculate_iou(box1, box2)

        return iou_matrix

    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _match_detections_to_tracks(
        self,
        iou_matrix: np.ndarray,
        det_confidences: np.ndarray,
        track_ids: List[int]
    ) -> Tuple[List[int], List[int]]:
        """Match detections to tracks using IoU and confidence"""
        matched_det_indices = []
        matched_track_indices = []

        # Simple greedy matching based on IoU threshold
        for det_idx in range(len(det_confidences)):
            best_iou = 0
            best_track_idx = -1

            for track_idx in range(len(track_ids)):
                if track_idx in matched_track_indices:
                    continue

                iou = iou_matrix[det_idx, track_idx]
                if iou > self.match_thresh and iou > best_iou:
                    best_iou = iou
                    best_track_idx = track_idx

            if best_track_idx != -1:
                matched_det_indices.append(det_idx)
                matched_track_indices.append(best_track_idx)

        return matched_det_indices, matched_track_indices

    def _update_lost_tracks(self):
        """Update lost tracks and remove old ones"""
        tracks_to_remove = []

        for track_id, track_data in self.tracks.items():
            if track_data["state"] == "tracked":
                # Track is still active
                track_data["total_frames"] += 1
            else:
                # Track is lost
                frames_lost = self.frame_count - track_data["last_seen"]
                if frames_lost > self.track_buffer:
                    tracks_to_remove.append(track_id)
                else:
                    track_data["state"] = "lost"

        # Remove old lost tracks
        for track_id in tracks_to_remove:
            del self.tracks[track_id]

    def get_tracks(self) -> List[Dict[str, Any]]:
        """Get all active tracks"""
        return [
            track_data for track_data in self.tracks.values()
            if track_data["state"] == "tracked"
        ]

    def get_all_tracks(self) -> List[Dict[str, Any]]:
        """Get all tracks (including lost ones)"""
        return list(self.tracks.values())

    def reset(self):
        """Reset tracker state"""
        self.track_id_count = 0
        self.tracks = {}
        self.lost_tracks = {}
        self.frame_count = 0
