"""
YOLO-World video visualization module with tracker file support
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


class YOLOWorldVideoVisualizer:
    """Handles video visualization with YOLO-World + ByteTracker data"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.previous_boxes: Dict[str, Any] = {}

    def visualize_with_tracker_data(  # noqa: C901
        self,
        video_path: str,
        output_path: str,
        tracker_data: Dict[str, Any],
        highlighted_objects: Optional[List[Dict[str, Any]]] = None,
        original_box_color: Union[Tuple[int, int, int], List[int]] = (0, 255, 0),
        highlight_color: Union[Tuple[int, int, int], List[int]] = (0, 0, 255),
        text_color: Union[Tuple[int, int, int], List[int]] = (255, 255, 255),
        line_thickness: int = 2,
        show_attributes: bool = True,
        show_confidence: bool = True,
        show_tracking: bool = True,
    ) -> None:
        """Visualize video with tracker data and highlighted objects"""
        self.logger.info(f"Visualizing video with tracker data: {video_path}")

        try:
            # Extract data from tracker format
            detections = tracker_data.get("detections", [])
            tracks = tracker_data.get("tracks", [])

            # Create highlighted objects lookup
            highlighted_track_ids = set()
            if highlighted_objects:
                for obj in highlighted_objects:
                    if "track_id" in obj:
                        highlighted_track_ids.add(obj["track_id"])
                    elif "object_id" in obj:
                        # Try to find track_id from object_id
                        for track in tracks:
                            if obj["object_id"] in str(track.get("track_id", "")):
                                highlighted_track_ids.add(track["track_id"])
                                break

            # Open input video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Setup video writer
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")  # type: ignore[attr-defined]
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # Create frame detection lookup
            frame_detections = {}
            for frame_data in detections:
                frame_id = frame_data["frame_id"]
                frame_detections[frame_id] = frame_data["objects"]

            # Track visualization data
            track_trajectories: Dict[int, List[Tuple[int, int]]] = {}
            track_colors: Dict[int, Tuple[int, int, int]] = {}

            # Store last known detections for interpolation (to prevent flickering)
            last_known_detections: Dict[int, Dict[str, Any]] = (
                {}
            )  # track_id -> detection
            last_frame_with_detections = -1
            last_known_frame_detections: List[Dict[str, Any]] = (
                []
            )  # Full list of last detections

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Get detections for this frame
                current_detections = frame_detections.get(frame_count, [])

                # If no detections for this frame, use interpolated detections from last known frame
                if not current_detections and last_known_frame_detections:
                    # Use last known detections to prevent flickering
                    # Create deep copies to avoid modifying original data
                    current_detections = []
                    for det in last_known_frame_detections:
                        interpolated_det = det.copy()
                        if "bbox" in interpolated_det:
                            interpolated_det["bbox"] = interpolated_det["bbox"].copy()
                        if "attributes" in interpolated_det:
                            interpolated_det["attributes"] = interpolated_det[
                                "attributes"
                            ].copy()
                        interpolated_det["_interpolated"] = True
                        current_detections.append(interpolated_det)
                elif current_detections:
                    # Update last known detections
                    last_known_detections.clear()
                    last_known_frame_detections = []
                    for det in current_detections:
                        track_id = det.get("track_id")
                        if track_id is not None:
                            last_known_detections[track_id] = det.copy()
                        # Store full detection for interpolation
                        det_copy = det.copy()
                        if "bbox" in det_copy:
                            det_copy["bbox"] = det_copy["bbox"].copy()
                        if "attributes" in det_copy:
                            det_copy["attributes"] = det_copy["attributes"].copy()
                        last_known_frame_detections.append(det_copy)
                    last_frame_with_detections = frame_count

                # Draw tracking trajectories
                if show_tracking:
                    self._draw_tracking_trajectories(
                        frame, current_detections, track_trajectories, track_colors
                    )

                # Draw detections
                for detection in current_detections:
                    track_id = detection.get("track_id")
                    is_highlighted = (
                        track_id in highlighted_track_ids
                        if highlighted_track_ids
                        else False
                    )

                    # Choose color
                    if is_highlighted:
                        box_color = highlight_color
                    else:
                        box_color = original_box_color

                    # Draw bounding box
                    box_color_tuple: Tuple[int, int, int] = (
                        tuple(box_color) if isinstance(box_color, list) else box_color  # type: ignore[assignment]
                    )
                    text_color_tuple: Tuple[int, int, int] = (
                        tuple(text_color) if isinstance(text_color, list) else text_color  # type: ignore[assignment]
                    )
                    self._draw_detection_box(
                        frame,
                        detection,
                        box_color_tuple,
                        text_color_tuple,
                        line_thickness,
                        show_attributes,
                        show_confidence,
                        show_tracking,
                    )

                # Write frame
                out.write(frame)
                frame_count += 1

                # Log progress
                if frame_count % 100 == 0:
                    self.logger.info(f"Processed {frame_count}/{total_frames} frames")

            # Cleanup
            cap.release()
            out.release()

            self.logger.info(f"Video visualization saved to: {output_path}")

        except Exception as e:
            self.logger.error(f"Error visualizing video: {e}")
            raise

    def _draw_tracking_trajectories(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        track_trajectories: Dict[int, List[Tuple[int, int]]],
        track_colors: Dict[int, Tuple[int, int, int]],
    ):
        """Draw tracking trajectories for objects"""
        for detection in detections:
            track_id = detection.get("track_id")
            if not track_id:
                continue

            # Get center point
            bbox = detection.get("bbox", [0, 0, 0, 0])
            center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))

            # Initialize track trajectory if needed
            if track_id not in track_trajectories:
                track_trajectories[track_id] = []
                # Generate unique color for this track
                track_colors[track_id] = self._get_color_for_track_id(track_id)

            # Add current position
            track_trajectories[track_id].append(center)

            # Keep only last 30 positions
            if len(track_trajectories[track_id]) > 30:
                track_trajectories[track_id] = track_trajectories[track_id][-30:]

            # Draw trajectory
            if len(track_trajectories[track_id]) > 1:
                color = track_colors[track_id]
                for i in range(len(track_trajectories[track_id]) - 1):
                    cv2.line(
                        frame,
                        track_trajectories[track_id][i],
                        track_trajectories[track_id][i + 1],
                        color,
                        thickness=2,
                    )

    def _draw_detection_box(
        self,
        frame: np.ndarray,
        detection: Dict[str, Any],
        box_color: Tuple[int, int, int],
        text_color: Tuple[int, int, int],
        line_thickness: int,
        show_attributes: bool,
        show_confidence: bool,
        show_tracking: bool,
    ):
        """Draw detection bounding box with labels"""
        bbox = detection.get("bbox", [0, 0, 0, 0])
        track_id = detection.get("track_id")

        x1, y1, x2, y2 = map(int, bbox)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, line_thickness)

        # Prepare label text
        label_parts = [detection.get("label", "object")]

        if show_confidence:
            confidence = detection.get("confidence", 0)
            label_parts.append(f"{confidence:.2f}")

        if show_tracking and "track_id" in detection:
            track_id = detection.get("track_id")
            label_parts.append(f"ID:{track_id}")

        if show_attributes and "attributes" in detection:
            attrs = detection["attributes"]
            if "size" in attrs:
                label_parts.append(attrs["size"])
            if "position" in attrs:
                label_parts.append(attrs["position"])

        # Draw label
        label = " ".join(label_parts)
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

        # Draw label background
        cv2.rectangle(
            frame,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0], y1),
            box_color,
            -1,
        )

        # Draw label text
        cv2.putText(
            frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1
        )

    def _get_color_for_track_id(self, track_id: int) -> Tuple[int, int, int]:
        """Generate a unique color for a track ID"""
        # Use track_id to generate consistent colors
        np.random.seed(track_id)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        return color

    def load_tracker_file(self, tracker_file_path: str) -> Dict[str, Any]:
        """Load tracker data from file"""
        try:
            with open(tracker_file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading tracker file: {e}")
            raise

    def visualize_from_tracker_file(
        self,
        video_path: str,
        tracker_file_path: str,
        output_path: str,
        highlighted_objects: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> None:
        """Visualize video from tracker file"""
        tracker_data = self.load_tracker_file(tracker_file_path)
        self.visualize_with_tracker_data(
            video_path, output_path, tracker_data, highlighted_objects, **kwargs
        )
