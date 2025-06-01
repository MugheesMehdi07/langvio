"""
Video visualization module
"""

import logging
from typing import Any, Dict, List, Union, Tuple
import cv2
import numpy as np


class VideoVisualizer:
    """Handles video visualization with detection overlays and tracking"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def visualize_with_highlights(
            self,
            video_path: str,
            output_path: str,
            all_frame_detections: Dict[str, List[Dict[str, Any]]],
            highlighted_objects: List[Dict[str, Any]],
            original_box_color: Union[Tuple[int, int, int], List[int]] = (0, 255, 0),
            highlight_color: Union[Tuple[int, int, int], List[int]] = (0, 0, 255),
            text_color: Union[Tuple[int, int, int], List[int]] = (255, 255, 255),
            line_thickness: int = 2,
            show_attributes: bool = True,
            show_confidence: bool = True,
    ) -> None:
        """Visualize all detections on a video with highlighted objects in a different color"""
        self.logger.info(f"Visualizing all detections on video: {video_path}")

        try:
            # Create a lookup for highlighted objects by frame
            highlighted_by_frame = {}

            for obj in highlighted_objects:
                frame_key = obj.get("frame_key")
                detection = obj.get("detection")

                if frame_key and detection:
                    if frame_key not in highlighted_by_frame:
                        highlighted_by_frame[frame_key] = []

                    highlighted_by_frame[frame_key].append(detection)

            # Open input video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video: {video_path}")

            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # For tracking visualization - keep track of past positions
            tracks = {}  # Dictionary mapping track_id to list of past positions

            # Process frames
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Check if we have detections for this frame
                frame_key = str(frame_idx)
                if frame_key in all_frame_detections:
                    # Get current detections
                    all_detections = all_frame_detections[frame_key]

                    # Get highlighted detections for this frame
                    highlighted_detections = highlighted_by_frame.get(frame_key, [])

                    # Create set of highlighted detections for quick lookup
                    highlighted_signatures = set()
                    for det in highlighted_detections:
                        if "bbox" in det and "label" in det:
                            signature = (
                                det["label"],
                                tuple(det["bbox"]) if isinstance(det["bbox"], list) else det["bbox"]
                            )
                            highlighted_signatures.add(signature)

                    # Update tracks for visualization
                    for det in all_detections:
                        if "track_id" in det:
                            track_id = det["track_id"]
                            # Get center of bounding box
                            bbox = det["bbox"]
                            center = (
                                int((bbox[0] + bbox[2]) / 2),
                                int((bbox[1] + bbox[3]) / 2),
                            )

                            # Add to track history
                            if track_id not in tracks:
                                tracks[track_id] = []

                            # Keep only last 30 positions
                            if len(tracks[track_id]) > 30:
                                tracks[track_id] = tracks[track_id][-30:]

                            tracks[track_id].append(center)

                    # Draw trajectory lines for tracked objects
                    for track_id, positions in tracks.items():
                        if len(positions) > 1:
                            # Generate unique color for this track
                            track_color = self._get_color_for_id(track_id)

                            # Draw line connecting positions
                            for i in range(len(positions) - 1):
                                cv2.line(
                                    frame,
                                    positions[i],
                                    positions[i + 1],
                                    track_color,
                                    thickness=max(1, line_thickness - 1),
                                )

                    # Draw all detections with appropriate colors
                    for det in all_detections:
                        # Check if this detection is in the highlighted set
                        is_highlighted = False
                        if "bbox" in det and "label" in det:
                            signature = (
                                det["label"],
                                tuple(det["bbox"]) if isinstance(det["bbox"], list) else det["bbox"]
                            )
                            is_highlighted = signature in highlighted_signatures

                        # Choose color based on whether the detection is highlighted
                        box_color = highlight_color if is_highlighted else original_box_color

                        # Use thicker lines for highlighted objects
                        thickness = line_thickness + 1 if is_highlighted else line_thickness

                        # Draw the detection with the chosen color and thickness
                        frame = self._draw_single_detection(
                            frame, det, box_color, text_color, thickness,
                            show_attributes, show_confidence, is_highlighted
                        )

                # Write frame
                writer.write(frame)
                frame_idx += 1

            # Clean up
            cap.release()
            writer.release()
            self.logger.info(f"Saved visualized video to: {output_path}")
        except Exception as e:
            self.logger.error(f"Error visualizing video: {e}")

    def _get_color_for_id(self, track_id: int) -> Tuple[int, int, int]:
        """Generate a consistent color for a given track ID"""
        # Use the track_id to generate repeatable colors
        hue = (track_id * 137 % 360) / 360.0  # Use prime number to distribute colors
        sat = 0.7 + (track_id % 3) * 0.1  # Vary saturation slightly
        val = 0.8 + (track_id % 2) * 0.2  # Vary value slightly

        # Convert HSV to RGB then to BGR
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        r, g, b = int(r * 255), int(g * 255), int(b * 255)

        return (b, g, r)  # Return BGR for OpenCV

    def _draw_single_detection(
            self,
            image: np.ndarray,
            det: Dict[str, Any],
            box_color: Union[Tuple[int, int, int], List[int]],
            text_color: Union[Tuple[int, int, int], List[int]],
            line_thickness: int,
            show_attributes: bool,
            show_confidence: bool,
            is_highlighted: bool = False,
    ) -> np.ndarray:
        """Draw a single detection on a video frame"""
        if "bbox" not in det:
            return image  # Skip detections without bounding boxes

        # Extract bounding box
        x1, y1, x2, y2 = det["bbox"]

        # Make sure coordinates are integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Check for valid box dimensions
        if x2 <= x1 or y2 <= y1:
            return image  # Skip invalid boxes

        # Create label based on configuration
        label_parts = [det["label"]]

        # Add confidence if requested
        if show_confidence and "confidence" in det:
            conf = det["confidence"]
            if isinstance(conf, (int, float)):
                label_parts.append(f"{conf:.2f}")

        # Add attributes if requested and present (limit to 2 most important)
        if show_attributes and "attributes" in det and det["attributes"]:
            # Prioritize color and size attributes
            priority_attrs = []
            for key in ["color", "size"]:
                if key in det["attributes"]:
                    priority_attrs.append(f"{key}:{det['attributes'][key]}")

            # Add up to 2 priority attributes to avoid cluttering
            if priority_attrs:
                label_parts.extend(priority_attrs[:2])

        # Add activities if present (limit to 1 most important)
        if "activities" in det and det["activities"] and len(det["activities"]) > 0:
            # Only add the first activity to avoid cluttering
            label_parts.append(f"[{det['activities'][0]}]")

        # Add "HIGHLIGHT" tag if this is a highlighted detection
        if is_highlighted:
            label_parts.append("*")

        # Combine into label
        label = " | ".join(label_parts)

        # Draw bounding box with line thickness scaled by image size
        thickness = max(
            1, min(line_thickness, int(min(image.shape[0], image.shape[1]) / 500))
        )
        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, thickness)

        # Calculate text size and scale font size based on image dimensions
        font_scale = max(0.3, min(0.5, min(image.shape[0], image.shape[1]) / 1000))
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(label, font, font_scale, 1)[0]

        # Draw text background with slight transparency
        alpha = 0.6  # Transparency factor
        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (x1, y1 - text_size[1] - 5),
            (x1 + text_size[0], y1),
            box_color,
            -1,
        )
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        # Draw text
        cv2.putText(
            image, label, (x1, y1 - 5), font, font_scale, text_color, 1
        )

        return image