"""
Manages visualization creation and configuration
"""

import logging
from typing import Any, Dict, List

import cv2

from langvio.media.processor import MediaProcessor
from langvio.utils.file_utils import is_video_file


class VisualizationManager:
    """Manages visualization creation and configuration"""

    def __init__(self, config):
        self.config = config
        self.media_processor = MediaProcessor(config.get_media_config())
        self.logger = logging.getLogger(__name__)

    def create_visualization(
        self,
        media_path: str,
        detections: Dict[str, Any],
        highlighted_objects: List[Dict[str, Any]],
        query_params: Dict[str, Any],
    ) -> str:
        """Create visualization with highlighted objects"""
        # Generate output path
        output_path = self.media_processor.get_output_path(media_path)

        # Get visualization config
        visualization_config = self._get_visualization_config(query_params)

        # Check if this is a video
        is_video = is_video_file(media_path)

        if is_video:
            # For videos, create visualization with comprehensive data
            self._create_video_visualization(
                media_path,
                output_path,
                detections,
                highlighted_objects,
                visualization_config,
            )
        else:
            # For images, we have object-level data
            self._create_image_visualization(
                media_path,
                output_path,
                detections,
                highlighted_objects,
                visualization_config,
            )

        return output_path

    def _get_visualization_config(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get visualization configuration based on query parameters"""
        # Get default visualization config
        viz_config = self.config.config["media"]["visualization"].copy()

        # Customize based on task type
        task_type = query_params.get("task_type", "identification")

        if task_type == "counting":
            # For counting tasks, use a different color
            viz_config["box_color"] = [255, 0, 0]  # Red for counting

        elif task_type == "verification":
            # For verification tasks, use a different color
            viz_config["box_color"] = [0, 0, 255]  # Blue for verification

        elif task_type in ["tracking", "activity"]:
            # For tracking/activity tasks, use a more visible color
            viz_config["box_color"] = [255, 165, 0]  # Orange for tracking/activity
            viz_config["line_thickness"] = 3  # Thicker lines

        # If specific attributes were requested, adjust the visualization
        if query_params.get("attributes"):
            # If looking for specific attributes, highlight them more
            viz_config["line_thickness"] += 1

        return viz_config

    def _create_image_visualization(
        self,
        image_path: str,
        output_path: str,
        detections: Dict[str, Any],
        highlighted_objects: List[Dict[str, Any]],
        viz_config: Dict[str, Any],
    ) -> None:
        """Create image visualization"""
        original_box_color = viz_config["box_color"]
        highlight_color = [0, 0, 255]  # Red color (BGR) for highlighted objects
        image_objects = detections.get("objects", [])

        self.media_processor.visualize_image_with_highlights(
            image_path,
            output_path,
            image_objects,
            [
                obj["detection"] for obj in highlighted_objects
            ],  # Extract detection objects
            original_box_color=original_box_color,
            highlight_color=highlight_color,
            text_color=viz_config["text_color"],
            line_thickness=viz_config["line_thickness"],
            show_attributes=viz_config.get("show_attributes", True),
            show_confidence=viz_config.get("show_confidence", True),
        )

    def _create_video_visualization(
        self,
        video_path: str,
        output_path: str,
        video_results: Dict[str, Any],
        highlighted_objects: List[Dict[str, Any]],
        viz_config: Dict[str, Any],
    ) -> None:
        """Create enhanced video visualization"""
        import cv2

        try:
            # Extract data from results
            frame_detections = video_results.get("frame_detections", {})
            summary = video_results.get("summary", {})

            if not frame_detections:
                # Fallback: copy original video if no detections
                import shutil

                shutil.copy2(video_path, output_path)
                return

            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                import shutil

                shutil.copy2(video_path, output_path)
                return

            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # Prepare overlay information
            overlay_info = self._prepare_overlay_information(summary)

            # Create highlighted object lookup
            highlighted_lookup = set()
            for obj in highlighted_objects:
                if "detection" in obj and "object_id" in obj["detection"]:
                    highlighted_lookup.add(obj["detection"]["object_id"])

            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Draw detections if we have them for this frame
                frame_key = str(frame_idx)
                if frame_key in frame_detections:
                    frame = self._draw_detections_on_frame(
                        frame,
                        frame_detections[frame_key],
                        highlighted_lookup,
                        viz_config,
                    )

                # Add comprehensive overlay
                frame = self._add_comprehensive_overlay(
                    frame, overlay_info, frame_idx, fps
                )

                writer.write(frame)
                frame_idx += 1

            cap.release()
            writer.release()
            self.logger.info(f"Created enhanced video visualization: {output_path}")

        except Exception as e:
            self.logger.error(f"Error creating enhanced video visualization: {e}")
            # Fallback: copy original video
            import shutil

            shutil.copy2(video_path, output_path)

    def _prepare_overlay_information(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare comprehensive overlay information from video summary"""
        overlay_info = {"lines": [], "stats": {}, "insights": []}

        # Video info
        video_info = summary.get("video_info", {})
        if video_info.get("primary_objects"):
            overlay_info["lines"].append(
                f"Objects: {', '.join(video_info['primary_objects'][:3])}"
            )

        # YOLO11 counting results (PRIORITY)
        counting = summary.get("counting_analysis", {})
        if counting:
            if "total_crossings" in counting:
                overlay_info["lines"].append(
                    f"Crossings: {counting['total_crossings']}"
                )
            if "net_flow" in counting and counting["net_flow"] != 0:
                flow_direction = "In" if counting["net_flow"] > 0 else "Out"
                overlay_info["lines"].append(
                    f"Net Flow: {abs(counting['net_flow'])} {flow_direction}"
                )

        # Speed analysis
        speed = summary.get("speed_analysis", {})
        if speed and speed.get("speed_available"):
            if "average_speed_kmh" in speed:
                overlay_info["lines"].append(
                    f"Avg Speed: {speed['average_speed_kmh']} km/h"
                )

        # Movement patterns
        temporal = summary.get("temporal_relationships", {})
        if temporal:
            movement = temporal.get("movement_patterns", {})
            if movement:
                moving_count = movement.get("moving_count", 0)
                stationary_count = movement.get("stationary_count", 0)
                overlay_info["lines"].append(
                    f"Moving: {moving_count}, Static: {stationary_count}"
                )

        # Primary insights
        insights = summary.get("primary_insights", [])
        overlay_info["insights"] = insights[:3]  # Top 3 insights

        return overlay_info

    def _draw_detections_on_frame(
        self,
        frame,
        detections: List[Dict[str, Any]],
        highlighted_lookup: set,
        viz_config: Dict[str, Any],
    ):
        """Draw detections on frame with highlighting support"""
        default_color = viz_config.get("box_color", [0, 255, 0])
        highlight_color = [0, 0, 255]  # Red for highlighted objects
        text_color = viz_config.get("text_color", [255, 255, 255])
        thickness = viz_config.get("line_thickness", 2)

        for det in detections:
            if "bbox" not in det:
                continue

            try:
                x1, y1, x2, y2 = map(int, det["bbox"])

                # Determine if this object should be highlighted
                obj_id = det.get("object_id", "")
                is_highlighted = obj_id in highlighted_lookup

                # Choose color and thickness
                color = highlight_color if is_highlighted else default_color
                line_thickness = thickness + 1 if is_highlighted else thickness

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thickness)

                # Prepare label
                label_parts = [det.get("label", "object")]

                # Add confidence
                if "confidence" in det:
                    label_parts.append(f"{det['confidence']:.2f}")

                # Add key attributes
                attributes = det.get("attributes", {})
                for attr_key in ["color", "size"]:
                    if attr_key in attributes and attributes[attr_key] != "unknown":
                        label_parts.append(f"{attr_key[0]}:{attributes[attr_key]}")

                # Add highlight indicator
                if is_highlighted:
                    label_parts.append("★")

                label = " ".join(label_parts)

                # Draw label background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                text_size = cv2.getTextSize(label, font, font_scale, 1)[0]

                # Draw background rectangle
                cv2.rectangle(
                    frame,
                    (x1, y1 - text_size[1] - 5),
                    (x1 + text_size[0], y1),
                    color,
                    -1,
                )

                # Draw text
                cv2.putText(frame, label, (x1, y1 - 5), font, font_scale, text_color, 1)

            except Exception:
                continue  # Skip problematic detections

        return frame

    def _add_comprehensive_overlay(
        self, frame, overlay_info: Dict[str, Any], frame_idx: int, fps: float
    ):
        """Add comprehensive overlay with stats and insights"""
        height, width = frame.shape[:2]

        # Semi-transparent overlay background
        overlay = frame.copy()

        # Top-left stats panel
        if overlay_info["lines"]:
            panel_height = len(overlay_info["lines"]) * 25 + 20
            cv2.rectangle(overlay, (10, 10), (300, panel_height), (0, 0, 0), -1)

            y_offset = 30
            for line in overlay_info["lines"]:
                cv2.putText(
                    overlay,
                    line,
                    (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                y_offset += 25

        # Bottom-left insights panel
        insights = overlay_info.get("insights", [])
        if insights:
            # Calculate panel dimensions
            max_text_width = max(len(insight) for insight in insights) * 8
            panel_width = min(max_text_width + 20, width - 20)
            panel_height = len(insights) * 25 + 20
            panel_y = height - panel_height - 10

            cv2.rectangle(
                overlay, (10, panel_y), (panel_width, height - 10), (0, 0, 0), -1
            )

            y_offset = panel_y + 20
            for insight in insights:
                # Truncate long insights
                display_insight = insight[:50] + "..." if len(insight) > 50 else insight
                cv2.putText(
                    overlay,
                    display_insight,
                    (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,  # Yellow text
                )
                y_offset += 25

        # Top-right timestamp
        timestamp = f"Time: {frame_idx / fps:.1f}s"
        cv2.putText(
            overlay,
            timestamp,
            (width - 150, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # Blend overlay with original frame (60% transparency)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        return frame
