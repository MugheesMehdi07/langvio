"""
Enhanced YOLO-based vision processor with YOLO11 utilities integration
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import torch
from ultralytics import YOLO, YOLOE

from langvio.prompts.constants import DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_VIDEO_SAMPLE_RATE
from langvio.vision.base import BaseVisionProcessor
from langvio.vision.utils import extract_detections, optimize_for_memory
from langvio.vision.yolo.yolo11_utils import (process_video_with_yolo11,
                                             process_image_with_yolo11,
                                             check_yolo11_solutions_available)


class YOLOProcessor(BaseVisionProcessor):
    """Enhanced vision processor using YOLO models with YOLO11 utilities integration"""

    def __init__(self, name: str, model_path: str, confidence: float = DEFAULT_CONFIDENCE_THRESHOLD, **kwargs):
        """
        Initialize YOLO processor.

        Args:
            name: Processor name
            model_path: Path to the YOLO model
            confidence: Confidence threshold for detections
            **kwargs: Additional parameters for YOLO
        """
        config = {
            "model_path": model_path,
            "confidence": confidence,
            **kwargs,
        }
        super().__init__(name, config)
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.model_type = kwargs.get("model_type", "yolo")

    def initialize(self) -> bool:
        """
        Initialize the YOLO model.

        Returns:
            True if initialization was successful
        """
        try:
            self.logger.info(f"Loading {self.model_type} model: {self.config['model_path']}")

            # Determine model type (YOLO or YOLOe)
            if self.model_type == "yoloe":
                self.model = YOLOE(self.config['model_path'])
            else:
                self.model = YOLO(self.config['model_path'])

            # Check if YOLO11 Solutions is available
            self.has_yolo11_solutions = check_yolo11_solutions_available()
            if self.has_yolo11_solutions:
                self.logger.info("YOLO11 Solutions is available for advanced metrics")
            else:
                self.logger.info("YOLO11 Solutions not available - using basic detection only")

            return True
        except Exception as e:
            self.logger.error(f"Error loading YOLO model: {e}")
            return False

    def process_image(self, image_path: str, query_params: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process an image with YOLO with enhanced detection capabilities.
        Uses YOLO11 Solutions for advanced metrics if available.

        Args:
            image_path: Path to the input image
            query_params: Parameters from the query processor

        Returns:
            Dictionary with all detection results and metrics
        """
        self.logger.info(f"Processing image: {image_path}")

        # Load model if not already loaded
        if not self.model:
            self.initialize()

        with torch.no_grad():
            try:
                optimize_for_memory()

                # Get image dimensions for relative positioning
                image_dimensions = self._get_image_dimensions(image_path)

                # Run basic object detection
                results = self.model(image_path, conf=self.config["confidence"])

                # Extract detections
                detections = extract_detections(results)

                # Enhance detections with attributes based on the image
                detections = self._enhance_detections_with_attributes(detections, image_path)

                # Calculate relative positions if image dimensions provided
                if image_dimensions:
                    from langvio.vision.utils import calculate_relative_positions, detect_spatial_relationships
                    detections = calculate_relative_positions(detections, *image_dimensions)
                    detections = detect_spatial_relationships(detections)

                # Set up frame_detections with basic detections
                frame_detections = {"0": detections}

                # Process with YOLO11 Solutions if available
                if self.has_yolo11_solutions and query_params.get("task_type") in ["counting", "analysis"]:
                    try:
                        yolo11_metrics = process_image_with_yolo11(
                            image_path=image_path,
                            model_path=self.config["model_path"],
                            confidence=self.config["confidence"]
                        )

                        # Add metrics to result
                        frame_detections["metrics"] = yolo11_metrics

                        # Create a summary combining basic detections with metrics
                        frame_detections["summary"] = self._create_summary(detections, yolo11_metrics)

                    except Exception as e:
                        self.logger.error(f"Error processing image with YOLO11 solutions: {e}")

                return frame_detections

            except Exception as e:
                self.logger.error(f"Error processing image: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                return {"0": [], "error": str(e)}

    def process_video(
            self,
            video_path: str,
            query_params: Dict[str, Any],
            sample_rate: int = DEFAULT_VIDEO_SAMPLE_RATE,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process a video with YOLO with enhanced activity and tracking detection.
        Uses YOLO11 Solutions for advanced metrics if available.

        Args:
            video_path: Path to the input video
            query_params: Parameters from the query processor
            sample_rate: Process every Nth frame

        Returns:
            Dictionary with all detection results, tracking, and metrics
        """
        self.logger.info(f"Processing video: {video_path} (sample rate: {sample_rate})")

        # Load model if not already loaded
        if not self.model:
            self.initialize()

        with torch.no_grad():
            try:
                optimize_for_memory()

                # Get video properties
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise ValueError(f"Failed to open video: {video_path}")

                # Get video dimensions
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                video_dimensions = (width, height)
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 25.0  # Default FPS

                cap.release()  # Release and reopen for processing

                # Process frames with tracking for consistent IDs
                frame_detections = {}

                # Use tracking for the whole video to get consistent track IDs
                results = self.model.track(
                    source=video_path,
                    conf=self.config["confidence"],
                    persist=True,
                    stream=True  # Stream for memory efficiency
                )

                # Process each frame from the tracking results
                for frame_idx, result in enumerate(results):
                    # Process every Nth frame (according to sample_rate)
                    if frame_idx % sample_rate != 0:
                        continue

                    # Extract detections with track IDs
                    detections = []

                    if result.boxes and hasattr(result.boxes, 'id') and result.boxes.id is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        track_ids = result.boxes.id.int().cpu().numpy()
                        cls_ids = result.boxes.cls.int().cpu().numpy()
                        confs = result.boxes.conf.cpu().numpy()

                        for i, (box, track_id, cls_id, conf) in enumerate(zip(boxes, track_ids, cls_ids, confs)):
                            x1, y1, x2, y2 = map(int, box)
                            label = self.model.names[cls_id]

                            # Calculate center and dimensions
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            width = x2 - x1
                            height = y2 - y1

                            detection = {
                                "label": label,
                                "confidence": float(conf),
                                "bbox": [x1, y1, x2, y2],
                                "class_id": int(cls_id),
                                "track_id": int(track_id),
                                "center": (float(center_x), float(center_y)),
                                "dimensions": (float(width), float(height)),
                                "area": float(width * height),
                                "attributes": {},
                                "activities": []
                            }

                            detections.append(detection)

                    # Calculate relative positions and relationships
                    from langvio.vision.utils import calculate_relative_positions, detect_spatial_relationships
                    detections = calculate_relative_positions(detections, *video_dimensions)
                    detections = detect_spatial_relationships(detections)

                    # Enhance with attributes (using original frame from result)
                    if hasattr(result, 'orig_img'):
                        # For video frames, we pass the actual frame instead of a path
                        detections = self._enhance_detections_with_attributes(
                            detections, None, result.orig_img
                        )

                    # Store results for this frame
                    frame_detections[str(frame_idx)] = detections

                # Analyze for activities across frames
                if frame_detections:
                    frame_detections = self._analyze_video_for_activities(
                        frame_detections, query_params
                    )

                # Process with YOLO11 Solutions for additional metrics if available
                if self.has_yolo11_solutions and query_params.get("task_type") in ["counting", "analysis", "tracking", "activity"]:
                    try:
                        yolo11_metrics = process_video_with_yolo11(
                            video_path=video_path,
                            model_path=self.config["model_path"],
                            sample_rate=sample_rate,
                            confidence=self.config["confidence"]
                        )

                        # Add metrics to result
                        frame_detections["metrics"] = yolo11_metrics

                        # Create summary combining detections with metrics
                        frame_detections["summary"] = self._create_video_summary(frame_detections, yolo11_metrics)

                    except Exception as e:
                        self.logger.error(f"Error processing video with YOLO11 solutions: {e}")

                return frame_detections

            except Exception as e:
                self.logger.error(f"Error processing video: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                return {"error": str(e)}

    def _enhance_detections_with_attributes(
        self,
        detections: List[Dict[str, Any]],
        image_path: Optional[str] = None,
        image: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Enhance detections with attribute information.

        Args:
            detections: List of detection dictionaries
            image_path: Path to the image (optional)
            image: Direct image array for video frames (optional)

        Returns:
            Detections with added attributes
        """
        # Load image if path is provided and image is not
        if image_path and image is None:
            try:
                image = cv2.imread(image_path)
            except Exception as e:
                self.logger.error(f"Error loading image for attribute detection: {e}")
                return detections

        # If we don't have an image, return unchanged detections
        if image is None:
            return detections

        image_height, image_width = image.shape[:2]

        for det in detections:
            # Skip if no bbox
            if "bbox" not in det:
                continue

            # Extract bounding box
            x1, y1, x2, y2 = det["bbox"]

            # Skip invalid boxes
            if (x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or
                x2 > image_width or y2 > image_height):
                continue

            # Get the object region
            obj_region = image[y1:y2, x1:x2]

            # Initialize attributes dictionary if not present
            if "attributes" not in det:
                det["attributes"] = {}

            # Calculate basic size attribute
            area = (x2 - x1) * (y2 - y1)
            image_area = image_width * image_height
            relative_size = area / image_area

            if relative_size < 0.05:
                det["attributes"]["size"] = "small"
            elif relative_size < 0.25:
                det["attributes"]["size"] = "medium"
            else:
                det["attributes"]["size"] = "large"

            # Extract dominant color
            if obj_region.size > 0:
                # Get color information
                from langvio.vision.color_detection import ColorDetector
                color_info = ColorDetector.get_color_profile(obj_region)

                # Add to detection attributes
                det["attributes"]["color"] = color_info["dominant_color"]
                det["attributes"]["is_multicolored"] = color_info["is_multicolored"]

                # Add all detected colors
                det["attributes"]["colors"] = list(color_info["color_percentages"].keys())

        return detections

    def _create_summary(self, detections: List[Dict[str, Any]], metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary that combines basic detections with YOLO11 metrics for images.

        Args:
            detections: List of detection dictionaries
            metrics: YOLO11 metrics

        Returns:
            Summary dictionary
        """
        # Count objects by type
        counts = {}
        for det in detections:
            label = det["label"]
            if label not in counts:
                counts[label] = 0
            counts[label] += 1

        # Create summary
        summary = {
            "counts": counts,
            "total_objects": len(detections),
            "object_types": list(counts.keys())
        }

        # Add YOLO11 metrics if available
        if metrics:
            if "counting" in metrics:
                summary["counting"] = metrics["counting"]

            if "distance" in metrics:
                summary["distance"] = metrics["distance"]

        return summary

    def _create_video_summary(self, frame_detections: Dict[str, List[Dict[str, Any]]], metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary that combines frame detections with YOLO11 metrics for videos.

        Args:
            frame_detections: Dictionary of frame detections
            metrics: YOLO11 metrics

        Returns:
            Summary dictionary
        """
        # Count objects by type across all frames
        counts = {}
        track_ids = set()

        for frame_key, detections in frame_detections.items():
            if not frame_key.isdigit():  # Skip non-frame keys like "metrics" or "summary"
                continue

            for det in detections:
                label = det["label"]
                if label not in counts:
                    counts[label] = 0
                counts[label] += 1

                # Track unique objects by track_id
                if "track_id" in det:
                    track_ids.add((label, det["track_id"]))

        # Create summary
        summary = {
            "counts": counts,
            "total_frames_analyzed": len([k for k in frame_detections.keys() if k.isdigit()]),
            "total_detections": sum(counts.values()),
            "unique_tracked_objects": len(track_ids),
            "object_types": list(counts.keys())
        }

        # Count unique objects by type
        unique_by_type = {}
        for label, _ in track_ids:
            if label not in unique_by_type:
                unique_by_type[label] = 0
            unique_by_type[label] += 1

        summary["unique_by_type"] = unique_by_type

        # Add YOLO11 metrics if available
        if metrics:
            if "counting" in metrics:
                summary["counting"] = metrics["counting"]

            if "speed" in metrics:
                summary["speed"] = metrics["speed"]

        return summary