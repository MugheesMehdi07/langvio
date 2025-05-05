"""
Enhanced YOLO-based vision processor
"""

import logging
import os
import tempfile
from typing import Any, Dict, List

import cv2
import torch
from ultralytics import YOLO, YOLOE

from langvio.prompts.constants import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_VIDEO_SAMPLE_RATE,
)
from langvio.vision.base import BaseVisionProcessor
from langvio.vision.utils import extract_detections, optimize_for_memory
from langvio.vision.yolo.yolo11_utils import check_yolo11_solutions_available


class YOLOProcessor(BaseVisionProcessor):
    """Enhanced vision processor using YOLO models"""

    def __init__(self, name: str, model_path: str, confidence: float, **kwargs):
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
            "confidence": DEFAULT_CONFIDENCE_THRESHOLD,
            **kwargs,
        }
        super().__init__(name, config)
        self.logger = logging.getLogger(__name__)
        self.model = None

    def initialize(self) -> bool:
        """
        Initialize the YOLO model.

        Returns:
            True if initialization was successful
        """
        try:
            self.logger.info(f"Loading YOLO model: {self.config['model_path']}")
            self.model = (
                YOLO(self.config["model_path"])
                if self.name == "yolo"
                else YOLOE(self.config["model_path"])
            )
            return True
        except Exception as e:
            self.logger.error(f"Error loading YOLO model: {e}")
            return False

    # Update langvio/vision/yolo/detector.py

    def process_image(self, image_path: str, query_params: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process an image with YOLO with enhanced detection capabilities.
        Uses YOLO11 Solutions for advanced metrics.

        Args:
            image_path: Path to the input image
            query_params: Parameters from the query processor

        Returns:
            Dictionary with all detection results without filtering
        """
        self.logger.info(f"Processing image: {image_path}")

        # Load model if not already loaded
        if not self.model:
            self.initialize()

        with torch.no_grad():
            # Run detection
            try:
                optimize_for_memory()
                # Get image dimensions for relative positioning
                image_dimensions = self._get_image_dimensions(image_path)

                # Run basic object detection
                results = self.model(image_path, conf=self.config["confidence"])

                # Extract detections
                detections = extract_detections(results)

                # Enhance detections with attributes based on the image
                detections = self._enhance_detections_with_attributes(
                    detections, image_path
                )

                # Calculate relative positions if image dimensions provided
                if image_dimensions:
                    from langvio.vision.utils import (
                        calculate_relative_positions,
                        detect_spatial_relationships,
                    )

                    detections = calculate_relative_positions(detections, *image_dimensions)
                    detections = detect_spatial_relationships(detections)

                # Set up initial frame_detections
                frame_detections = {"0": detections}


                yolo11_available = check_yolo11_solutions_available()

                # Process with YOLO11 Solutions if available
                all_metrics = []


                self.logger.info(f"All Metrics: {all_metrics}")

                # Combine metrics
                # summary = combine_metrics(frame_detections, all_metrics)

                # Add summary to frame detections
                # frame_detections["summary"] = summary

                return frame_detections
            except Exception as e:
                self.logger.error(f"Error processing image: {e}")
                return {"0": []}

    def process_video(
            self,
            video_path: str,
            query_params: Dict[str, Any],
            sample_rate: int = DEFAULT_VIDEO_SAMPLE_RATE,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process a video with YOLO with enhanced activity and tracking detection.
        Uses YOLO11 Solutions for advanced metrics in a single pass.

        Args:
            video_path: Path to the input video
            query_params: Parameters from the query processor
            sample_rate: Process every Nth frame

        Returns:
            Dictionary with all detection results without filtering
        """
        self.logger.info(f"Processing video: {video_path} (sample rate: {sample_rate})")

        # Load model if not already loaded
        if not self.model:
            self.initialize()

        with torch.no_grad():
            # Open video
            try:
                optimize_for_memory()
                # Check if YOLO11 Solutions is available
                yolo11_available = check_yolo11_solutions_available()

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

                # Extract target classes if provided
                target_objects = query_params.get("target_objects", [])

                # Process frames
                frame_detections = {}
                all_metrics = []

                # Use tracking for the whole video to get consistent track IDs
                results = self.model.track(source=video_path, conf=self.config["confidence"], persist=True)

                # Get YOLO11 metrics from a sample frame

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
                    from langvio.vision.utils import (
                        calculate_relative_positions,
                        detect_spatial_relationships,
                    )

                    detections = calculate_relative_positions(
                        detections, *video_dimensions
                    )
                    detections = detect_spatial_relationships(detections)

                    # Enhance with attributes (using original frame from result)
                    if hasattr(result, 'orig_img'):
                        detections = self._enhance_detections_with_attributes(
                            detections, None, result.orig_img
                        )

                    # Store results for this frame
                    frame_detections[str(frame_idx)] = detections

                # Analyze for activities and tracking across frames
                if frame_detections:
                    frame_detections = self._analyze_video_for_activities(
                        frame_detections, query_params
                    )

                # Combine all metrics
                # summary = combine_metrics(frame_detections, all_metrics)

                # Add summary to frame detections
                # frame_detections["summary"] = summary

                return frame_detections
            except Exception as e:
                self.logger.error(f"Error processing video: {e}")
                return {}
