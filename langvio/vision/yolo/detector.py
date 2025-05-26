"""
Simplified YOLO-based vision processor with integrated YOLOe and YOLO11 processing
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO, YOLOE

from langvio.prompts.constants import DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_VIDEO_SAMPLE_RATE
from langvio.vision.base import BaseVisionProcessor
from langvio.vision.utils import extract_detections, optimize_for_memory, calculate_relative_positions, detect_spatial_relationships
from langvio.vision.yolo.yolo11_utils import (
    check_yolo11_solutions_available,
    create_object_counter,
    create_speed_estimator,
    process_frame_with_yolo11,
    parse_solution_results, initialize_yolo11_tools
)


class YOLOProcessor(BaseVisionProcessor):
    """Simplified vision processor using YOLO models with integrated YOLO11 metrics"""

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

        # Check YOLO11 Solutions availability
        self.has_yolo11_solutions = check_yolo11_solutions_available()
        if self.has_yolo11_solutions:
            self.logger.info("YOLO11 Solutions is available for metrics")
        else:
            self.logger.info("YOLO11 Solutions not available - using basic detection only")

    def initialize(self) -> bool:
        """
        Initialize the YOLO model.

        Returns:
            True if initialization was successful
        """
        try:
            self.logger.info(f"Loading {self.model_type} model: {self.config['model_path']}")

            # Use YOLOE by default for better performance
            if self.model_type == "yoloe":
                self.model = YOLOE(self.config['model_path'])
            else:
                self.model = YOLO(self.config['model_path'])

            return True
        except Exception as e:
            self.logger.error(f"Error loading YOLO model: {e}")
            return False

    def initialize_video_capture(self, video_path: str) -> Tuple[cv2.VideoCapture, Tuple[int, int, float, int]]:
        """
        Initialize video capture and extract video properties.

        Args:
            video_path: Path to the video file

        Returns:
            Tuple containing (video_capture, (width, height, fps, total_frames))

        Raises:
            ValueError: If video cannot be opened
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0  # Default FPS
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        return cap, (width, height, fps, total_frames)

    def process_image(self, image_path: str, query_params: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process an image with YOLOe with enhanced detection capabilities.
        For images: only uses YOLOe with attribute computation.

        Args:
            image_path: Path to the input image
            query_params: Parameters from the query processor

        Returns:
            Dictionary with detection results and attributes
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

                # Run basic object detection with YOLOe
                results = self.model(image_path, conf=self.config["confidence"])

                # Extract detections
                detections = extract_detections(results)

                # Add position, size and color attributes to detections
                detections = self._enhance_detections_with_attributes(detections, image_path)

                # Calculate relative positions
                if image_dimensions:
                    detections = calculate_relative_positions(detections, *image_dimensions)
                    detections = detect_spatial_relationships(detections)

                # Create result with detections
                frame_detections = {"0": detections}

                # Create a simple summary of detections
                frame_detections["summary"] = self._create_simple_summary(detections)

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
            show_frame: bool = False
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process a video in a single loop, running both YOLOe and YOLO11 on each frame.

        Args:
            video_path: Path to the input video
            query_params: Parameters from the query processor
            sample_rate: Process every Nth frame

        Returns:
            Dictionary with combined detection results and metrics
        """
        self.logger.info(f"Processing video: {video_path}")

        # Load model if not already loaded
        if not self.model:
            self.initialize()

        # Initialize results container
        frame_detections = {}
        counter_results = None
        speed_results = None
        cap = None

        try:
            # 1. Initialize video and auxiliary tools
            cap, video_props = self.initialize_video_capture(video_path)
            width, height, fps, total_frames = video_props
            counter, speed_estimator = initialize_yolo11_tools(width, height)

            # 2. Process all frames
            frame_idx = 0
            with torch.no_grad():
                while cap.isOpened():
                    # Read frame
                    success, frame = cap.read()
                    if not success:
                        break

                    cv2.imshow("Video Frame", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break


                    # Process every Nth frame based on sample_rate
                    # if frame_idx % sample_rate == 0:
                    result = self._process_single_frame(
                        frame, frame_idx, total_frames,
                        width, height,
                        counter, speed_estimator
                    )


                    # Extract results
                    detections = result.get("detections", [])

                    for det in detections:
                        if "bbox" not in det or "label" not in det:
                            continue

                        x1, y1, x2, y2 = map(int, det["bbox"])
                        label = det["label"]
                        color = (0, 255, 0)  # Green bounding box

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, color, 2)

                        frame_detections[str(frame_idx)] = detections

                        # Update metrics if available
                        if "counter_result" in result and result["counter_result"]:
                            counter_results = result["counter_result"]
                        if "speed_result" in result and result["speed_result"]:
                            speed_results = result["speed_result"]

                    # Increment frame counter
                    frame_idx += 1

            # 3. Create summary from results
            return self._finalize_results(frame_detections, counter_results, speed_results)

        except Exception as e:
            self.logger.error(f"Error processing video: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"error": str(e)}
        finally:
            # Ensure resources are properly released
            if cap is not None:
                cap.release()

    def _process_single_frame(
            self,
            frame: np.ndarray,
            frame_idx: int,
            total_frames: int,
            width: int,
            height: int,
            counter: Any,
            speed_estimator: Any
    ) -> Dict[str, Any]:
        """
        Process a single video frame with both YOLOe and YOLO11.

        Args:
            frame: The video frame to process
            frame_idx: Current frame index
            total_frames: Total number of frames
            width: Video width
            height: Video height
            counter: YOLO11 counter object or None
            speed_estimator: YOLO11 speed estimator object or None

        Returns:
            Dictionary with frame processing results
        """
        result = {
            "detections": [],
            "counter_result": None,
            "speed_result": None
        }

        try:
            self.logger.info(f"Processing frame {frame_idx}/{total_frames}")

            # 1. Run basic object detection with YOLOe
            model_results = self.model(frame, conf=self.config["confidence"])

            # 2. Extract and enhance detections
            detections = extract_detections(model_results)

            if len(detections) > 0:
                # Add position, size and color attributes to detections
                detections = self._enhance_detections_with_attributes(detections, None, frame)

                # Calculate relative positions
                detections = calculate_relative_positions(detections, width, height)
                detections = detect_spatial_relationships(detections)

            result["detections"] = detections

            # 3. Run YOLO11 metrics if available
            if self.has_yolo11_solutions and (counter or speed_estimator):
                try:
                    counter_frame, speed_frame = process_frame_with_yolo11(
                        frame, counter, speed_estimator
                    )
                    result["counter_result"] = counter_frame
                    result["speed_result"] = speed_frame
                except Exception as e:
                    self.logger.warning(f"YOLO11 processing error on frame {frame_idx}: {e}")

        except Exception as e:
            self.logger.warning(f"Error processing frame {frame_idx}: {e}")
            # We return partial results instead of raising to continue processing

        return result

    def _finalize_results(
            self,
            frame_detections: Dict[str, List[Dict[str, Any]]],
            counter_results: Any,
            speed_results: Any
    ) -> Dict[str, Any]:
        """
        Finalize and structure the results with metrics and summaries.

        Args:
            frame_detections: Dictionary mapping frame indices to detections
            counter_results: Results from the YOLO11 counter
            speed_results: Results from the YOLO11 speed estimator

        Returns:
            Structured results dictionary
        """
        # Parse YOLO11 metrics for easier use by LLM
        metrics = {}
        if counter_results:
            metrics["counting"] = parse_solution_results(counter_results)
        if speed_results:
            metrics["speed"] = parse_solution_results(speed_results)

        # Add metrics to result if available
        if metrics:
            frame_detections["metrics"] = metrics
            frame_detections["summary"] = self._create_combined_summary(frame_detections, metrics)
        else:
            # Create basic summary without metrics
            frame_detections["summary"] = self._create_simple_summary(frame_detections)

        return frame_detections

    def _enhance_detections_with_attributes(
        self,
        detections: List[Dict[str, Any]],
        image_path: Optional[str] = None,
        image: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Enhance detections with size and color attributes.

        Args:
            detections: List of detection dictionaries
            image_path: Path to the image (optional)
            image: Direct image array for video frames (optional)

        Returns:
            Detections with added size and color attributes
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
            obj_region = image[int(y1):int(y2), int(x1):int(x2)]

            # Initialize attributes dictionary if not present
            if "attributes" not in det:
                det["attributes"] = {}

            # Calculate size attribute
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

        return detections

    def _create_simple_summary(self, detections) -> Dict[str, Any]:
        """
        Create a simple summary from basic YOLOe detections.

        Args:
            detections: Either a list of detections (for images) or
                        a dictionary of frames with detections (for videos)

        Returns:
            Summary dictionary
        """
        # For a single image (list of detections)
        if isinstance(detections, list):
            # Count objects by type
            counts = {}
            for det in detections:
                label = det["label"]
                if label not in counts:
                    counts[label] = 0
                counts[label] += 1

            return {
                "counts": counts,
                "total_objects": len(detections),
                "object_types": list(counts.keys())
            }

        # For a video (dictionary of frames)
        elif isinstance(detections, dict):
            # Count objects by type across all frames
            counts = {}
            track_ids = set()

            # Go through all frame keys
            for frame_key, frame_detections in detections.items():
                if not frame_key.isdigit():  # Skip non-frame keys
                    continue

                for det in frame_detections:
                    label = det["label"]
                    if label not in counts:
                        counts[label] = 0
                    counts[label] += 1

                    # Track unique objects by track_id
                    if "track_id" in det:
                        track_ids.add((label, det["track_id"]))

            # Count unique objects by type
            unique_by_type = {}
            for label, _ in track_ids:
                if label not in unique_by_type:
                    unique_by_type[label] = 0
                unique_by_type[label] += 1

            return {
                "counts": counts,
                "total_frames_analyzed": len([k for k in detections.keys() if k.isdigit()]),
                "total_detections": sum(counts.values()),
                "unique_tracked_objects": len(track_ids),
                "unique_by_type": unique_by_type,
                "object_types": list(counts.keys())
            }

        # Default empty summary
        return {}

    def _create_combined_summary(self, frame_detections: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a combined summary that merges YOLOe detections with YOLO11 metrics.

        Args:
            frame_detections: Dictionary with frame-by-frame detections
            metrics: Dictionary with YOLO11 metrics

        Returns:
            Combined summary dictionary
        """
        # Get basic summary from YOLOe
        summary = self._create_simple_summary(frame_detections)

        # Add YOLO11 metrics if available
        if metrics:
            if "counting" in metrics:
                counting = metrics["counting"]
                if "summary" in counting:
                    summary["counting_summary"] = counting["summary"]
                if "in_count" in counting and "out_count" in counting:
                    summary["in_count"] = counting["in_count"]
                    summary["out_count"] = counting["out_count"]
                if "class_counts" in counting:
                    summary["class_direction_counts"] = counting["class_counts"]

            if "speed" in metrics:
                speed = metrics["speed"]
                if "total_tracks" in speed:
                    summary["tracked_objects_with_speed"] = speed["total_tracks"]
                if "avg_speed" in speed:
                    summary["average_speed"] = speed["avg_speed"]
                if "class_speeds" in speed:
                    summary["class_speeds"] = speed["class_speeds"]

        return summary