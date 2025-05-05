"""
Enhanced utility functions that use YOLO11 Solutions for visual analysis
The key improvement is integration of object counting, distance calculation, and speed estimation
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


def check_yolo11_solutions_available() -> bool:
    """
    Check if YOLO11 Solutions is available.

    Returns:
        True if available, False otherwise
    """
    try:
        from ultralytics import solutions
        return True
    except (ImportError, AttributeError):
        logger.warning("YOLO11 Solutions not available. Install with: pip install ultralytics>=8.0.0")
        return False


def process_image_with_yolo11(
        image_path: str,
        model_path: str,
        confidence: float = 0.5
) -> Dict[str, Any]:
    """
    Process an image with YOLO11 Solutions, applying counting and distance metrics.

    Args:
        image_path: Path to the image file
        model_path: Path to the YOLO model file
        confidence: Confidence threshold for detections

    Returns:
        Dictionary with metrics from YOLO11 Solutions
    """
    try:
        # Check if YOLO11 Solutions is available
        if not check_yolo11_solutions_available():
            return {"error": "YOLO11 Solutions not available"}

        from ultralytics import solutions

        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            return {"error": f"Failed to read image: {image_path}"}

        height, width = image.shape[:2]

        # Initialize solution objects
        counter = solutions.ObjectCounter(
            model=model_path,
            region=[(0, 0), (width, 0), (width, height), (0, height)],
            classes=[],  # Count all classes
            conf=confidence,
            verbose=False
        )

        distance_calculator = solutions.DistanceCalculation(
            model=model_path,
            conf=confidence,
            verbose=False
        )

        # Process with solutions
        counter_results = counter.predict(image)
        distance_results = distance_calculator.predict(image)

        # Extract metrics
        metrics = {
            "counting": counter_results,
            "distance": distance_results
        }

        return metrics

    except Exception as e:
        logger.error(f"Error processing image with YOLO11: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}


def process_video_with_yolo11(
        video_path: str,
        model_path: str,
        sample_rate: int = 5,
        confidence: float = 0.5
) -> Dict[str, Any]:
    """
    Process a video with YOLO11 Solutions, applying counting, distance, and speed metrics.

    Args:
        video_path: Path to the video file
        model_path: Path to the YOLO model file
        sample_rate: Process every Nth frame
        confidence: Confidence threshold for detections

    Returns:
        Dictionary with metrics from YOLO11 Solutions
    """
    try:
        # Check if YOLO11 Solutions is available
        if not check_yolo11_solutions_available():
            return {"error": "YOLO11 Solutions not available"}

        from ultralytics import solutions

        # Open video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": f"Failed to open video: {video_path}"}

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0  # Default FPS

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize solution objects
        counter = solutions.ObjectCounter(
            model=model_path,
            region=[(0, 0), (width, 0), (width, height), (0, height)],
            verbose=False,
            show=False
        )

        speed_estimator = solutions.SpeedEstimator(
            model=model_path,
            verbose=False,
            show=False,
        )

        # Initialize results
        counter_results = None
        speed_results = None
        frame_idx = 0

        # Process key frames
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Process at sample rate
            if frame_idx % sample_rate == 0:
                logger.info(f"Processing frame {frame_idx}/{total_frames} for metrics")

                    # Update counter (keeps track of objects across frames)
                counter_frame = counter(frame)
                counter_results = counter_frame

                # Update speed estimator
                speed_frame = speed_estimator(frame)
                speed_results = speed_frame

            frame_idx += 1

        # Release video capture
        cap.release()

        # Compile metrics
        metrics = {
            "counting": counter_results,
            "speed": speed_results,
            "frames_processed": frame_idx // sample_rate
        }
        print(f"Metrics: {metrics}")
        return metrics

    except Exception as e:
        logger.error(f"Error processing video with YOLO11: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}


def combine_metrics(basic_detections: Dict[str, List[Dict[str, Any]]], yolo11_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine basic YOLO detections with advanced YOLO11 metrics.

    Args:
        basic_detections: Dictionary with basic detection results
        yolo11_metrics: Dictionary with YOLO11 metrics

    Returns:
        Combined metrics dictionary
    """
    combined = {}

    # Count objects by type from basic detections
    counts = {}
    track_ids = set()

    for frame_key, detections in basic_detections.items():
        if not frame_key.isdigit():  # Skip non-frame keys like "metrics"
            continue

        for det in detections:
            label = det["label"]
            if label not in counts:
                counts[label] = 0
            counts[label] += 1

            # Track unique objects by track_id
            if "track_id" in det:
                track_ids.add((label, det["track_id"]))

    # Add basic detection counts
    combined["counts"] = counts
    combined["total_objects"] = sum(counts.values())
    combined["unique_tracked_objects"] = len(track_ids)

    # Add YOLO11 metrics if available
    if yolo11_metrics:
        if "counting" in yolo11_metrics:
            combined["yolo11_counts"] = yolo11_metrics["counting"]

        if "speed" in yolo11_metrics:
            combined["speed_data"] = yolo11_metrics["speed"]

        if "distance" in yolo11_metrics:
            combined["distance_data"] = yolo11_metrics["distance"]

    return combined