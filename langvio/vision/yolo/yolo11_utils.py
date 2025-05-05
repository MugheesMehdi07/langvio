"""
Enhanced utility functions that use YOLO11 Solutions for video processing
The key improvement is processing complete videos frame by frame
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


def process_video_with_yolo11(
        video_path: str,
        model_path: str,
        sample_rate: int = 5,
        confidence: float = 0.5
) -> Dict[str, Any]:
    """
    Process an entire video with YOLO11 Solutions, applying counting, distance, and speed metrics
    to each frame.

    Args:
        video_path: Path to the video file
        model_path: Path to the YOLO model file
        sample_rate: Process every Nth frame
        confidence: Confidence threshold for detections

    Returns:
        Dictionary with combined metrics and frame-by-frame detections
    """
    try:
        from ultralytics import YOLO, solutions

        # Check if YOLO11 Solutions is available
        if not check_yolo11_solutions_available():
            return {"error": "YOLO11 Solutions not available"}

        # Load model
        # model = YOLO(model_path)

        # Open video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": f"Failed to open video: {video_path}"}

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0  # Default FPS if not available

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize solution objects (create once, reuse for all frames)
        counter = solutions.ObjectCounter(
            model=model_path,
            region=[(0, 0), (width, 0), (width, height), (0, height)],
            verbose=False,
            show=False

        )

        distance_calculator = solutions.DistanceCalculation(
            model=model_path,
            verbose=False,
            show=False

        )

        speed_estimator = solutions.SpeedEstimator(
            model=model_path,
            verbose=False,
            show=False

        )

        # Initialize results storage
        frame_detections = {}
        all_metrics = {
            "counting": {},
            "distance": {},
            "speed": {}
        }

        # Create object trackers - these will be maintained across all frames
        track_objects = {}
        next_track_id = 0

        # Process frames
        frame_idx = 0

        while cap.isOpened() and frame_idx < total_frames :
            # Read frame
            success, frame = cap.read()
            if not success:
                break

            # Process every Nth frame based on sample_rate
            if frame_idx % sample_rate == 0:
                logger.info(f"Processing frame {frame_idx}/{total_frames}")

            #counter
            results_count = counter(frame)
            print(results_count)

            # results_distance = distance_calculator(frame)
            # print(results_distance)

            results_speed = speed_estimator(frame)
            print(results_speed)
            frame_idx += 1

        # Release video capture
        cap.release()

        all_metrics["counting"] = results_count
        # all_metrics["distance"] = results_distance
        all_metrics["speed"] = results_speed
        print(all_metrics)

        # Generate summary metrics
        # summary = create_summary_metrics(frame_detections, all_metrics)
        # Add summary to frame detections
        # frame_detections["summary"] = summary

        return frame_detections

    except Exception as e:
        logger.error(f"Error processing video with YOLO11: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}