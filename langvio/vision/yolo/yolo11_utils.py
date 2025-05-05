# langvio/vision/yolo11_utils.py

"""
Utility functions that use YOLO11 Solutions for enhanced detection metrics
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

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


def process_counting(
        frame: np.ndarray,
        model: Any,
        detections: List[Dict[str, Any]],
        target_classes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Process frame with YOLO11 Solutions ObjectCounter.

    Args:
        frame: Input frame as numpy array
        model: YOLO model instance
        detections: Detections for the frame
        target_classes: Optional list of target classes to count

    Returns:
        Dictionary with counting metrics
    """
    try:
        from ultralytics import solutions

        # Define region (full frame)
        h, w = frame.shape[:2]
        region = [(0, 0), (w, 0), (w, h), (0, h)]

        # Convert target classes to class indices if provided
        class_ids = None
        if target_classes:
            class_ids = []
            for target in target_classes:
                # Find index for this class name
                if target in model.names.values():
                    for idx, name in model.names.items():
                        if name == target:
                            class_ids.append(idx)
                            break

        # Initialize and use counter
        counter = solutions.ObjectCounter(
            model=model,
            region=region,
            classes=class_ids,
            verbose=False
        )

        # Process frame
        results = counter(frame)

        # Update detections with "counted" attribute
        for det in detections:
            if "attributes" not in det:
                det["attributes"] = {}
            det["attributes"]["counted"] = True

        # Extract metrics
        metrics = {
            "count_in": getattr(results, "in_count", 0),
            "count_out": getattr(results, "out_count", 0),
            "count_total": getattr(results, "in_count", 0) + getattr(results, "out_count", 0)
        }

        logger.info(f"YOLO11 counting metrics: {metrics}")
        return metrics

    except Exception as e:
        logger.error(f"Error in YOLO11 counting: {e}")
        return {}


def process_distances(
        frame: np.ndarray,
        model: Any,
        detections: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Process frame with YOLO11 Solutions DistanceCalculator.

    Args:
        frame: Input frame as numpy array
        model: YOLO model instance
        detections: Detections for the frame

    Returns:
        Dictionary with distance metrics
    """
    try:
        from ultralytics import solutions

        # Initialize and use distance calculator
        distance_calculator = solutions.DistanceCalculator(
            model=model,
            verbose=False
        )

        # Process frame
        results = distance_calculator(frame)

        # Extract distances and update detections
        pairs = []

        # Check if pairs exist in results
        if hasattr(results, "pairs"):
            for pair in results.pairs:
                # Extract pair data
                id1 = getattr(pair, "id1", None)
                id2 = getattr(pair, "id2", None)
                coord1 = getattr(pair, "coord1", (0, 0))
                coord2 = getattr(pair, "coord2", (0, 0))
                distance = getattr(pair, "distance", 0)

                # Find corresponding detections
                det1_idx = None
                det2_idx = None
                obj1_type = None
                obj2_type = None

                # First try to match by track_id if available
                for i, det in enumerate(detections):
                    if "track_id" in det and det["track_id"] == id1:
                        det1_idx = i
                        obj1_type = det["label"]
                    if "track_id" in det and det["track_id"] == id2:
                        det2_idx = i
                        obj2_type = det["label"]

                # If not found by track_id, try to match by coordinates
                if det1_idx is None or det2_idx is None:
                    for i, det in enumerate(detections):
                        if "center" in det:
                            center = det["center"]
                            # Match with some tolerance
                            if (abs(center[0] - coord1[0]) < 10 and
                                    abs(center[1] - coord1[1]) < 10):
                                det1_idx = i
                                obj1_type = det["label"]
                            if (abs(center[0] - coord2[0]) < 10 and
                                    abs(center[1] - coord2[1]) < 10):
                                det2_idx = i
                                obj2_type = det["label"]

                # Add relationship to detections if found
                if det1_idx is not None and det2_idx is not None:
                    # Add relationship to first detection
                    if "relationships" not in detections[det1_idx]:
                        detections[det1_idx]["relationships"] = []

                    detections[det1_idx]["relationships"].append({
                        "object": detections[det2_idx]["label"],
                        "object_id": det2_idx,
                        "relations": ["distance"],
                        "distance": distance
                    })

                    # Add relationship to second detection
                    if "relationships" not in detections[det2_idx]:
                        detections[det2_idx]["relationships"] = []

                    detections[det2_idx]["relationships"].append({
                        "object": detections[det1_idx]["label"],
                        "object_id": det1_idx,
                        "relations": ["distance"],
                        "distance": distance
                    })

                # Add to pairs list with object types if available
                pairs.append({
                    "object1": obj1_type if obj1_type else f"Object {id1}",
                    "object2": obj2_type if obj2_type else f"Object {id2}",
                    "distance": distance
                })

        # Extract metrics
        metrics = {
            "distance_pairs": pairs,
            "total_pairs": len(pairs)
        }

        logger.info(f"YOLO11 distance processing completed: {len(pairs)} pairs found")
        return metrics

    except Exception as e:
        logger.error(f"Error in YOLO11 distance calculation: {e}")
        return {}


def process_speeds(
        frame: np.ndarray,
        model: Any,
        detections: List[Dict[str, Any]],
        fps: float = 25.0
) -> Dict[str, Any]:
    """
    Process frame with YOLO11 Solutions SpeedEstimator.

    Args:
        frame: Input frame as numpy array
        model: YOLO model instance
        detections: Detections for the frame
        fps: Frames per second for the video

    Returns:
        Dictionary with speed metrics
    """
    try:
        from ultralytics import solutions

        # Initialize and use speed estimator
        speed_estimator = solutions.SpeedEstimator(
            model=model,
            verbose=False
        )

        # Process frame
        results = speed_estimator(frame)

        # Extract speed data and update detections
        speeds = []

        if hasattr(results, "objects"):
            for obj in results.objects:
                # Extract object data
                obj_id = getattr(obj, "id", None)
                speed = getattr(obj, "speed", 0)

                # Find matching detection
                for det in detections:
                    if "track_id" in det and det["track_id"] == obj_id:
                        # Add speed to detection
                        if "attributes" not in det:
                            det["attributes"] = {}
                        det["attributes"]["speed"] = f"{speed:.2f} pixels/sec"

                        # Add activity based on speed
                        if "activities" not in det:
                            det["activities"] = []

                        if speed < 5:
                            if "stationary" not in det["activities"]:
                                det["activities"].append("stationary")
                        elif speed < 30:
                            if "moving_slowly" not in det["activities"]:
                                det["activities"].append("moving_slowly")
                        else:
                            if "moving_quickly" not in det["activities"]:
                                det["activities"].append("moving_quickly")

                        # Add to speeds list
                        speeds.append({
                            "object_id": obj_id,
                            "object_type": det["label"],
                            "speed": speed
                        })
                        break

        # Calculate average speeds by object type
        speed_by_type = {}

        for speed_info in speeds:
            obj_type = speed_info["object_type"]
            if obj_type not in speed_by_type:
                speed_by_type[obj_type] = []
            speed_by_type[obj_type].append(speed_info["speed"])

        # Calculate averages
        avg_speed_by_class = {}
        for obj_type, obj_speeds in speed_by_type.items():
            if obj_speeds:
                avg_speed_by_class[obj_type] = sum(obj_speeds) / len(obj_speeds)

        # Extract metrics
        metrics = {
            "speeds": speeds,
            "avg_speeds": avg_speed_by_class
        }

        logger.info(f"YOLO11 speed estimation completed: {len(speeds)} objects tracked")
        return metrics

    except Exception as e:
        logger.error(f"Error in YOLO11 speed estimation: {e}")
        return {}


def get_fps_from_video(video_path: str) -> float:
    """
    Get frames per second from a video file.

    Args:
        video_path: Path to the video file

    Returns:
        Frames per second (defaults to 25.0 if cannot be determined)
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            if fps <= 0:
                return 25.0  # Default if detection fails
            return fps
        else:
            return 25.0  # Default if cannot open video
    except Exception:
        return 25.0  # Default on error


def combine_metrics(
        frame_detections: Dict[str, List[Dict[str, Any]]],
        metrics_list: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Combine all metrics into a single summary.

    Args:
        frame_detections: Dictionary with frame detections
        metrics_list: List of metrics dictionaries

    Returns:
        Combined metrics summary
    """
    # Start with basic counting metrics
    # Count by class
    count_by_class = {}
    for frame_key, detections in frame_detections.items():
        if not frame_key.isdigit():
            continue

        for det in detections:
            label = det["label"]
            if label not in count_by_class:
                count_by_class[label] = 0
            count_by_class[label] += 1

    # Track unique objects
    unique_tracks = {}
    for frame_key, detections in frame_detections.items():
        if not frame_key.isdigit():
            continue

        for det in detections:
            if "track_id" in det:
                label = det["label"]
                if label not in unique_tracks:
                    unique_tracks[label] = set()
                unique_tracks[label].add(det["track_id"])

    # Build summary dictionary
    summary = {
        "counts": count_by_class
    }

    # Add unique object counts if available
    if unique_tracks:
        summary["unique_objects"] = {label: len(tracks) for label, tracks in unique_tracks.items()}
        summary["total_unique_objects"] = sum(len(tracks) for tracks in unique_tracks.values())

    # Add all metrics from the list
    for metrics in metrics_list:
        # Add each metric, avoiding duplicates
        for key, value in metrics.items():
            # Don't overwrite counts if we already have them
            if key == "counts" and "counts" in summary:
                continue

            summary[key] = value

    return summary