"""
YOLO11 utilities for object counting, speed estimation, and more
"""

import logging
from typing import Any, Dict, Optional, Tuple

from langvio.prompts.constants import YOLO11_CONFIG

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


def initialize_yolo11_tools(width: int, height: int) -> Tuple[Any, Any]:
    """
    Initialize YOLO11 Solutions tools if available.

    Args:
        width: Video width
        height: Video height

    Returns:
        Tuple containing (counter, speed_estimator) - may be None if not available
    """
    counter = None
    speed_estimator = None
    has_yolo11_solutions=False

    # Check if YOLO11 Solutions are available
    has_yolo11_solutions = check_yolo11_solutions_available()

    if has_yolo11_solutions:
        try:
            yolo11_config = YOLO11_CONFIG
            # Create counter with region covering the full frame
            counter = create_object_counter(
                model_path=yolo11_config["model_path"],
                confidence=yolo11_config["confidence"],
                region=[(0, 0), (width, 0), (width, height), (0, height)]
            )

            # Create speed estimator
            speed_estimator = create_speed_estimator(
                model_path=yolo11_config["model_path"],
                confidence=yolo11_config["confidence"],
                region_width=width
            )
        except Exception as e:
            logger.warning(f"Failed to initialize YOLO11 tools: {e}")
            has_yolo11_solutions = False

    return counter, speed_estimator

def create_object_counter(model_path: str, confidence: float, region: Optional[list] = None):
    """
    Create a YOLO11 ObjectCounter.

    Args:
        model_path: Path to the YOLO model
        confidence: Confidence threshold
        region: Optional counting region coordinates [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]

    Returns:
        ObjectCounter object or None if not available
    """
    if not check_yolo11_solutions_available():
        return None

    try:
        from ultralytics import solutions

        if region is None:
            # Default region covers entire frame and will be adjusted in first frame
            region = [(0, 0), (100, 0), (100, 100), (0, 100)]

        counter = solutions.ObjectCounter(
            model=model_path,
            region=region,
            show=False
        )

        return counter
    except Exception as e:
        logger.error(f"Error creating object counter: {e}")
        return None


def create_speed_estimator(model_path: str, confidence: float, region_width: Optional[int] = None):
    """
    Create a YOLO11 SpeedEstimator.

    Args:
        model_path: Path to the YOLO model
        confidence: Confidence threshold
        region_width: Optional width of the region in pixels

    Returns:
        SpeedEstimator object or None if not available
    """
    if not check_yolo11_solutions_available():
        return None

    try:
        from ultralytics import solutions

        # Create speed estimator
        speed_estimator = solutions.SpeedEstimator(
            model=model_path,
            conf=confidence,
            show=False,
            verbose=False
        )

        return speed_estimator
    except Exception as e:
        logger.error(f"Error creating speed estimator: {e}")
        return None


def process_frame_with_yolo11(frame, counter, speed_estimator):
    """
    Process a single frame with YOLO11 Solutions.

    Args:
        frame: CV2 frame
        counter: Object counter instance
        speed_estimator: Speed estimator instance

    Returns:
        Tuple of (counter_results, speed_results)
    """
    counter_results = None
    speed_results = None

    try:
        # Update counter if available
        if counter:
            counter_results = counter(frame.copy())

        # Update speed estimator if available
        if speed_estimator:
            speed_results = speed_estimator(frame.copy())

    except Exception as e:
        logger.error(f"Error processing frame with YOLO11: {e}")

    return counter_results, speed_results


def parse_solutions_results(counter_results: Any, speed_results: Any) -> Dict[str, Any]:
    """
    Enhanced parsing of YOLO11 results with better structure for GPT.

    Args:
        counter_results: Object counting results
        speed_results: Speed estimation results

    Returns:
        Dictionary with structured YOLO11 metrics
    """
    metrics = {}

    # Parse counting results
    if counter_results and hasattr(counter_results, 'in_count'):
        counting_data = {
            "total_objects_entered": counter_results.in_count,
            "total_objects_exited": counter_results.out_count,
            "net_objects_flow": counter_results.in_count - counter_results.out_count,
            "total_crossings": counter_results.in_count + counter_results.out_count
        }

        # Add class-wise counts if available
        if hasattr(counter_results, 'classwise_count'):
            class_breakdown = {}
            for class_name, directions in counter_results.classwise_count.items():
                if directions["IN"] > 0 or directions["OUT"] > 0:
                    class_breakdown[class_name] = {
                        "entered": directions["IN"],
                        "exited": directions["OUT"],
                        "net_flow": directions["IN"] - directions["OUT"]
                    }

            if class_breakdown:
                counting_data["by_object_type"] = class_breakdown

        metrics["counting"] = counting_data

    # Parse speed results
    if speed_results and hasattr(speed_results, 'total_tracks'):
        speed_data = {
            "objects_with_speed_data": speed_results.total_tracks
        }

        # Add average speed if available
        if hasattr(speed_results, 'avg_speed') and speed_results.avg_speed:
            speed_data["average_speed_kmh"] = round(speed_results.avg_speed, 1)

        # Add class-wise speeds if available
        if hasattr(speed_results, 'class_speeds'):
            speed_by_class = {}
            for class_name, speeds in speed_results.class_speeds.items():
                if speeds:  # Only include classes with speed data
                    avg_speed = sum(speeds) / len(speeds) if speeds else 0
                    speed_by_class[class_name] = {
                        "average_speed_kmh": round(avg_speed, 1),
                        "sample_count": len(speeds),
                        "speed_range": {
                            "min": round(min(speeds), 1),
                            "max": round(max(speeds), 1)
                        } if len(speeds) > 1 else None
                    }

            if speed_by_class:
                speed_data["by_object_type"] = speed_by_class

        metrics["speed_analysis"] = speed_data

    return metrics