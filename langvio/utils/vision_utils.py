"""
Utility functions for vision processing
"""

from typing import Any, Dict, List, Optional, Tuple



def create_visualization_detections_for_video(
    all_detections: Dict[str, List[Dict[str, Any]]],
    highlight_objects: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Create a filtered detections dictionary for video visualization.

    Args:
        all_detections: Dictionary with all detection results
        highlight_objects: List of directly referenced objects to highlight

    Returns:
        Dictionary with filtered detection results for visualization
    """
    visualization_detections = {frame_key: [] for frame_key in all_detections.keys()}

    # Process each directly referenced object
    for obj_info in highlight_objects:
        frame_key = obj_info.get("frame_key")
        detection = obj_info.get("detection")

        if frame_key and detection and frame_key in visualization_detections:
            visualization_detections[frame_key].append(detection)

    return visualization_detections


def create_visualization_detections_for_image(
    highlight_objects: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Create a list of detections for image visualization.

    Args:
        highlight_objects: List of directly referenced objects to highlight

    Returns:
        List of detection objects for visualization
    """
    return [
        obj_info.get("detection")
        for obj_info in highlight_objects
        if obj_info.get("detection") is not None
    ]
