"""
Utility functions for vision processing
"""

from typing import List, Dict, Any, Tuple, Optional


def add_spatial_context(
    detections: List[Dict[str, Any]], dimensions: Optional[Tuple[int, int]]
) -> List[Dict[str, Any]]:
    """
    Add spatial context to detections (positions and relationships).

    Args:
        detections: List of detection dictionaries
        dimensions: Optional tuple of (width, height)

    Returns:
        Enhanced detections with spatial context
    """
    # Skip if no dimensions provided
    if not dimensions or not detections:
        return detections

    # Calculate relative positions
    from langvio.vision.utils import (
        calculate_relative_positions,
        detect_spatial_relationships,
    )

    # Add relative positions based on image dimensions
    detections = calculate_relative_positions(detections, *dimensions)

    # Add spatial relationships between objects
    detections = detect_spatial_relationships(detections)

    return detections


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
