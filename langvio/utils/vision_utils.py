"""
Utility functions for vision processing
"""

from typing import Any, Dict, List, Optional, Tuple

"""
Updated utility functions for vision processing
"""

from typing import Any, Dict, List, Optional, Tuple


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


# Updated to work with highlighting
def create_visualization_detections_for_video(
    all_detections: Dict[str, List[Dict[str, Any]]],
    highlight_objects: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Create a filtered detections dictionary for video visualization.
    NOTE: This function is now mainly kept for backward compatibility.
    With the new visualization approach, we use all_detections directly.

    Args:
        all_detections: Dictionary with all detection results
        highlight_objects: List of directly referenced objects to highlight

    Returns:
        Dictionary with filtered detection results for visualization
    """
    # With the new approach, we return all detections
    # The highlighting is done by checking against highlight_objects in the visualization
    return all_detections


# Updated to work with highlighting
def create_visualization_detections_for_image(
    highlight_objects: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Create a list of detections for image visualization.
    NOTE: This function is now mainly kept for backward compatibility.
    With the new visualization approach, we keep all detections.

    Args:
        highlight_objects: List of directly referenced objects to highlight

    Returns:
        List of detection objects for highlighting
    """
    # Extract detection objects from highlight_objects
    return [
        obj_info.get("detection")
        for obj_info in highlight_objects
        if obj_info.get("detection") is not None
    ]