"""
Spatial relationship analysis utilities
"""

from typing import Any, Dict, List


def add_spatial_relationships(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add spatial relationships between objects.

    Args:
        detections: List of detection dictionaries with center points

    Returns:
        Updated detections with relationship information
    """
    for i, det1 in enumerate(detections):
        if "center" not in det1:
            continue

        det1["relationships"] = []
        center1_x, center1_y = det1["center"]

        for j, det2 in enumerate(detections):
            if i == j or "center" not in det2:
                continue

            center2_x, center2_y = det2["center"]
            relations = []

            # Basic directional relationships
            relations.append("left_of" if center1_x < center2_x else "right_of")
            relations.append("above" if center1_y < center2_y else "below")

            # Distance relationship
            distance = (
                (center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2
            ) ** 0.5
            relations.append("near" if distance < 100 else "far")

            # Check containment if bboxes available
            if "bbox" in det1 and "bbox" in det2:
                x1_1, y1_1, x2_1, y2_1 = det1["bbox"]
                x1_2, y1_2, x2_2, y2_2 = det2["bbox"]

                if x1_1 > x1_2 and y1_1 > y1_2 and x2_1 < x2_2 and y2_1 < y2_2:
                    relations.append("inside")
                elif x1_2 > x1_1 and y1_2 > y1_1 and x2_2 < x2_1 and y2_2 < y2_1:
                    relations.append("contains")

            det1["relationships"].append(
                {
                    "object": det2["label"],
                    "object_id": det2.get("object_id", f"obj_{j}"),
                    "relations": relations,
                }
            )

    return detections


def calculate_relative_positions(
    detections: List[Dict[str, Any]], image_width: int, image_height: int
) -> List[Dict[str, Any]]:
    """
    Calculate relative positions and sizes of detections.

    Args:
        detections: List of detection dictionaries
        image_width: Width of the image
        image_height: Height of the image

    Returns:
        Updated list of detections with relative position information
    """
    image_area = image_width * image_height

    for det in detections:
        if "bbox" in det:
            x1, y1, x2, y2 = det["bbox"]
            area = (x2 - x1) * (y2 - y1)
            det["relative_size"] = area / image_area

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            det["center"] = (center_x, center_y)
            det["relative_position"] = (center_x / image_width, center_y / image_height)

            rx, ry = det["relative_position"]
            position = ""

            if ry < 0.33:
                position += "top-"
            elif ry < 0.66:
                position += "middle-"
            else:
                position += "bottom-"

            if rx < 0.33:
                position += "left"
            elif rx < 0.66:
                position += "center"
            else:
                position += "right"

            det["position_area"] = position

    return detections


def detect_spatial_relationships(  # noqa: C901
    detections: List[Dict[str, Any]], distance_threshold: float = 0.2
) -> List[Dict[str, Any]]:
    """
    Detect spatial relationships between objects.

    Args:
        detections: List of detection dictionaries
        distance_threshold: Threshold for 'near' relationship
        (as fraction of image width)

    Returns:
        Updated list of detections with relationship information
    """
    if len(detections) < 2:
        return detections

    for i, det1 in enumerate(detections):
        for j, det2 in enumerate(detections):
            if i == j:
                continue

            center1_x, center1_y = det1["center"]
            center2_x, center2_y = det2["center"]
            box1 = det1["bbox"]
            box2 = det2["bbox"]

            relationship = {
                "object": det2["label"],
                "object_id": j,
                "relations": [],
            }

            if center1_x < center2_x:
                relationship["relations"].append("left_of")
            else:
                relationship["relations"].append("right_of")

            if center1_y < center2_y:
                relationship["relations"].append("above")
            else:
                relationship["relations"].append("below")

            distance = (
                (center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2
            ) ** 0.5

            if distance < distance_threshold * (
                det1.get("dimensions", [100, 100])[0]
                + det2.get("dimensions", [100, 100])[0]
            ):
                relationship["relations"].append("near")
            else:
                relationship["relations"].append("far")

            x1_1, y1_1, x2_1, y2_1 = box1
            x1_2, y1_2, x2_2, y2_2 = box2

            if x1_1 > x1_2 and y1_1 > y1_2 and x2_1 < x2_2 and y2_1 < y2_2:
                relationship["relations"].append("inside")
            elif x1_2 > x1_1 and y1_2 > y1_1 and x2_2 < x2_1 and y2_2 < y2_1:
                relationship["relations"].append("contains")

            if "relationships" not in det1:
                det1["relationships"] = []
            det1["relationships"].append(relationship)

    return detections
