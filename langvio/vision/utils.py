"""
Enhanced utilities for vision processing
"""

from typing import Any, Dict, List


# from langvio.prompts.constants import (
#     VISUAL_ATTRIBUTES,
#     SPATIAL_RELATIONS,
#     ACTIVITIES,
#     DEFAULT_IOU_THRESHOLD
# )
# Add to langvio/vision/yolo11_utils.py

def optimize_for_memory():
    """
    Set PyTorch memory optimization settings.
    """
    import os
    import torch

    # Set environment variables to reduce memory fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Enable memory caching to reduce allocations
    torch.cuda.empty_cache()

    # Set to use TF32 precision if available (for Ampere and later GPUs)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Set inference mode for PyTorch
    torch.set_grad_enabled(False)

def extract_detections(results) -> List[Dict[str, Any]]:
    """
    Extract detections from YOLO results with enhanced attributes.

    Args:
        results: Raw YOLO results

    Returns:
        List of detection dictionaries with enhanced attributes
    """
    detections = []

    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = result.names[cls_id]

            # Calculate center point and dimensions for spatial relation analysis
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            area = width * height

            # Enhanced detection object with additional fields
            detections.append(
                {
                    "label": label,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "class_id": cls_id,
                    # Additional attributes for spatial and attribute analysis
                    "center": (center_x, center_y),
                    "dimensions": (width, height),
                    "area": area,
                    "relative_size": None,  # Will be calculated later based on image dimensions
                    "attributes": {},  # Placeholder for detected attributes
                    "activities": [],  # Placeholder for detected activities (for videos)
                    "relationships": [],  # Will be populated during relationship analysis
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
        # Calculate relative size compared to image
        det["relative_size"] = det["area"] / image_area

        # Calculate relative positions (0-1)
        center_x, center_y = det["center"]
        det["relative_position"] = (center_x / image_width, center_y / image_height)

        # Classify position in image (top-left, center, etc.)
        rx, ry = det["relative_position"]
        position = ""

        # Vertical position
        if ry < 0.33:
            position += "top-"
        elif ry < 0.66:
            position += "middle-"
        else:
            position += "bottom-"

        # Horizontal position
        if rx < 0.33:
            position += "left"
        elif rx < 0.66:
            position += "center"
        else:
            position += "right"

        det["position_area"] = position

    return detections


def detect_spatial_relationships(
    detections: List[Dict[str, Any]], distance_threshold: float = 0.2
) -> List[Dict[str, Any]]:
    """
    Detect spatial relationships between objects.

    Args:
        detections: List of detection dictionaries
        distance_threshold: Threshold for 'near' relationship (as fraction of image width)

    Returns:
        Updated list of detections with relationship information
    """
    if len(detections) < 2:
        return detections

    for i, det1 in enumerate(detections):
        for j, det2 in enumerate(detections):
            if i == j:
                continue

            # Get centers and boxes
            center1_x, center1_y = det1["center"]
            center2_x, center2_y = det2["center"]
            box1 = det1["bbox"]
            box2 = det2["bbox"]

            # Initialize relationship entry
            relationship = {"object": det2["label"], "object_id": j, "relations": []}

            # Check left/right
            if center1_x < center2_x:
                relationship["relations"].append("left_of")
            else:
                relationship["relations"].append("right_of")

            # Check above/below
            if center1_y < center2_y:
                relationship["relations"].append("above")
            else:
                relationship["relations"].append("below")

            # Check near
            distance = (
                (center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2
            ) ** 0.5
            if distance < distance_threshold * (
                det1["dimensions"][0] + det2["dimensions"][0]
            ):
                relationship["relations"].append("near")
            else:
                relationship["relations"].append("far")

            # Check inside/containing
            x1_1, y1_1, x2_1, y2_1 = box1
            x1_2, y1_2, x2_2, y2_2 = box2

            if x1_1 > x1_2 and y1_1 > y1_2 and x2_1 < x2_2 and y2_1 < y2_2:
                relationship["relations"].append("inside")
            elif x1_2 > x1_1 and y1_2 > y1_1 and x2_2 < x2_1 and y2_2 < y2_1:
                relationship["relations"].append("contains")

            # Add the relationship to the detection
            det1["relationships"].append(relationship)

    return detections




def filter_by_attributes(
    detections: List[Dict[str, Any]], required_attributes: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Filter detections by required attributes.

    Args:
        detections: List of detection dictionaries
        required_attributes: List of required attribute dictionaries
                            (e.g. [{"attribute": "color", "value": "red"}])

    Returns:
        Filtered list of detections
    """
    if not required_attributes:
        return detections

    filtered = []

    for det in detections:
        matches_all = True

        for req in required_attributes:
            attr_name = req.get("attribute")
            attr_value = req.get("value")

            # Skip if the attribute isn't specified
            if not attr_name or not attr_value:
                continue

            # Check if the detection has this attribute with matching value
            if (
                attr_name not in det["attributes"]
                or det["attributes"][attr_name] != attr_value
            ):
                matches_all = False
                break

        if matches_all:
            filtered.append(det)

    return filtered


def filter_by_spatial_relations(
    detections: List[Dict[str, Any]], required_relations: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Filter detections by required spatial relationships.

    Args:
        detections: List of detection dictionaries
        required_relations: List of required relationship dictionaries
                           (e.g. [{"relation": "above", "object": "table"}])

    Returns:
        Filtered list of detections
    """
    if not required_relations:
        return detections

    filtered = []

    for det in detections:
        should_include = True

        for req_rel in required_relations:
            relation_type = req_rel.get("relation")
            target_object = req_rel.get("object")

            # Skip if relation or target object isn't specified
            if not relation_type or not target_object:
                continue

            # Check if this detection has the required relationship
            has_relation = False

            for rel in det["relationships"]:
                if (
                    rel["object"].lower() == target_object.lower()
                    and relation_type in rel["relations"]
                ):
                    has_relation = True
                    break

            if not has_relation:
                should_include = False
                break

        if should_include:
            filtered.append(det)

    return filtered


def filter_by_activities(
    detections: List[Dict[str, Any]], required_activities: List[str]
) -> List[Dict[str, Any]]:
    """
    Filter detections by required activities.

    Args:
        detections: List of detection dictionaries
        required_activities: List of required activities (e.g. ["walking", "running"])

    Returns:
        Filtered list of detections
    """
    if not required_activities:
        return detections

    filtered = []

    for det in detections:
        for activity in required_activities:
            if activity.lower() in [a.lower() for a in det["activities"]]:
                filtered.append(det)
                break

    return filtered
