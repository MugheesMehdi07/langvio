"""
Enhanced utilities for vision processing
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch


def optimize_for_memory():
    """
    Set PyTorch memory optimization settings.
    """
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
    Extract detections from YOLO results with basic attributes.

    Args:
        results: Raw YOLO results

    Returns:
        List of detection dictionaries with basic attributes
    """
    detections = []

    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = result.names[cls_id]

            # Basic detection object
            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "class_id": cls_id,
            })

    return detections


def add_unified_attributes(
        detections: List[Dict[str, Any]],
        width: int,
        height: int,
        input_data: Any,  # image_path (str) or frame (np.ndarray)
        needs_color: bool,
        needs_spatial: bool,
        needs_size: bool,
        is_video_frame: bool
) -> List[Dict[str, Any]]:
    """
    Unified method to add attributes to detections.
    Works for both images and video frames.

    Args:
        detections: List of detection dictionaries
        width: Image/frame width
        height: Image/frame height
        input_data: Image path (str) for images, frame array for videos
        needs_color: Whether to include color detection
        needs_spatial: Whether to include spatial relationships
        needs_size: Whether to include size attributes
        is_video_frame: Whether this is a video frame

    Returns:
        Enhanced detections with requested attributes
    """
    if not detections:
        return detections

    # Get image data for color detection if needed
    image_data = None
    if needs_color:
        if is_video_frame:
            image_data = input_data  # input_data is already a frame
        else:
            try:
                image_data = cv2.imread(input_data)  # input_data is image path
            except Exception:
                pass

    # Process each detection
    enhanced_detections = []
    for i, det in enumerate(detections):
        if "bbox" not in det:
            enhanced_detections.append(det)
            continue

        x1, y1, x2, y2 = det["bbox"]

        # Skip invalid boxes
        if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            enhanced_detections.append(det)
            continue

        # Add basic position info (always needed for tracking)
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        det["center"] = (center_x, center_y)
        det["object_id"] = f"obj_{i}"

        # Initialize attributes
        attributes = {}

        # Add size attributes if needed
        if needs_size:
            area = (x2 - x1) * (y2 - y1)
            relative_size = area / (width * height)
            attributes["size"] = "small" if relative_size < 0.05 else "medium" if relative_size < 0.25 else "large"
            attributes["relative_size"] = relative_size

        # Add position attributes if needed (for spatial queries)
        if needs_spatial:
            rx, ry = center_x / width, center_y / height
            pos_v = "top" if ry < 0.33 else "middle" if ry < 0.66 else "bottom"
            pos_h = "left" if rx < 0.33 else "center" if rx < 0.66 else "right"
            attributes["position"] = f"{pos_v}-{pos_h}"

            # Add relative position for advanced spatial analysis
            det["relative_position"] = (rx, ry)

        # Add color attributes if needed (expensive)
        if needs_color and image_data is not None:
            try:
                obj_region = image_data[int(y1):int(y2), int(x1):int(x2)]
                if obj_region.size > 0:
                    from langvio.vision.color_detection import ColorDetector
                    color_info = ColorDetector.get_color_profile(obj_region)
                    attributes["color"] = color_info["dominant_color"]
                    attributes["is_multicolored"] = color_info["is_multicolored"]
            except Exception:
                attributes["color"] = "unknown"

        det["attributes"] = attributes
        enhanced_detections.append(det)

    # Add spatial relationships if needed (expensive)
    if needs_spatial and len(enhanced_detections) > 1:
        enhanced_detections = add_spatial_relationships(enhanced_detections)

    return enhanced_detections


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
            distance = ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5
            relations.append("near" if distance < 100 else "far")

            # Check containment if bboxes available
            if "bbox" in det1 and "bbox" in det2:
                x1_1, y1_1, x2_1, y2_1 = det1["bbox"]
                x1_2, y1_2, x2_2, y2_2 = det2["bbox"]

                if x1_1 > x1_2 and y1_1 > y1_2 and x2_1 < x2_2 and y2_1 < y2_2:
                    relations.append("inside")
                elif x1_2 > x1_1 and y1_2 > y1_1 and x2_2 < x2_1 and y2_2 < y2_1:
                    relations.append("contains")

            det1["relationships"].append({
                "object": det2["label"],
                "object_id": det2.get("object_id", f"obj_{j}"),
                "relations": relations
            })

    return detections


def compress_detections_for_output(detections: List[Dict[str, Any]], is_video: bool = False) -> List[Dict[str, Any]]:
    """
    Compress detections for GPT consumption - remove verbose fields.

    Args:
        detections: List of detection dictionaries
        is_video: Whether this is for video processing

    Returns:
        Compressed detection list with essential fields only
    """
    compressed = []

    for det in detections:
        # Keep only essentials for GPT
        obj = {
            "id": det.get("object_id", f"obj_{len(compressed)}"),
            "type": det["label"],
            "confidence": round(det["confidence"], 2)
        }

        # Add attributes only if they exist
        attributes = det.get("attributes", {})
        if "size" in attributes:
            obj["size"] = attributes["size"]
        if "position" in attributes:
            obj["position"] = attributes["position"]
        if "color" in attributes and attributes["color"] != "unknown":
            obj["color"] = attributes["color"]

        # Add track_id if available (for videos)
        if "track_id" in det:
            obj["track_id"] = det["track_id"]

        # Add relationships for images (simplified)
        if not is_video and "relationships" in det and det["relationships"]:
            key_rels = []
            for rel in det["relationships"][:2]:  # Max 2 relationships to avoid verbosity
                if rel.get("relations"):
                    key_rels.append({
                        "to": rel["object"],
                        "relation": rel["relations"][0]  # Primary relation only
                    })
            if key_rels:
                obj["key_relationships"] = key_rels

        compressed.append(obj)

    return compressed


def update_object_tracker(
        tracker: Dict[str, Dict],
        detections: List[Dict[str, Any]],
        frame_idx: int,
        fps: float
):
    """
    Update object tracker for video processing - no per-frame storage.

    Args:
        tracker: Object tracker dictionary
        detections: Current frame detections
        frame_idx: Current frame index
        fps: Video FPS
    """
    current_time = frame_idx / fps

    for det in detections:
        label = det["label"]
        track_id = det.get("track_id", f"untracked_{len(tracker)}")
        obj_key = f"{label}_{track_id}"

        if obj_key not in tracker:
            tracker[obj_key] = {
                "type": label,
                "first_seen": current_time,
                "last_seen": current_time,
                "appearances": 1,
                "total_confidence": det["confidence"],
                "positions": [det.get("center", (0, 0))]  # Keep only few positions
            }
        else:
            # Update existing object
            tracker[obj_key]["last_seen"] = current_time
            tracker[obj_key]["appearances"] += 1
            tracker[obj_key]["total_confidence"] += det["confidence"]

            # Keep only last 3 positions to analyze movement
            if len(tracker[obj_key]["positions"]) < 3:
                tracker[obj_key]["positions"].append(det.get("center", (0, 0)))
            else:
                # Replace oldest position
                tracker[obj_key]["positions"] = tracker[obj_key]["positions"][1:] + [det.get("center", (0, 0))]


def update_time_window(
        windows: Dict[int, Dict],
        detections: List[Dict[str, Any]],
        window_idx: int
):
    """
    Aggregate detections into time windows for video analysis.

    Args:
        windows: Time windows dictionary
        detections: Current detections
        window_idx: Current time window index
    """
    if window_idx not in windows:
        windows[window_idx] = {"counts": {}, "total": 0}

    for det in detections:
        label = det["label"]
        windows[window_idx]["counts"][label] = windows[window_idx]["counts"].get(label, 0) + 1
        windows[window_idx]["total"] += 1


def analyze_movement_patterns(object_tracker: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Analyze movement patterns from object tracker.

    Args:
        object_tracker: Dictionary of tracked objects

    Returns:
        Movement analysis dictionary
    """
    patterns = {
        "stationary_objects": 0,
        "moving_objects": 0,
        "directional_movement": {}
    }

    for obj_key, obj_data in object_tracker.items():
        positions = obj_data["positions"]
        obj_type = obj_data["type"]

        if len(positions) < 2:
            patterns["stationary_objects"] += 1
            continue

        # Calculate movement distance
        first_pos = positions[0]
        last_pos = positions[-1]
        distance = ((last_pos[0] - first_pos[0])**2 + (last_pos[1] - first_pos[1])**2)**0.5

        if distance < 50:  # Threshold for stationary
            patterns["stationary_objects"] += 1
        else:
            patterns["moving_objects"] += 1

            # Determine primary direction
            dx = last_pos[0] - first_pos[0]
            dy = last_pos[1] - first_pos[1]

            if abs(dx) > abs(dy):
                direction = "right" if dx > 0 else "left"
            else:
                direction = "down" if dy > 0 else "up"

            # Track directional movement by object type
            if obj_type not in patterns["directional_movement"]:
                patterns["directional_movement"][obj_type] = {}
            patterns["directional_movement"][obj_type][direction] = \
                patterns["directional_movement"][obj_type].get(direction, 0) + 1

    return patterns


def create_temporal_analysis(time_windows: Dict[int, Dict], fps: float) -> Dict[str, Any]:
    """
    Create temporal analysis from time windows.

    Args:
        time_windows: Dictionary of time windows
        fps: Video FPS

    Returns:
        Temporal analysis dictionary
    """
    if not time_windows:
        return {}

    # Convert window indices to timestamps
    timeline_data = []
    for window_idx, window_data in sorted(time_windows.items()):
        timestamp = window_idx * 2  # 2-second windows
        timeline_data.append({
            "time": timestamp,
            "total_objects": window_data["total"],
            "object_types": window_data["counts"]
        })

    # Find peak activity
    if timeline_data:
        peak_window = max(timeline_data, key=lambda x: x["total_objects"])
        avg_objects = sum(item["total_objects"] for item in timeline_data) / len(timeline_data)
    else:
        peak_window = {"time": 0, "total_objects": 0}
        avg_objects = 0

    return {
        "peak_activity_time": peak_window["time"],
        "peak_activity_count": peak_window["total_objects"],
        "avg_objects_per_window": round(avg_objects, 1),
        "activity_timeline": timeline_data[:5]  # Limit to first 5 windows for GPT
    }


def identify_object_clusters(detections: List[Dict[str, Any]], distance_threshold: int = 150) -> List[List[int]]:
    """
    Identify clusters of objects in an image.

    Args:
        detections: List of detections with center coordinates
        distance_threshold: Maximum distance between objects in a cluster

    Returns:
        List of clusters, each containing detection indices
    """
    if len(detections) < 2:
        return []

    clusters = []
    used_objects = set()

    for i, det1 in enumerate(detections):
        if i in used_objects or "center" not in det1:
            continue

        cluster = [i]
        center1 = det1["center"]

        for j, det2 in enumerate(detections):
            if j <= i or j in used_objects or "center" not in det2:
                continue

            center2 = det2["center"]
            distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5

            if distance < distance_threshold:
                cluster.append(j)
                used_objects.add(j)

        if len(cluster) > 1:
            clusters.append(cluster)
            for obj_idx in cluster:
                used_objects.add(obj_idx)

    return clusters


