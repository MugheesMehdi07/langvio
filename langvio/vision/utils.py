"""
Enhanced utilities for vision processing
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict, deque


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
            "label": det["label"],
            "confidence": round(det["confidence"], 2),
            "bbox": det["bbox"]  # Keep bounding box for visualization
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


"""
Enhanced temporal and spatial analysis utilities for video processing
"""


class TemporalObjectTracker:
    """Tracks objects across video frames for temporal relationship analysis."""

    def __init__(self, max_history: int = 30):
        self.max_history = max_history
        self.object_histories = defaultdict(lambda: {
            "positions": deque(maxlen=max_history),
            "timestamps": deque(maxlen=max_history),
            "attributes": deque(maxlen=max_history),
            "first_seen": None,
            "last_seen": None,
            "total_appearances": 0
        })

    def update_frame(self, frame_idx: int, detections: List[Dict[str, Any]], fps: float):
        """Update tracking with new frame detections."""
        timestamp = frame_idx / fps

        for det in detections:
            track_id = det.get("track_id", f"untracked_{det.get('object_id', 'unknown')}")
            obj_key = f"{det['label']}_{track_id}"

            history = self.object_histories[obj_key]

            # Update position history
            center = det.get("center", (0, 0))
            history["positions"].append(center)
            history["timestamps"].append(timestamp)

            # Update attributes (store latest)
            history["attributes"].append(det.get("attributes", {}))

            # Update tracking metadata
            if history["first_seen"] is None:
                history["first_seen"] = timestamp
            history["last_seen"] = timestamp
            history["total_appearances"] += 1

    def get_movement_patterns(self) -> Dict[str, Any]:
        """Analyze movement patterns across all tracked objects."""
        patterns = {
            "stationary_objects": [],
            "moving_objects": [],
            "fast_moving_objects": [],
            "directional_movements": defaultdict(list),
            "interaction_events": []
        }

        for obj_key, history in self.object_histories.items():
            if len(history["positions"]) < 3:
                patterns["stationary_objects"].append(obj_key)
                continue

            # Calculate movement metrics
            positions = list(history["positions"])
            movement_distance = self._calculate_total_movement(positions)
            avg_speed = self._calculate_average_speed(positions, list(history["timestamps"]))
            primary_direction = self._get_primary_direction(positions)

            # Categorize object movement
            if movement_distance < 50:  # Threshold for stationary
                patterns["stationary_objects"].append(obj_key)
            elif avg_speed > 100:  # Threshold for fast movement
                patterns["fast_moving_objects"].append({
                    "object": obj_key,
                    "avg_speed": avg_speed,
                    "direction": primary_direction
                })
            else:
                patterns["moving_objects"].append({
                    "object": obj_key,
                    "avg_speed": avg_speed,
                    "direction": primary_direction
                })

            # Track directional movements
            if primary_direction:
                patterns["directional_movements"][primary_direction].append(obj_key)

        return patterns

    def get_temporal_relationships(self) -> List[Dict[str, Any]]:
        """Identify temporal relationships between objects."""
        relationships = []

        obj_keys = list(self.object_histories.keys())
        for i, obj1_key in enumerate(obj_keys):
            for obj2_key in obj_keys[i + 1:]:
                obj1_hist = self.object_histories[obj1_key]
                obj2_hist = self.object_histories[obj2_key]

                # Check for temporal overlap
                overlap = self._calculate_temporal_overlap(obj1_hist, obj2_hist)
                if overlap > 0.5:  # Significant overlap
                    relationships.append({
                        "object1": obj1_key.split('_')[0],  # Get object type
                        "object2": obj2_key.split('_')[0],
                        "relationship": "co_occurring",
                        "overlap_ratio": overlap,
                        "duration": min(obj1_hist["last_seen"] - obj1_hist["first_seen"],
                                        obj2_hist["last_seen"] - obj2_hist["first_seen"])
                    })

        return relationships

    def _calculate_total_movement(self, positions: List[Tuple]) -> float:
        """Calculate total movement distance."""
        if len(positions) < 2:
            return 0

        total_distance = 0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i - 1][0]
            dy = positions[i][1] - positions[i - 1][1]
            total_distance += (dx ** 2 + dy ** 2) ** 0.5

        return total_distance

    def _calculate_average_speed(self, positions: List[Tuple], timestamps: List[float]) -> float:
        """Calculate average speed in pixels per second."""
        if len(positions) < 2 or len(timestamps) < 2:
            return 0

        total_distance = self._calculate_total_movement(positions)
        total_time = timestamps[-1] - timestamps[0]

        return total_distance / total_time if total_time > 0 else 0

    def _get_primary_direction(self, positions: List[Tuple]) -> Optional[str]:
        """Get primary movement direction."""
        if len(positions) < 2:
            return None

        start_pos = positions[0]
        end_pos = positions[-1]

        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]

        if abs(dx) < 10 and abs(dy) < 10:  # Minimal movement
            return "stationary"

        if abs(dx) > abs(dy):
            return "right" if dx > 0 else "left"
        else:
            return "down" if dy > 0 else "up"

    def _calculate_temporal_overlap(self, hist1: Dict, hist2: Dict) -> float:
        """Calculate temporal overlap ratio between two objects."""
        if not (hist1["first_seen"] and hist1["last_seen"] and
                hist2["first_seen"] and hist2["last_seen"]):
            return 0

        # Calculate overlap period
        overlap_start = max(hist1["first_seen"], hist2["first_seen"])
        overlap_end = min(hist1["last_seen"], hist2["last_seen"])

        if overlap_start >= overlap_end:
            return 0

        overlap_duration = overlap_end - overlap_start
        total_duration = max(hist1["last_seen"], hist2["last_seen"]) - min(hist1["first_seen"], hist2["first_seen"])

        return overlap_duration / total_duration if total_duration > 0 else 0


class SpatialRelationshipAnalyzer:
    """Analyzes spatial relationships between objects in video frames."""

    def __init__(self):
        self.relationship_history = defaultdict(list)
        self.spatial_patterns = defaultdict(int)

    def update_relationships(self, detections: List[Dict[str, Any]]):
        """Update spatial relationships for current frame detections."""
        if len(detections) < 2:
            return

        frame_relationships = []

        for i, det1 in enumerate(detections):
            for j, det2 in enumerate(detections[i + 1:], i + 1):
                relationship = self._analyze_object_pair(det1, det2)
                if relationship:
                    frame_relationships.append(relationship)

                    # Track patterns
                    pattern_key = f"{det1['label']}-{relationship['relation']}-{det2['label']}"
                    self.spatial_patterns[pattern_key] += 1

        # Store relationships with timestamp
        if frame_relationships:
            self.relationship_history[len(self.relationship_history)] = frame_relationships

    def get_common_spatial_patterns(self, min_occurrences: int = 3) -> Dict[str, int]:
        """Get spatial patterns that occur frequently."""
        return {pattern: count for pattern, count in self.spatial_patterns.items()
                if count >= min_occurrences}

    def get_relationship_summary(self) -> Dict[str, Any]:
        """Get summary of spatial relationships throughout video."""
        if not self.relationship_history:
            return {}

        # Aggregate relationships across all frames
        relation_counts = defaultdict(int)
        object_pair_counts = defaultdict(int)

        for frame_rels in self.relationship_history.values():
            for rel in frame_rels:
                relation_counts[rel['relation']] += 1
                pair_key = f"{rel['object1']}-{rel['object2']}"
                object_pair_counts[pair_key] += 1

        return {
            "most_common_relations": dict(sorted(relation_counts.items(),
                                                 key=lambda x: x[1], reverse=True)[:5]),
            "frequent_object_pairs": dict(sorted(object_pair_counts.items(),
                                                 key=lambda x: x[1], reverse=True)[:5]),
            "spatial_patterns": self.get_common_spatial_patterns(),
            "total_relationship_events": sum(relation_counts.values())
        }

    def _analyze_object_pair(self, det1: Dict[str, Any], det2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze spatial relationship between two objects."""
        if not (det1.get("center") and det2.get("center")):
            return None

        center1 = det1["center"]
        center2 = det2["center"]

        # Calculate relative positions
        dx = center2[0] - center1[0]
        dy = center2[1] - center1[1]
        distance = (dx ** 2 + dy ** 2) ** 0.5

        # Determine primary relationship
        if distance < 100:  # Close proximity
            relation = "near"
        elif abs(dx) > abs(dy):
            relation = "right_of" if dx > 0 else "left_of"
        else:
            relation = "below" if dy > 0 else "above"

        return {
            "object1": det1["label"],
            "object2": det2["label"],
            "relation": relation,
            "distance": distance,
            "confidence": min(det1.get("confidence", 0.5), det2.get("confidence", 0.5))
        }


# Enhanced utility functions for frame processing
def add_tracking_info(detections: List[Dict[str, Any]], frame_idx: int) -> List[Dict[str, Any]]:
    """Add tracking information to detections."""
    for i, det in enumerate(detections):
        if "track_id" not in det:
            det["track_id"] = f"track_{frame_idx}_{i}"
        if "object_id" not in det:
            det["object_id"] = f"obj_{frame_idx}_{i}"
    return detections


def add_color_attributes(detections: List[Dict[str, Any]], frame: np.ndarray, needs_color: bool) -> List[
    Dict[str, Any]]:
    """Add color attributes to detections (optimized for video)."""
    if not needs_color or frame is None:
        return detections

    from langvio.vision.color_detection import ColorDetector

    for det in detections:
        if "bbox" not in det:
            continue

        try:
            x1, y1, x2, y2 = map(int, det["bbox"])
            if x1 >= x2 or y1 >= y2:
                continue

            # Extract object region
            obj_region = frame[y1:y2, x1:x2]
            if obj_region.size > 0:
                # Get dominant color only (faster than full profile)
                dominant_color = ColorDetector.detect_color(obj_region, return_all=False)

                if "attributes" not in det:
                    det["attributes"] = {}
                det["attributes"]["color"] = dominant_color
        except Exception:
            continue  # Skip on error

    return detections


def add_size_and_position_attributes(detections: List[Dict[str, Any]], width: int, height: int) -> List[Dict[str, Any]]:
    """Add size and position attributes (fast computation)."""
    image_area = width * height

    for det in detections:
        if "bbox" not in det:
            continue

        x1, y1, x2, y2 = det["bbox"]

        # Calculate center and size
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        det["center"] = (center_x, center_y)

        # Size attribute
        area = (x2 - x1) * (y2 - y1)
        relative_size = area / image_area

        if "attributes" not in det:
            det["attributes"] = {}

        det["attributes"]["size"] = "small" if relative_size < 0.05 else "medium" if relative_size < 0.25 else "large"

        # Position attribute
        rx, ry = center_x / width, center_y / height
        pos_v = "top" if ry < 0.33 else "middle" if ry < 0.66 else "bottom"
        pos_h = "left" if rx < 0.33 else "center" if rx < 0.66 else "right"
        det["attributes"]["position"] = f"{pos_v}-{pos_h}"

    return detections