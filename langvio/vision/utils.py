"""
Enhanced utilities for vision processing - imports from reorganized modules
"""

# Import core detection utilities
from langvio.utils.detection import (
    extract_detections,
    optimize_for_memory,
    add_unified_attributes,
    add_tracking_info,
    add_color_attributes,
    add_size_and_position_attributes,
    compress_detections_for_output,
    identify_object_clusters
)

# Import spatial utilities
from langvio.utils.spatial import (
    add_spatial_relationships,
    calculate_relative_positions,
    detect_spatial_relationships
)

# Keep the complex temporal analysis classes here as they're vision-specific
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict, deque


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
