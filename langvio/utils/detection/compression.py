"""
Detection result compression utilities for LLM consumption
"""

from typing import Any, Dict, List


def compress_detections_for_output(
    detections: List[Dict[str, Any]], is_video: bool = False
) -> List[Dict[str, Any]]:
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
            "bbox": det["bbox"],  # Keep bounding box for visualization
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
            for rel in det["relationships"][:2]:  # Max 2 relations to avoid verbosity
                if rel.get("relations"):
                    key_rels.append(
                        {
                            "to": rel["object"],
                            "relation": rel["relations"][0],  # Primary relation only
                        }
                    )
            if key_rels:
                obj["key_relationships"] = key_rels

        compressed.append(obj)

    return compressed


def identify_object_clusters(
    detections: List[Dict[str, Any]], distance_threshold: int = 150
) -> List[List[int]]:
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
            distance = (
                (center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2
            ) ** 0.5

            if distance < distance_threshold:
                cluster.append(j)
                used_objects.add(j)

        if len(cluster) > 1:
            clusters.append(cluster)
            for obj_idx in cluster:
                used_objects.add(obj_idx)

    return clusters
