"""
Summary builders for Qwen VLM detections.

Produces the summary schema consumed by ``langvio.utils.llm_utils`` (explanation
prompt), the video overlay, and the web UI (``video_info``,
``counting_analysis``, ``temporal_relationships``, ``object_analysis``,
``primary_insights``, ``image_info``, ``object_distribution``...).
"""

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

# Speeds are measured in frame-widths per second.
_STATIONARY_SPEED = 0.015
_FAST_SPEED = 0.15


def _plural(label: str, n: int) -> str:
    if n == 1:
        return label
    if label == "person":
        return "people"
    return label if label.endswith("s") else f"{label}s"


def build_image_summary(
    detections: List[Dict[str, Any]],
    width: int,
    height: int,
    query_params: Dict[str, Any],
    scene_description: Optional[str] = None,
) -> Dict[str, Any]:
    by_type: Counter = Counter()
    by_position: Counter = Counter()
    by_size: Counter = Counter()
    by_color: Counter = Counter()
    for det in detections:
        by_type[det["label"]] += 1
        attrs = det.get("attributes", {})
        if attrs.get("position"):
            by_position[attrs["position"]] += 1
        if attrs.get("size"):
            by_size[attrs["size"]] += 1
        if attrs.get("color") and attrs["color"] != "unknown":
            by_color[attrs["color"]] += 1

    patterns = [
        f"{n} {_plural(label, n)} detected" for label, n in by_type.most_common()
    ]
    if by_position:
        pos, n = by_position.most_common(1)[0]
        if n > 1:
            patterns.append(f"Objects are concentrated in the {pos} region ({n})")

    return {
        "image_info": {
            "resolution": f"{width}x{height}",
            "total_objects": len(detections),
            "unique_types": len(by_type),
        },
        "object_distribution": {
            "by_type": dict(by_type.most_common()),
            "by_position": dict(by_position),
            "by_size": dict(by_size),
            "by_color": dict(by_color),
        },
        "notable_patterns": patterns,
        "scene_description": scene_description or "",
        # Kept for backwards compatibility with the YOLO-World summary shape.
        "total_objects": len(detections),
        "object_counts": dict(by_type),
        "image_dimensions": {"width": width, "height": height},
        "query_type": query_params.get("task_type", "identification"),
    }


def _track_motion(track: Dict[str, Any], width: int, fps: float) -> Dict[str, Any]:
    history = track["history"]
    (f0, b0, _), (f1, b1, _) = history[0], history[-1]
    c0 = np.array([(b0[0] + b0[2]) / 2, (b0[1] + b0[3]) / 2])
    c1 = np.array([(b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2])
    duration = max((f1 - f0) / fps, 1e-6)
    dx, dy = c1 - c0
    speed = float(np.hypot(dx, dy) / width / duration) if len(history) > 1 else 0.0

    if len(history) < 2:
        behavior = "brief"
    elif speed < _STATIONARY_SPEED:
        behavior = "stationary"
    elif speed < _FAST_SPEED:
        behavior = "moving"
    else:
        behavior = "fast_moving"

    direction = None
    if behavior in ("moving", "fast_moving"):
        if abs(dx) >= abs(dy):
            direction = "left_to_right" if dx > 0 else "right_to_left"
        else:
            direction = "toward_camera" if dy > 0 else "away_from_camera"

    midline = width / 2
    crossings = []
    for (fa, ba, _), (fb, bb, _) in zip(history, history[1:]):
        xa, xb = (ba[0] + ba[2]) / 2, (bb[0] + bb[2]) / 2
        if (xa - midline) * (xb - midline) < 0:
            crossings.append("left_to_right" if xb > xa else "right_to_left")

    return {
        "behavior": behavior,
        "direction": direction,
        "speed": speed,
        "crossings": crossings,
        "observations": len(history),
    }


def build_video_summary(
    tracks: List[Dict[str, Any]],
    keyframes: Dict[str, List[Dict[str, Any]]],
    width: int,
    height: int,
    fps: float,
    total_frames: int,
    query_params: Dict[str, Any],
    model_name: str,
    scene_description: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Args:
        tracks: Internal ``SparseTracker.tracks`` values (with history).
        keyframes: frame_id -> detections for analysed keyframes only.
    """
    duration = total_frames / fps if fps else 0.0
    keyframe_ids = sorted(int(k) for k in keyframes)
    first_kf = keyframe_ids[0] if keyframe_ids else 0
    last_kf = keyframe_ids[-1] if keyframe_ids else 0

    # Peak simultaneous count per label is the most reliable count signal.
    peak: Dict[str, int] = defaultdict(int)
    peak_frame: Dict[str, int] = {}
    per_frame_totals = []
    for fid in keyframe_ids:
        counts = Counter(d["label"] for d in keyframes[str(fid)])
        per_frame_totals.append(sum(counts.values()))
        for label, n in counts.items():
            if n > peak[label]:
                peak[label], peak_frame[label] = n, fid

    motions = {t["track_id"]: _track_motion(t, width, fps) for t in tracks}
    unique: Counter = Counter(t["label"] for t in tracks)
    confirmed: Counter = Counter(
        t["label"] for t in tracks if motions[t["track_id"]]["observations"] > 1
    )

    movement = Counter(m["behavior"] for m in motions.values())
    directions: Counter = Counter(
        m["direction"] for m in motions.values() if m["direction"]
    )

    by_type: Dict[str, Dict[str, Any]] = {}
    total_crossings = 0
    crossing_dirs: Counter = Counter()
    for t in tracks:
        label, m = t["label"], motions[t["track_id"]]
        entry = by_type.setdefault(
            label, {"entered": 0, "exited": 0, "net_flow": 0, "crossings": 0}
        )
        if t["first_frame"] > first_kf:
            entry["entered"] += 1
        if t["last_frame"] < last_kf:
            entry["exited"] += 1
        entry["crossings"] += len(m["crossings"])
        total_crossings += len(m["crossings"])
        crossing_dirs.update(m["crossings"])
    for entry in by_type.values():
        entry["net_flow"] = entry["entered"] - entry["exited"]
        entry["dominance"] = (
            "entering"
            if entry["net_flow"] > 0
            else "exiting" if entry["net_flow"] < 0 else "balanced"
        )

    entered = sum(e["entered"] for e in by_type.values())
    exited = sum(e["exited"] for e in by_type.values())
    if crossing_dirs:
        top_dir, top_n = crossing_dirs.most_common(1)[0]
        flow_direction = top_dir if top_n > sum(crossing_dirs.values()) / 2 else "mixed"
    else:
        flow_direction = "none"

    characteristics: Dict[str, Dict[str, Any]] = {}
    for label in unique:
        label_tracks = [t for t in tracks if t["label"] == label]
        behaviors = Counter(motions[t["track_id"]]["behavior"] for t in label_tracks)
        attrs: Counter = Counter()
        for t in label_tracks:
            for _, _, det in t["history"]:
                for key in ("size", "position"):
                    value = det.get("attributes", {}).get(key)
                    if value:
                        attrs[f"{key}:{value}"] += 1
        characteristics[label] = {
            "total_instances": max(confirmed[label], peak.get(label, 0)),
            "confirmed_instances": confirmed[label],
            "max_simultaneous": peak.get(label, 0),
            "movement_behavior": behaviors.most_common(1)[0][0],
            "common_attributes": dict(attrs.most_common(5)),
        }

    avg_objects = float(np.mean(per_frame_totals)) if per_frame_totals else 0.0
    moving_share = (movement["moving"] + movement["fast_moving"]) / max(
        1, sum(movement.values())
    )
    if avg_objects >= 6 or (avg_objects >= 3 and moving_share > 0.5):
        activity = "high"
    elif avg_objects >= 2 or moving_share > 0.3:
        activity = "moderate"
    elif avg_objects > 0:
        activity = "low"
    else:
        activity = "none"

    primary_objects = [label for label, _ in unique.most_common()]

    # Tracks seen in a single keyframe are often re-detections of someone already
    # tracked, so they are excluded; the peak frame is a hard lower bound.
    estimated = {
        label: max(confirmed[label], peak.get(label, 0)) for label in unique
    }

    insights: List[str] = []
    for label in primary_objects:
        t_sec = peak_frame.get(label, 0) / fps if fps else 0
        insights.append(
            f"About {estimated[label]} distinct {_plural(label, estimated[label])} "
            f"in the video"
        )
        insights.append(
            f"Up to {peak[label]} {_plural(label, peak[label])} visible at once "
            f"(t={t_sec:.1f}s)"
        )
    if directions:
        d, n = directions.most_common(1)[0]
        insights.append(f"Dominant movement: {d.replace('_', ' ')} ({n} tracks)")

    return {
        "video_info": {
            "duration_seconds": round(duration, 1),
            "resolution": f"{width}x{height}",
            "fps": round(fps, 2),
            "activity_level": activity,
            "primary_objects": primary_objects,
        },
        "counting_analysis": {
            "total_crossings": total_crossings,
            "objects_entered": entered,
            "objects_exited": exited,
            "net_flow": entered - exited,
            "flow_direction": flow_direction,
            "by_object_type": by_type,
            "most_active_type": primary_objects[0] if primary_objects else None,
            "estimated_objects": estimated,
            "max_simultaneous": dict(peak),
            "unique_objects": dict(unique),
            "confirmed_objects": dict(confirmed),
        },
        "temporal_relationships": {
            "movement_patterns": {
                "stationary_count": movement["stationary"],
                "moving_count": movement["moving"],
                "fast_moving_count": movement["fast_moving"],
                "brief_count": movement["brief"],
                "primary_directions": dict(directions),
            },
            "co_occurrence_events": 0,
        },
        "object_analysis": {
            "object_characteristics": characteristics,
            "most_common_types": primary_objects,
            "total_unique_objects": sum(estimated.values()),
        },
        "primary_insights": insights,
        "scene_description": scene_description or "",
        "total_frames": len(keyframe_ids),
        "tracking_enabled": True,
        "query_type": query_params.get("task_type", "identification"),
        "model_name": model_name,
    }
