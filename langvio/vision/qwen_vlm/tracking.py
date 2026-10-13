"""
Tracking and box interpolation for sparsely sampled VLM keyframes.

VLM grounding costs seconds per frame, so videos are analysed at roughly one
keyframe per second. ByteTrack-style IoU matching breaks down at that spacing
(people move more than their own width between samples), so this tracker
matches on a constant-velocity prediction with a size-normalised distance gate,
then linearly interpolates boxes between keyframes for smooth rendering.
"""

import copy
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

_INVALID = 1e6


def _center(b: List[float]) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])


def _diag(b: List[float]) -> float:
    return float(np.hypot(b[2] - b[0], b[3] - b[1])) or 1.0


def iou(a: List[float], b: List[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class SparseTracker:
    """Label-aware tracker tuned for keyframes that are ~1s apart."""

    def __init__(self, max_missed: int = 2, distance_gate: float = 1.2):
        """
        Args:
            max_missed: Keyframes a track may go unmatched before it is closed.
            distance_gate: Max centre distance (in units of box diagonal) between
                a track's predicted position and a detection for them to match.
        """
        self.max_missed = max_missed
        self.distance_gate = distance_gate
        self._next_id = 1
        self.active: Dict[int, Dict[str, Any]] = {}
        self.tracks: Dict[int, Dict[str, Any]] = {}

    def _predict(self, track: Dict[str, Any], frame_id: int) -> List[float]:
        dt = frame_id - track["last_frame"]
        dx, dy = track["velocity"] * dt
        b = track["bbox"]
        return [b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy]

    def update(
        self, detections: List[Dict[str, Any]], frame_id: int
    ) -> List[Dict[str, Any]]:
        track_ids = list(self.active.keys())
        predictions = [self._predict(self.active[t], frame_id) for t in track_ids]

        matches: Dict[int, int] = {}
        if track_ids and detections:
            cost = np.full((len(track_ids), len(detections)), _INVALID)
            for i, tid in enumerate(track_ids):
                track = self.active[tid]
                for j, det in enumerate(detections):
                    if det["label"] != track["label"]:
                        continue
                    pred = predictions[i]
                    dist = np.linalg.norm(_center(pred) - _center(det["bbox"]))
                    norm = dist / max(_diag(pred), _diag(det["bbox"]))
                    if norm > self.distance_gate:
                        continue
                    cost[i, j] = norm + (1.0 - iou(pred, det["bbox"]))
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] < _INVALID:
                    matches[c] = track_ids[r]

        matched_tracks = set()
        for j, det in enumerate(detections):
            tid = matches.get(j)
            if tid is None:
                tid = self._next_id
                self._next_id += 1
                track = {
                    "track_id": tid,
                    "label": det["label"],
                    "class_name": det["label"],
                    "bbox": det["bbox"],
                    "velocity": np.zeros(2),
                    "first_frame": frame_id,
                    "last_frame": frame_id,
                    "missed": 0,
                    "history": [],
                }
                self.active[tid] = track
                self.tracks[tid] = track
            else:
                track = self.active[tid]
                dt = max(1, frame_id - track["last_frame"])
                new_v = (_center(det["bbox"]) - _center(track["bbox"])) / dt
                # Smooth velocity so one noisy box doesn't throw off prediction.
                track["velocity"] = (
                    new_v
                    if len(track["history"]) < 2
                    else 0.6 * new_v + 0.4 * track["velocity"]
                )
                track["bbox"] = det["bbox"]
                track["last_frame"] = frame_id
                track["missed"] = 0

            det["track_id"] = tid
            det["object_id"] = f"{det['label']}_{tid}"
            track["history"].append((frame_id, list(det["bbox"]), det))
            matched_tracks.add(tid)

        for tid in list(self.active.keys()):
            if tid in matched_tracks:
                continue
            self.active[tid]["missed"] += 1
            if self.active[tid]["missed"] > self.max_missed:
                del self.active[tid]

        return detections

    def get_all_tracks(self) -> List[Dict[str, Any]]:
        """JSON-friendly track records (history reduced to frame/bbox pairs)."""
        out = []
        for t in self.tracks.values():
            out.append(
                {
                    "track_id": t["track_id"],
                    "class_name": t["label"],
                    "first_seen": t["first_frame"],
                    "last_seen": t["last_frame"],
                    "total_frames": len(t["history"]),
                    "bbox": t["bbox"],
                    "history": [(f, b) for f, b, _ in t["history"]],
                }
            )
        return out


def interpolate_tracks(
    tracker: SparseTracker,
    total_frames: int,
    step: int,
    max_gap_steps: Optional[int] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build per-frame detections for every frame of the video.

    Boxes are linearly interpolated between consecutive observations of the
    same track and held for half a keyframe interval at each track's ends, so
    the rendered video moves smoothly instead of jumping once per keyframe.
    """
    if max_gap_steps is None:
        # Only interpolate between consecutive keyframes. Allowing larger
        # gaps "ghosts" a box across frames where the object was unmatched —
        # i.e. actually offscreen — which is the opposite of what we want.
        max_gap_steps = 1
    max_gap = max_gap_steps * step
    hold = max(1, step // 2)
    live_track_ids = set(tracker.active.keys())

    per_frame: Dict[int, List[Dict[str, Any]]] = {f: [] for f in range(total_frames)}

    def emit(frame: int, det: Dict[str, Any], bbox: List[float]) -> None:
        if 0 <= frame < total_frames:
            d = copy.copy(det)
            d["bbox"] = [int(round(v)) for v in bbox]
            per_frame[frame].append(d)

    # Drop one-off tracks that appeared in a single interior keyframe and were
    # never seen again — these are overwhelmingly VLM hallucinations (ghost
    # boxes on empty road, shadows, lens flares). Keep single-observation
    # tracks that touch the first or last keyframe, since those can be
    # legitimate objects entering/leaving at the video boundary.
    first_keyframe_window = step
    last_keyframe_window = total_frames - step

    for track in tracker.tracks.values():
        history = track["history"]
        if len(history) == 1:
            only_frame = history[0][0]
            at_boundary = (
                only_frame <= first_keyframe_window
                or only_frame >= last_keyframe_window
            )
            if not at_boundary:
                continue
        first_frame = history[0][0]
        for f in range(first_frame - hold, first_frame):
            emit(f, history[0][2], history[0][1])

        for (f0, b0, d0), (f1, b1, _) in zip(history, history[1:]):
            if f1 - f0 > max_gap:
                # Gap larger than a single keyframe: object was unmatched in
                # between. Briefly hold the previous box and the next box at
                # their respective ends, but do NOT interpolate across — the
                # object wasn't actually there.
                for f in range(f0, min(f0 + hold, f1)):
                    emit(f, d0, b0)
                for f in range(max(f0 + 1, f1 - hold), f1):
                    emit(f, d0, b1)
                continue
            span = f1 - f0
            a0, a1 = np.array(b0, dtype=float), np.array(b1, dtype=float)
            for f in range(f0, f1):
                t = (f - f0) / span
                emit(f, d0, list((1 - t) * a0 + t * a1))

        last_frame, last_box, last_det = history[-1]
        # Only trail a box past the final observation when the track was
        # still alive at the end of the video. A track that was closed
        # mid-video means the object left the scene — dropping its box
        # right away matches reality instead of freezing it on screen.
        reached_video_end = (
            track["track_id"] in live_track_ids
            or last_frame >= total_frames - step
        )
        if reached_video_end:
            for f in range(last_frame, last_frame + hold + 1):
                emit(f, last_det, last_box)

    return {str(f): dets for f, dets in per_frame.items()}
