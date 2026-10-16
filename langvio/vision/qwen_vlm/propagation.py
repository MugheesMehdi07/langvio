"""
Optical-flow box propagation between VLM keyframes.

Keyframes are 1-2+ s apart, so sliding boxes linearly between them drifts off
anything that turns or changes speed, and a walking person moves further than
the tracker's distance gate between samples. This pass reads the video once,
moves every live box with sparse Lucas-Kanade flow (plain OpenCV on a
downscaled grey frame - no extra model calls), and hands the flow-predicted
positions to the tracker at each keyframe so fast movers stay one track.
"""

import copy
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from langvio.vision.qwen_vlm.tracking import SparseTracker

_LK_WIN = (15, 15)
_LK_LEVELS = 3
_LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
_MAX_POINTS = 40
_MIN_POINTS = 4
_RESEED_EVERY = 10  # frames
_FB_ERROR = 1.0  # max forward-backward flow error (flow-resolution px)
_MAX_SCALE_STEP = 0.05  # max box size change per frame
_MIN_VISIBLE = 0.3  # stop following a box once less of it is inside the frame

Box = np.ndarray  # float [x1, y1, x2, y2] in original video pixels


class _Flow:
    """Sparse LK flow on downscaled grey frames; boxes stay in video pixels."""

    def __init__(self, width: int, height: int, flow_size: int):
        self.width, self.height = width, height
        self.scale = min(1.0, flow_size / float(max(width, height)))

    def gray(self, frame: np.ndarray) -> np.ndarray:
        if self.scale < 1.0:
            frame = cv2.resize(
                frame,
                None,
                fx=self.scale,
                fy=self.scale,
                interpolation=cv2.INTER_LINEAR,
            )
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def seed(self, gray: np.ndarray, box: Box) -> Optional[np.ndarray]:
        """Pick corner points inside the box (its inner part, to avoid background)."""
        x1, y1, x2, y2 = box * self.scale
        mx, my = (x2 - x1) * 0.15, (y2 - y1) * 0.15
        h, w = gray.shape
        ix1, iy1 = max(0, int(x1 + mx)), max(0, int(y1 + my))
        ix2, iy2 = min(w, int(np.ceil(x2 - mx))), min(h, int(np.ceil(y2 - my)))
        if ix2 - ix1 < 3 or iy2 - iy1 < 3:
            return None
        # On the crop: a mask would still scan the whole frame.
        pts = cv2.goodFeaturesToTrack(gray[iy1:iy2, ix1:ix2], _MAX_POINTS, 0.01, 3)
        if pts is None or len(pts) < _MIN_POINTS:
            return None
        return (pts + np.array([ix1, iy1], dtype=np.float32)).astype(np.float32)

    @staticmethod
    def track(
        prev: np.ndarray, cur: np.ndarray, groups: List[np.ndarray]
    ) -> List[Optional[Tuple[np.ndarray, np.ndarray]]]:
        """Flow several point groups in one LK call.

        Returns, per group, the (old, new) points that survived a
        forward-backward check, or None when too few did.
        """
        p0 = np.concatenate(groups).astype(np.float32)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            prev,
            cur,
            p0,
            np.empty_like(p0),
            winSize=_LK_WIN,
            maxLevel=_LK_LEVELS,
            criteria=_LK_CRITERIA,
        )
        back, st2, _ = cv2.calcOpticalFlowPyrLK(
            cur,
            prev,
            p1,
            np.empty_like(p1),
            winSize=_LK_WIN,
            maxLevel=_LK_LEVELS,
            criteria=_LK_CRITERIA,
        )
        fb = np.linalg.norm((p0 - back).reshape(-1, 2), axis=1)
        ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < _FB_ERROR)

        out: List[Optional[Tuple[np.ndarray, np.ndarray]]] = []
        start = 0
        for group in groups:
            sl = slice(start, start + len(group))
            start += len(group)
            good = ok[sl]
            if good.sum() < _MIN_POINTS:
                out.append(None)
            else:
                out.append((p0[sl][good].reshape(-1, 2), p1[sl][good].reshape(-1, 2)))
        return out

    def shift(self, box: Box, old: np.ndarray, new: np.ndarray) -> Box:
        """Move/rescale a box by the median motion of its points."""
        d = np.median(new - old, axis=0) / self.scale
        r0 = np.linalg.norm(old - old.mean(axis=0), axis=1)
        r1 = np.linalg.norm(new - new.mean(axis=0), axis=1)
        spread = r0 > 2
        s = (
            float(np.median(r1[spread] / r0[spread]))
            if spread.sum() >= _MIN_POINTS
            else 1.0
        )
        s = float(np.clip(s, 1 - _MAX_SCALE_STEP, 1 + _MAX_SCALE_STEP))
        cx, cy = (box[0] + box[2]) / 2 + d[0], (box[1] + box[3]) / 2 + d[1]
        hw, hh = (box[2] - box[0]) * s / 2, (box[3] - box[1]) * s / 2
        return np.array([cx - hw, cy - hh, cx + hw, cy + hh])

    def visible(self, box: Box) -> bool:
        area = (box[2] - box[0]) * (box[3] - box[1])
        ix = max(0.0, min(box[2], self.width) - max(box[0], 0.0))
        iy = max(0.0, min(box[3], self.height) - max(box[1], 0.0))
        return area > 0 and ix * iy / area >= _MIN_VISIBLE


def propagate_tracks(  # noqa: C901
    video_path: str,
    keyframes: Dict[int, List[Dict[str, Any]]],
    tracker: SparseTracker,
    total_frames: int,
    step: int,
    flow_size: int = 640,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    """
    Track keyframe detections through the video with optical flow.

    Args:
        keyframes: frame_id -> detections (original video pixels), in any order.
        tracker: Fresh tracker; it is updated at every keyframe in frame order.

    Returns:
        (tracked keyframe detections keyed by str(frame_id),
         per-frame render detections for every frame of the video)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    flow = _Flow(width, height, flow_size)
    hold = max(1, step // 2)

    live: Dict[int, Dict[str, Any]] = {}  # track_id -> box, pts, age, det
    trail: Dict[int, Dict[int, Tuple[Box, Dict[str, Any]]]] = defaultdict(dict)
    recent: Deque[np.ndarray] = deque(maxlen=hold)  # grey frames before current
    tracked: Dict[str, List[Dict[str, Any]]] = {}
    pending = sorted(keyframes)

    def backfill(gray: np.ndarray, frame_id: int, new: List[Dict[str, Any]]):
        # A new track's object was usually already visible before the keyframe
        # that found it: follow it backwards for up to half a keyframe interval.
        heads: List[Tuple[Box, np.ndarray, Dict[str, Any]]] = []
        for det in new:
            box = np.array(det["bbox"], dtype=float)
            pts = flow.seed(gray, box)
            if pts is not None:
                heads.append((box, pts, det))
        cur = gray
        for k, earlier in enumerate(reversed(recent), 1):
            if not heads:
                return
            results = flow.track(cur, earlier, [h[1] for h in heads])
            still: List[Tuple[Box, np.ndarray, Dict[str, Any]]] = []
            for head, res in zip(heads, results):
                if res is None:
                    continue
                box = flow.shift(head[0], *res)
                if not flow.visible(box):
                    continue
                trail[head[2]["track_id"]][frame_id - k] = (box.copy(), head[2])
                still.append((box, res[1].reshape(-1, 1, 2), head[2]))
            heads, cur = still, earlier

    try:
        prev: Optional[np.ndarray] = None
        frame_id = 0
        while frame_id < total_frames:
            ok, frame = cap.read()
            if not ok:
                break
            gray = flow.gray(frame)

            if prev is not None and live:
                ids = [t for t, s in live.items() if s["pts"] is not None]
                results = (
                    flow.track(prev, gray, [live[t]["pts"] for t in ids]) if ids else []
                )
                moved = dict(zip(ids, results))
                for tid in list(live):
                    state = live[tid]
                    res = moved.get(tid)
                    if res is not None:
                        state["box"] = flow.shift(state["box"], *res)
                        state["pts"] = res[1].reshape(-1, 1, 2)
                    else:
                        # Textureless or lost: coast on the keyframe velocity.
                        vx, vy = tracker.tracks[tid]["velocity"]
                        state["box"] = state["box"] + np.array([vx, vy, vx, vy])
                        state["pts"] = None
                    if not flow.visible(state["box"]):
                        del live[tid]
                        continue
                    state["age"] += 1
                    if state["pts"] is None or state["age"] >= _RESEED_EVERY:
                        state["pts"] = flow.seed(gray, state["box"])
                        state["age"] = 0

            if pending and pending[0] == frame_id:
                pending.pop(0)
                predicted = {t: s["box"].tolist() for t, s in live.items()}
                dets = tracker.update(keyframes[frame_id], frame_id, predicted)
                tracked[str(frame_id)] = dets
                backfill(
                    gray, frame_id, [d for d in dets if d["track_id"] not in trail]
                )
                for det in dets:
                    box = np.array(det["bbox"], dtype=float)
                    live[det["track_id"]] = {
                        "box": box,
                        "pts": flow.seed(gray, box),
                        "age": 0,
                        "det": det,
                    }
                for tid in list(live):
                    if tid not in tracker.active:
                        del live[tid]

            for tid, state in live.items():
                trail[tid][frame_id] = (state["box"].copy(), state["det"])
            recent.append(gray)
            prev = gray
            frame_id += 1
    finally:
        cap.release()

    # Keyframes past the last decodable frame still count for tracking.
    for frame_id in pending:
        tracked[str(frame_id)] = tracker.update(keyframes[frame_id], frame_id)

    return tracked, _render(
        tracker, trail, keyframes, total_frames, step, width, height
    )


def _render(
    tracker: SparseTracker,
    trail: Dict[int, Dict[int, Tuple[Box, Dict[str, Any]]]],
    keyframes: Dict[int, List[Dict[str, Any]]],
    total_frames: int,
    step: int,
    width: int,
    height: int,
) -> Dict[str, List[Dict[str, Any]]]:
    per_frame: Dict[int, List[Dict[str, Any]]] = {f: [] for f in range(total_frames)}
    last_kf = max(keyframes) if keyframes else 0
    hold = max(1, step // 2)

    for tid, track in tracker.tracks.items():
        seen = [f for f, _, _ in track["history"]]
        # A track seen in a single interior keyframe is overwhelmingly a VLM
        # hallucination; ones at the video boundary may be entering/leaving.
        if len(seen) == 1 and step < seen[0] < total_frames - step:
            continue
        # After the last sighting, keep following only briefly unless the
        # object was still there at the final keyframe.
        end = total_frames - 1 if seen[-1] >= last_kf else seen[-1] + hold
        for f, (box, det) in trail.get(tid, {}).items():
            if f > end or not 0 <= f < total_frames:
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            d = copy.copy(det)
            d["bbox"] = [x1, y1, x2, y2]
            per_frame[f].append(d)

    return {str(f): dets for f, dets in per_frame.items()}
