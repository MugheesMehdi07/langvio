"""
Pick the tracked objects that fit a descriptive query.

"The person wearing glasses" is grounded as plain "person" so every person is
tracked reliably; a qualifier like glasses is not visible on every keyframe,
so asking for it per frame makes boxes flicker or jump between people. Instead
each track is shown to the VLM once - a few clear crops at full resolution -
and one call per batch of candidates decides which tracks match.
"""

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

_FILLER = {"a", "an", "the", "all", "every", "each", "any", "of", "one", "some"}
_CROP_PAD = 0.15
_CROP_SIDE = 448  # longest side of one view, in px
_GAP = 8  # px between views of the same candidate


def target_description(query_params: Dict[str, Any]) -> Optional[str]:
    """The phrase that singles out objects, or None when every object counts."""
    targets = [str(t).strip().lower() for t in query_params.get("target_objects", [])]
    description = str(query_params.get("target_description") or "").strip()
    if not description and len(targets) == 1:
        # Older parses carry qualifiers only as attribute filters.
        parts = [
            str(a.get("value", "")).strip()
            for a in query_params.get("attributes", [])
            if isinstance(a, dict) and a.get("value")
        ]
        if parts:
            description = f"{' '.join(parts)} {targets[0]}"
    plain = set(targets) | {f"{t}s" for t in targets} | {"people", "persons"}
    words = re.findall(r"[a-z0-9']+", description.lower())
    if not [w for w in words if w not in plain and w not in _FILLER]:
        return None
    return description


def pick_views(
    history: Sequence[Tuple[int, List[float], Any]], n_views: int = 2
) -> List[Tuple[int, List[float]]]:
    """Choose the clearest sightings of a track, spread out in time.

    The largest box is the closest, clearest look; a second view from a
    different moment helps when the object faced away in the first.
    """

    def area(b: List[float]) -> float:
        return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    if not history:
        return []
    ranked = sorted(history, key=lambda h: area(h[1]), reverse=True)
    best = ranked[0]
    views = [(best[0], list(best[1]))]
    big_enough = [h for h in ranked[1:] if area(h[1]) >= 0.6 * area(best[1])]
    while big_enough and len(views) < n_views:
        far = max(big_enough, key=lambda h: min(abs(h[0] - f) for f, _ in views))
        views.append((far[0], list(far[1])))
        big_enough.remove(far)
    return sorted(views)


def read_frames(video_path: str, frame_ids: Iterable[int]) -> Dict[int, np.ndarray]:
    """Full-resolution BGR frames (fine details like glasses need them)."""
    frames: Dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(video_path)
    try:
        for frame_id in sorted(set(frame_ids)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = cap.read()
            if ok:
                frames[frame_id] = frame
    finally:
        cap.release()
    return frames


def crop(frame: np.ndarray, box: Sequence[float]) -> Optional[Image.Image]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * _CROP_PAD, (y2 - y1) * _CROP_PAD
    ix1, iy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
    ix2, iy2 = min(w, int(x2 + px)), min(h, int(y2 + py))
    if ix2 - ix1 < 4 or iy2 - iy1 < 4:
        return None
    image = Image.fromarray(cv2.cvtColor(frame[iy1:iy2, ix1:ix2], cv2.COLOR_BGR2RGB))
    scale = _CROP_SIDE / max(image.size)
    if scale < 1:
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.Resampling.BICUBIC,
        )
    return image


def candidate_image(views: List[Image.Image], number: int) -> Image.Image:
    """Views side by side, with the candidate number drawn in the corner."""
    height = max(v.height for v in views)
    width = sum(v.width for v in views) + _GAP * (len(views) - 1)
    sheet = Image.new("RGB", (width, height), (0, 0, 0))
    x = 0
    for view in views:
        sheet.paste(view, (x, 0))
        x += view.width + _GAP
    canvas = np.array(sheet)
    label = str(number)
    scale = max(0.8, height / 300)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.rectangle(canvas, (0, 0), (tw + 12, th + 14), (255, 255, 0), -1)
    cv2.putText(
        canvas, label, (6, th + 7), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2
    )
    return Image.fromarray(canvas)


def selection_prompt(description: str, n: int, multi_view: bool) -> str:
    views = (
        " Panels side by side in one image are the same candidate at different "
        "moments."
        if multi_view
        else ""
    )
    return (
        f"Each of the {n} images shows one candidate, numbered 1 to {n} in the "
        f"order given (the number is also drawn in the corner).{views} Which "
        f'candidates match this description: "{description}"? Look closely at '
        "each one. Answer only with a JSON list of the matching numbers, e.g. "
        "[2] or [1, 3], and [] if none match."
    )


def parse_selection(text: str, n: int) -> Optional[List[int]]:
    """1-based candidate numbers from the reply; None if it can't be read."""
    match = re.search(r"\[([^\[\]]*)\]", text)
    if not match:
        return None
    picked: List[int] = []
    for token in re.findall(r"-?\d+", match.group(1)):
        k = int(token)
        if 1 <= k <= n and k not in picked:
            picked.append(k)
    return picked
