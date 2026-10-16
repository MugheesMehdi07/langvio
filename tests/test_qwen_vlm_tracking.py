"""Tests for Qwen-VL video tracking: flow propagation, repetition filter, vocabulary."""

from unittest.mock import patch

import cv2
import numpy as np
import pytest
from PIL import Image

from langvio.vision.qwen_vlm.detector import QwenVisionProcessor
from langvio.vision.qwen_vlm.propagation import propagate_tracks
from langvio.vision.qwen_vlm.tracking import SparseTracker, iou

W, H, SIDE, SPEED = 320, 240, 40, 4  # object moves 4 px/frame


def _true_box(frame: int):
    x = 20 + SPEED * frame
    return [x, 100, x + SIDE, 100 + SIDE]


@pytest.fixture
def moving_square_video(tmp_path):
    """A textured square sliding right over a textured static background."""
    rng = np.random.default_rng(0)
    background = cv2.GaussianBlur(
        rng.integers(0, 255, (H, W, 3), dtype=np.uint8), (5, 5), 0
    )
    patch_ = rng.integers(0, 255, (SIDE, SIDE, 3), dtype=np.uint8)
    path = str(tmp_path / "square.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter.fourcc(*"MJPG"), 10, (W, H))
    for f in range(60):
        frame = background.copy()
        x1, y1, x2, y2 = _true_box(f)
        frame[y1:y2, x1:x2] = patch_
        writer.write(frame)
    writer.release()
    return path


def _det(box):
    return {"label": "box", "confidence": 0.9, "bbox": list(box)}


def test_flow_links_fast_mover_and_follows_it(moving_square_video):
    # 80 px between keyframes for a 40 px object: beyond the tracker's
    # distance gate unless the flow prediction is used.
    keyframes = {f: [_det(_true_box(f))] for f in (0, 20, 40)}

    plain = SparseTracker()
    for f in sorted(keyframes):
        plain.update([_det(_true_box(f))], f)
    assert len(plain.tracks) == 3

    tracker = SparseTracker()
    tracked, render = propagate_tracks(moving_square_video, keyframes, tracker, 60, 20)

    assert len(tracker.tracks) == 1
    assert set(tracked) == {"0", "20", "40"}
    for f in (5, 10, 15, 30, 45):
        (det,) = render[str(f)]
        assert iou(det["bbox"], _true_box(f)) > 0.7, f


def test_single_interior_sighting_is_not_rendered(moving_square_video):
    # step=15: frame 20 is neither within the first nor the last step.
    keyframes = {0: [], 20: [_det(_true_box(20))], 40: []}
    _, render = propagate_tracks(
        moving_square_video, keyframes, SparseTracker(), 60, 15
    )
    assert all(not dets for dets in render.values())


def test_tracker_prefers_given_prediction():
    tracker = SparseTracker()
    tracker.update([_det([0, 0, 10, 10])], 0)
    far = [200, 0, 210, 10]
    tracker.update([_det(far)], 10, predicted={1: far})
    assert len(tracker.tracks) == 1


def _boxes(*items):
    return [(list(b), label) for b, label in items]


def test_repetition_run_and_band_strays_are_dropped():
    row = [([240 + 107 * i, 601, 347 + 107 * i, 693], "chair") for i in range(10)]
    real = [([0, 684, 129, 870], "chair"), ([1273, 343, 1545, 920], "person")]
    stray = [([500, 605, 560, 690], "object")]  # same band, inside the run
    kept = QwenVisionProcessor._drop_repetition_runs(
        _boxes(*real, *row, *stray), 1920, 1080, 5
    )
    assert kept == _boxes(*real)


def test_short_runs_and_disabled_filter_keep_boxes():
    row = _boxes(*[([100 * i, 0, 100 * i + 100, 50], "car") for i in range(4)])
    assert QwenVisionProcessor._drop_repetition_runs(row, 1920, 1080, 5) == row
    long_row = _boxes(*[([100 * i, 0, 100 * i + 100, 50], "car") for i in range(8)])
    assert QwenVisionProcessor._drop_repetition_runs(long_row, 1920, 1080, 0) == (
        long_row
    )


def test_column_run_is_dropped():
    col = [([920, 286 + 41 * i, 960, 327 + 41 * i], "water cooler") for i in range(5)]
    assert QwenVisionProcessor._drop_repetition_runs(_boxes(*col), 1920, 1080, 5) == []


def test_vocabulary_filters_generic_and_scene_parts():
    proc = QwenVisionProcessor("qwen_vlm")
    reply = '```json\n["Person", "chair", "wall", "object", "person", "desk"]\n```'
    with patch.object(proc, "_generate", return_value=(reply, (1000, 1000))):
        vocab = proc._discover_vocabulary([Image.new("RGB", (64, 64))])
    assert vocab == ["person", "chair", "desk"]


def test_vocabulary_failure_returns_empty():
    proc = QwenVisionProcessor("qwen_vlm")
    with patch.object(proc, "_generate", side_effect=RuntimeError("rate limited")):
        assert proc._discover_vocabulary([Image.new("RGB", (64, 64))]) == []


def test_label_hints_do_not_restrict_grounding():
    # Hints come from a few sampled frames; a person or bicycle missing from
    # them must still be looked for on every keyframe.
    proc = QwenVisionProcessor("qwen_vlm")
    prompt = proc._build_grounding_prompt({"label_hints": ["car", "manhole"]})
    assert "Locate every distinct object" in prompt
    assert "car, manhole" in prompt
    named = proc._build_grounding_prompt({"target_objects": ["dog"]})
    assert "following objects in the image: dog." in named


def test_scene_parts_dropped_unless_requested():
    proc = QwenVisionProcessor("qwen_vlm")
    reply = (
        '[{"bbox_2d": [10, 10, 200, 200], "label": "person"},'
        ' {"bbox_2d": [300, 10, 600, 400], "label": "window"}]'
    )
    image = Image.new("RGB", (1000, 1000))
    with patch.object(proc, "_generate", return_value=(reply, (1000, 1000))):
        open_ended = proc._ground(image, {"target_objects": []})
        asked = proc._ground(image, {"target_objects": ["window"]})
    assert [d["label"] for d in open_ended] == ["person"]
    assert "window" in [d["label"] for d in asked]
