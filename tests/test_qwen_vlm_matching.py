"""Tests for descriptive queries ("the person wearing glasses") with Qwen-VL."""

from unittest.mock import patch

from langvio.utils.llm_utils import format_video_summary
from langvio.vision.qwen_vlm import matching
from langvio.vision.qwen_vlm.detector import QwenVisionProcessor
from langvio.vision.qwen_vlm.summary import _track_motion
from langvio.vision.qwen_vlm.tracking import SparseTracker


def test_description_only_when_it_narrows_the_targets():
    td = matching.target_description
    assert td({"target_objects": ["person"], "target_description": ""}) is None
    assert td({"target_objects": ["person"], "target_description": "people"}) is None
    assert td({"target_objects": ["car"], "target_description": "all the cars"}) is None
    assert (
        td(
            {
                "target_objects": ["person"],
                "target_description": "person wearing glasses",
            }
        )
        == "person wearing glasses"
    )
    # Parses without the field still carry qualifiers as attribute filters.
    assert (
        td(
            {
                "target_objects": ["car"],
                "attributes": [{"attribute": "color", "value": "red"}],
            }
        )
        == "red car"
    )


def test_parse_selection():
    assert matching.parse_selection("[2, 3]", 4) == [2, 3]
    assert matching.parse_selection("```json\n[]\n```", 4) == []
    assert matching.parse_selection("[0, 2, 2, 9]", 4) == [2]
    assert matching.parse_selection("candidate two", 4) is None


def test_pick_views_prefers_large_boxes_far_apart():
    history = [
        (0, [0, 0, 100, 100], None),
        (10, [0, 0, 100, 95], None),
        (50, [0, 0, 90, 100], None),
        (60, [0, 0, 30, 30], None),  # too small to be a clear view
    ]
    assert [f for f, _ in matching.pick_views(history, 2)] == [0, 50]


def _tracked():
    tracker = SparseTracker()
    dets = [
        {"label": "person", "confidence": 0.9, "bbox": [0, 0, 50, 100]},
        {"label": "person", "confidence": 0.9, "bbox": [200, 0, 250, 100]},
    ]
    keyframes = {"0": tracker.update(dets, 0)}
    render = {"0": list(keyframes["0"]), "1": list(keyframes["0"])}
    return tracker, keyframes, render


def test_only_matching_tracks_are_kept():
    proc = QwenVisionProcessor("qwen_vlm")
    tracker, keyframes, render = _tracked()
    params = {"target_objects": ["person"], "target_description": "person in a hat"}
    with patch.object(proc, "_select_tracks", return_value={2}):
        info = proc._keep_matching_tracks("v.mp4", tracker, keyframes, render, params)
    assert info == {
        "query_match": {"description": "person in a hat", "candidates": 2, "matched": 1}
    }
    assert list(tracker.tracks) == [2]
    assert [d["track_id"] for d in keyframes["0"]] == [2]
    assert all([d["track_id"] for d in dets] == [2] for dets in render.values())


def test_failed_selection_keeps_everything():
    proc = QwenVisionProcessor("qwen_vlm")
    tracker, keyframes, render = _tracked()
    params = {"target_objects": ["person"], "target_description": "person in a hat"}
    with patch.object(proc, "_select_tracks", return_value=None):
        info = proc._keep_matching_tracks("v.mp4", tracker, keyframes, render, params)
    assert info["query_match"]["matched"] is None
    assert len(tracker.tracks) == 2 and len(render["1"]) == 2


def test_plain_query_skips_selection():
    proc = QwenVisionProcessor("qwen_vlm")
    tracker, keyframes, render = _tracked()
    with patch.object(proc, "_select_tracks") as select:
        info = proc._keep_matching_tracks(
            "v.mp4", tracker, keyframes, render, {"target_objects": ["person"]}
        )
    assert info == {} and not select.called and len(tracker.tracks) == 2


def test_image_grounding_asks_for_the_description():
    proc = QwenVisionProcessor("qwen_vlm")
    prompt = proc._build_grounding_prompt(
        {"target_objects": ["person"], "ground_description": "person wearing glasses"}
    )
    assert '"person wearing glasses"' in prompt
    assert "Leave out other person objects" in prompt


def test_no_match_is_told_to_the_explanation():
    summary = {
        "query_match": {
            "description": "person wearing glasses",
            "candidates": 4,
            "matched": 0,
        }
    }
    text = format_video_summary({"summary": summary}, {})
    assert 'No detected object matches "person wearing glasses"' in text


def test_jitter_near_the_midline_is_not_a_crossing():
    jitter = [
        (i, [940 + (i % 2) * 40, 0, 1000 + (i % 2) * 40, 100], None) for i in range(10)
    ]
    assert _track_motion({"history": jitter}, 1920, 30)["crossings"] == []
    walk = [(i, [200 * i, 0, 200 * i + 60, 100], None) for i in range(10)]
    assert _track_motion({"history": walk}, 1920, 30)["crossings"] == ["left_to_right"]
