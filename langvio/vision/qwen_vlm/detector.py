"""
Qwen-VL vision processor - open-vocabulary detection via VLM grounding.

Replaces the YOLO-World detector. Qwen-VL is prompted to ground the query's
target objects and returns ``[{"bbox_2d": [x1, y1, x2, y2], "label": ...}]``,
which is mapped back to the original image. The same model also writes a short
scene description from the image / video keyframes so the explanation step has
visual context beyond boxes.

By default the prompts go to a hosted Qwen-VL model over an API key
(``langvio.llm.qwen_api``); Qwen3-VL answers on a 0-1000 grid, while Qwen2/2.5
answers in the pixel space of the (resized) model input. Setting
``LANGVIO_BACKEND=local`` runs the weights in-process instead.
"""

import json
import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PIL import Image

from langvio.llm.qwen_api import QwenAPIClient, bbox_space, resolve_backend
from langvio.llm.qwen_loader import DEFAULT_VLM_MODEL, GENERATION_LOCK, load_qwen
from langvio.utils.detection import (
    add_unified_attributes,
    compress_detections_for_output,
)
from langvio.vision.base import BaseVisionProcessor
from langvio.vision.qwen_vlm import matching
from langvio.vision.qwen_vlm.propagation import propagate_tracks
from langvio.vision.qwen_vlm.summary import build_image_summary, build_video_summary
from langvio.vision.qwen_vlm.tracking import SparseTracker, interpolate_tracks, iou

# Qwen2.5-VL patches are 14px and merged 2x2, so the model input is a multiple of 28.
_PATCH_SIZE = 14

_BBOX_PATTERN = re.compile(
    r'"bbox_2d"\s*:\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,'
    r'\s*([-\d.]+)\s*\]\s*,?\s*(?:"label"\s*:\s*"([^"]*)")?'
)

_GENERIC_TARGETS = {"object", "objects", "thing", "things", "item", "items"}

# Scene parts the VLM tends to "detect" when asked for every object.
_NOT_OBJECTS = {
    "wall",
    "floor",
    "ceiling",
    "window",
    "road",
    "ground",
    "sky",
    "grass",
    "room",
    "building",
    "hair",
    "face",
    "head",
    "hand",
    "shirt",
    "clothing",
    "shadow",
    "light",
    "background",
}


class QwenVisionProcessor(BaseVisionProcessor):
    """Vision processor that uses Qwen2.5-VL grounding instead of YOLO."""

    def __init__(
        self,
        name: str,
        model_name: str = DEFAULT_VLM_MODEL,
        confidence: float = 0.9,
        max_image_size: int = 1024,
        max_new_tokens: int = 1024,
        max_video_frames: int = 24,
        keyframes_per_second: float = 1.0,
        scene_description: bool = True,
        scene_frames: int = 4,
        nms_iou: float = 0.6,
        backend: str = "auto",
        bbox_format: str = "auto",
        api_concurrency: int = 2,
        **kwargs,
    ):
        """
        Initialize the Qwen vision processor.

        Args:
            name: Processor name.
            model_name: HuggingFace repo id of the Qwen(2/2.5)-VL model.
            confidence: Confidence assigned to every grounded box (the VLM does
                not produce per-box scores).
            max_image_size: Longest side (px) images are downscaled to before
                grounding. 1024 is the sweet spot for the 3B model; larger inputs
                are slower and its grounding output becomes unstable.
            max_new_tokens: Generation budget for the grounding JSON.
            max_video_frames: Upper bound on keyframes analysed per video.
            keyframes_per_second: Target keyframe density for videos.
            scene_description: Whether to ask the VLM for a scene description.
            scene_frames: Keyframes shown to the VLM for the video description.
            nms_iou: Same-label boxes overlapping more than this are merged.
            backend: "auto" (default: hosted Qwen-VL if an API key is
                configured, else local weights), "api" or "local" (in-process
                weights, needs ``langvio[vlm]``). ``LANGVIO_BACKEND`` overrides it.
            bbox_format: Grounding coordinate space: "auto", "pixel" or
                "norm1000" (0-1000 grid, Qwen3 and later).
            api_concurrency: Parallel keyframe requests with the API backend.
            **kwargs: Extra config (tracker settings, api_model, api_base_url,
                api_key_env, api_extra_body, api_timeout, optical_flow,
                flow_size, auto_vocabulary, repetition_min_run, match_views,
                match_batch_size).
        """
        config = {
            "model_name": model_name,
            "confidence": confidence,
            "max_image_size": max_image_size,
            "max_new_tokens": max_new_tokens,
            "max_video_frames": max_video_frames,
            "keyframes_per_second": keyframes_per_second,
            "scene_description": scene_description,
            "scene_frames": scene_frames,
            "nms_iou": nms_iou,
            "backend": backend,
            "bbox_format": bbox_format,
            "api_concurrency": api_concurrency,
            **kwargs,
        }
        super().__init__(name, config)
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.hf_model: Any = None
        self.hf_processor: Any = None
        self.api_client: Optional[QwenAPIClient] = None
        self._label_ids: Dict[str, int] = {}

    def initialize(self) -> bool:
        try:
            if resolve_backend(self.config) == "api":
                self.api_client = QwenAPIClient.from_config(self.config)
                self.model = self.api_client
                self.logger.info(
                    f"Qwen vision via API: {self.api_client.model} "
                    f"({self.api_client.base_url})"
                )
            else:
                self.hf_model, self.hf_processor = load_qwen(self.model_name)
                self.model = self.hf_model
            return True
        except Exception as e:
            self.logger.error(f"Error loading Qwen vision model: {e}")
            return False

    def _ensure_model(self) -> None:
        if self.model is None and not self.initialize():
            raise RuntimeError(f"Could not load Qwen vision model '{self.model_name}'")

    @property
    def active_model_name(self) -> str:
        return self.api_client.model if self.api_client else self.model_name

    # ------------------------------------------------------------------ #
    # Model calls
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resize(image: Image.Image, max_side: int) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        scale = max_side / max(width, height)
        if scale < 1:
            image = image.resize(
                (max(1, int(width * scale)), max(1, int(height * scale))),
                Image.BICUBIC,
            )
        return image

    def _generate(
        self, images: List[Image.Image], text: str, max_new_tokens: int
    ) -> Tuple[str, Tuple[int, int]]:
        """Run one image(s)+text turn.

        Returns the decoded text and the (width, height) of the coordinate
        space grounding boxes in the answer refer to.
        """
        self._ensure_model()
        if self.api_client is not None:
            output = self.api_client.chat(
                [
                    {
                        "role": "user",
                        "content": self.api_client.user_content(images, text),
                    }
                ],
                max_tokens=max_new_tokens,
                temperature=0.0,
            )
            space = bbox_space(
                self.api_client.model, str(self.config.get("bbox_format", "auto"))
            )
            if space == "norm1000":
                return output, (1000, 1000)
            return output, images[0].size if images else (1, 1)

        import torch

        content: List[Dict[str, Any]] = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": text})
        prompt = self.hf_processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.hf_processor(
            text=[prompt], images=images, return_tensors="pt"
        ).to(self.hf_model.device)

        with GENERATION_LOCK, torch.inference_mode():
            generated = self.hf_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
            )
        trimmed = generated[:, inputs["input_ids"].shape[1] :]
        output = self.hf_processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Qwen2.5-VL answers in pixels of its resized input (patch grid * 14).
        grid_thw = inputs.get("image_grid_thw")
        if grid_thw is not None:
            size = (
                int(grid_thw[0][2]) * _PATCH_SIZE,
                int(grid_thw[0][1]) * _PATCH_SIZE,
            )
        else:
            size = images[0].size if images else (1, 1)
        if (
            bbox_space(self.model_name, str(self.config.get("bbox_format", "auto")))
            == "norm1000"
        ):
            size = (1000, 1000)
        return output, size

    @staticmethod
    def _targets(query_params: Dict[str, Any]) -> List[str]:
        return [
            t.strip()
            for t in query_params.get("target_objects", [])
            if t and t.strip().lower() not in _GENERIC_TARGETS
        ]

    def _build_grounding_prompt(self, query_params: Dict[str, Any]) -> str:
        # Wording was chosen empirically on the 3B model: an explicit schema plus
        # the "small/distant/occluded" hint finds the most instances, while an
        # "output [] if none" escape hatch makes it return empty lists far more
        # often. Re-benchmark before changing it.
        targets = self._targets(query_params)
        description = query_params.get("ground_description")
        if targets and description:
            # Images only: one call does the finding and the picking.
            task = (
                f"Locate only the objects matching this description: "
                f'"{description}". Leave out other {", ".join(targets)} '
                "objects that do not match it."
            )
            names = (
                "Label each box with the plain object name "
                f"({', '.join(targets)}), not the description."
            )
        elif targets:
            task = (
                "Locate every instance of the following objects in the image: "
                f"{', '.join(targets)}."
            )
            names = "Use the object names given above as labels."
        else:
            task = (
                "Locate every distinct object in the image (people, vehicles, "
                "animals and other salient objects)."
            )
            names = "Use short, common, singular nouns as labels."
            hints = query_params.get("label_hints") or []
            if hints:
                # Steer naming only: restricting to these would miss anything
                # absent from the frames the hints were drawn from.
                names = (
                    "Use short, common, singular nouns as labels, reusing these "
                    f"names where they fit: {', '.join(hints)}."
                )
        return (
            f"{task} Include small, distant and "
            "partially occluded ones, one box per individual. Output their bbox "
            'coordinates in JSON format as a list of {"bbox_2d": [x1, y1, x2, y2], '
            f'"label": "<object name>"}}. {names}'
        )

    def _ground(
        self, image: Image.Image, query_params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Run Qwen grounding on a PIL image; bboxes are in original image pixels."""
        orig_w, orig_h = image.size
        model_image = self._resize(image, int(self.config["max_image_size"]))

        text, (input_w, input_h) = self._generate(
            [model_image],
            self._build_grounding_prompt(query_params),
            int(self.config["max_new_tokens"]),
        )
        self.logger.debug(f"Qwen grounding output: {text}")

        detections = self._parse_boxes(
            text, orig_w / input_w, orig_h / input_h, orig_w, orig_h
        )
        # Scene parts (wall, hair, window...) only when the user asked for them.
        targets = {t.lower() for t in self._targets(query_params)}
        return [
            d
            for d in detections
            if d["label"] not in _NOT_OBJECTS or d["label"] in targets
        ]

    def _describe_scene(
        self, images: List[Image.Image], query_params: Dict[str, Any], is_video: bool
    ) -> str:
        if not self.config.get("scene_description", True) or not images:
            return ""
        query = query_params.get("query", "")
        focus = f' The user asked: "{query}".' if query else ""
        if is_video:
            intro = f"These {len(images)} frames are sampled in order from one video."
            ask = (
                "Describe the video as one continuous scene: the setting and what "
                "happens overall (who/what is present, actions, movement). Never "
                "mention frames, images or their order, and do not describe them "
                "one by one."
            )
        else:
            intro = "Look at this image."
            ask = (
                "Describe the setting and what is happening (who/what is present, "
                "actions, notable details)."
            )
        prompt = (
            f"{intro}{focus} {ask} Focus on details relevant to the question. "
            "Be factual, 3-4 sentences, and do not state exact counts."
        )
        side = 640 if is_video else 896
        try:
            text, _ = self._generate(
                [self._resize(img, side) for img in images], prompt, 220
            )
            return " ".join(text.split())
        except Exception as e:
            self.logger.warning(f"Scene description failed: {e}")
            return ""

    def _discover_vocabulary(self, images: List[Image.Image]) -> List[str]:
        """Ask once which object kinds a video shows.

        Without named targets the VLM picks different labels for the same
        objects on every keyframe, which fragments tracks. Passing one shared
        list as naming hints keeps labels consistent for the cost of a single
        extra call per video. The list comes from a few sampled frames, so it
        must never restrict what is grounded.
        """
        prompt = (
            f"These {len(images)} frames are sampled from one video. List the "
            "kinds of distinct, countable physical objects visible: people, "
            "vehicles, animals and other salient objects. Do not list surfaces, "
            "rooms or parts of objects (wall, floor, ceiling, window, road, "
            "hair, face, clothing). Answer only with a JSON list of at most "
            f"{int(self.config.get('vocabulary_size', 8))} short, common, "
            'singular nouns, most important first, e.g. ["person", "car"].'
        )
        try:
            text, _ = self._generate(images, prompt, 100)
        except Exception as e:
            self.logger.warning(f"Object vocabulary discovery failed: {e}")
            return []
        parsed = self._try_parse_json(text)
        if not isinstance(parsed, list):
            return []
        vocab: List[str] = []
        for item in parsed:
            name = str(item).strip().lower()
            if (
                name
                and name not in vocab
                and name not in _GENERIC_TARGETS
                and name not in _NOT_OBJECTS
            ):
                vocab.append(name)
        return vocab[: int(self.config.get("vocabulary_size", 8))]

    # ------------------------------------------------------------------ #
    # Box parsing
    # ------------------------------------------------------------------ #

    def _parse_boxes(
        self,
        text: str,
        scale_x: float,
        scale_y: float,
        width: int,
        height: int,
    ) -> List[Dict[str, Any]]:
        raw: List[Tuple[List[float], str]] = []

        parsed = self._try_parse_json(text)
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and len(item.get("bbox_2d", [])) == 4:
                    raw.append((item["bbox_2d"], str(item.get("label", "object"))))
        else:
            # Output may be truncated or slightly malformed; salvage what we can.
            for m in _BBOX_PATTERN.finditer(text):
                raw.append(([float(v) for v in m.groups()[:4]], m.group(5) or "object"))

        boxes: List[Tuple[List[int], str]] = []
        for bbox, label in raw:
            try:
                x1, y1, x2, y2 = (float(v) for v in bbox)
            except (TypeError, ValueError):
                continue
            boxes.append(
                (
                    [
                        int(max(0, min(width - 1, x1 * scale_x))),
                        int(max(0, min(height - 1, y1 * scale_y))),
                        int(max(0, min(width, x2 * scale_x))),
                        int(max(0, min(height, y2 * scale_y))),
                    ],
                    label.strip().lower() or "object",
                )
            )
        boxes = self._drop_repetition_runs(
            boxes, width, height, int(self.config.get("repetition_min_run", 5))
        )

        detections: List[Dict[str, Any]] = []
        confidence = float(self.config.get("confidence", 0.9))
        min_side = max(4, int(0.004 * max(width, height)))
        for (x1, y1, x2, y2), label in boxes:
            if x2 - x1 < min_side or y2 - y1 < min_side:
                continue
            detections.append(
                {
                    "label": label,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                    "class_id": self._label_ids.setdefault(label, len(self._label_ids)),
                }
            )
        detections = self._nms(detections, float(self.config.get("nms_iou", 0.6)))
        return self._drop_group_boxes(detections)

    @staticmethod
    def _drop_repetition_runs(  # noqa: C901
        boxes: List[Tuple[List[int], str]], width: int, height: int, min_run: int
    ) -> List[Tuple[List[int], str]]:
        """Drop VLM repetition loops.

        A degenerating model keeps emitting same-sized boxes that tile a row or
        column edge to edge (e.g. 14 "chair" boxes marching across a room, or a
        "water cooler" sliced into a stack), usually with a few stray boxes in
        the same band. A run of ``min_run``+ consecutive tiling boxes is
        removed along with any other box filling that band. 0 disables this.
        """
        if min_run <= 0 or len(boxes) < min_run:
            return boxes
        tol_x, tol_y = 0.02 * width, 0.02 * height

        def similar(a: float, b: float) -> bool:
            return a > 0 and b > 0 and 0.6 <= a / b <= 1.7

        def tiling(a: List[int], b: List[int]) -> Optional[str]:
            if (
                abs(a[1] - b[1]) <= tol_y
                and abs(a[3] - b[3]) <= tol_y
                and abs(b[0] - a[2]) <= tol_x
                and similar(b[2] - b[0], a[2] - a[0])
            ):
                return "row"
            if (
                abs(a[0] - b[0]) <= tol_x
                and abs(a[2] - b[2]) <= tol_x
                and abs(b[1] - a[3]) <= tol_y
                and similar(b[3] - b[1], a[3] - a[1])
            ):
                return "col"
            return None

        runs: List[Tuple[int, int, str]] = []
        start, axis = 0, None
        for i in range(1, len(boxes) + 1):
            link = None
            if i < len(boxes) and boxes[i][1] == boxes[i - 1][1]:
                link = tiling(boxes[i - 1][0], boxes[i][0])
            if link is not None and axis in (None, link):
                axis = link
                continue
            if axis is not None and i - start >= min_run:
                runs.append((start, i, axis))
            # The box that broke the run may start a new one with its predecessor.
            start, axis = (i - 1, link) if link is not None else (i, None)

        if not runs:
            return boxes
        drop: Set[int] = set()
        for s, e, run_axis in runs:
            drop.update(range(s, e))
            run = [boxes[k][0] for k in range(s, e)]
            lo = min(b[0] if run_axis == "row" else b[1] for b in run)
            hi = max(b[2] if run_axis == "row" else b[3] for b in run)
            band = np.mean(run, axis=0)
            for k, (b, _) in enumerate(boxes):
                if run_axis == "row":
                    in_band = (
                        abs(b[1] - band[1]) <= tol_y and abs(b[3] - band[3]) <= tol_y
                    )
                    center = (b[0] + b[2]) / 2
                else:
                    in_band = (
                        abs(b[0] - band[0]) <= tol_x and abs(b[2] - band[2]) <= tol_x
                    )
                    center = (b[1] + b[3]) / 2
                if in_band and lo <= center <= hi:
                    drop.add(k)
        return [box for k, box in enumerate(boxes) if k not in drop]

    @staticmethod
    def _nms(
        detections: List[Dict[str, Any]], threshold: float
    ) -> List[Dict[str, Any]]:
        """Suppress duplicate boxes.

        Same-label duplicates use the configured threshold; cross-label
        duplicates (e.g. the VLM emitting both "person" and "vehicle" on the
        same pixels) are suppressed at a stricter threshold so legitimately
        overlapping distinct objects — like a person in front of a car — are
        still kept.
        """
        cross_threshold = max(threshold, 0.75)
        kept: List[Dict[str, Any]] = []
        for det in detections:
            keep = True
            for k in kept:
                overlap = iou(k["bbox"], det["bbox"])
                same_label = k["label"] == det["label"]
                limit = threshold if same_label else cross_threshold
                if overlap >= limit:
                    keep = False
                    break
            if keep:
                kept.append(det)
        return kept

    @staticmethod
    def _drop_group_boxes(
        detections: List[Dict[str, Any]], containment: float = 0.9
    ) -> List[Dict[str, Any]]:
        """Drop boxes that enclose 2+ same-label boxes (VLM "group" boxes)."""

        def contains(outer: List[int], inner: List[int]) -> bool:
            ix1, iy1 = max(outer[0], inner[0]), max(outer[1], inner[1])
            ix2, iy2 = min(outer[2], inner[2]), min(outer[3], inner[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area = (inner[2] - inner[0]) * (inner[3] - inner[1])
            return area > 0 and inter / area >= containment

        kept = []
        for i, det in enumerate(detections):
            enclosed = sum(
                1
                for j, other in enumerate(detections)
                if i != j
                and other["label"] == det["label"]
                and contains(det["bbox"], other["bbox"])
            )
            if enclosed < 2:
                kept.append(det)
        return kept

    @staticmethod
    def _try_parse_json(text: str) -> Optional[Any]:
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        candidate = fenced.group(1) if fenced else text
        start = candidate.find("[")
        end = candidate.rfind("]")
        if start == -1 or end <= start:
            return None
        try:
            return json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            return None

    # ------------------------------------------------------------------ #
    # Image
    # ------------------------------------------------------------------ #

    def process_image(
        self, image_path: str, query_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        self.logger.info(f"Processing image with Qwen VLM: {image_path}")
        try:
            image = Image.open(image_path).convert("RGB")
            width, height = image.size

            description = matching.target_description(query_params)
            detections = self._ground(
                image, {**query_params, "ground_description": description}
            )
            for i, det in enumerate(detections):
                det["object_id"] = f"obj_{i}"

            needs_color = any(
                attr.get("attribute") == "color"
                for attr in query_params.get("attributes", [])
            )
            detections = add_unified_attributes(
                detections,
                width,
                height,
                image_path,
                needs_color,
                bool(query_params.get("spatial_relations", [])),
                True,
                is_video_frame=False,
            )
            scene = self._describe_scene([image], query_params, is_video=False)
            summary = build_image_summary(
                detections, width, height, query_params, scene
            )
            if description:
                summary["query_match"] = {
                    "description": description,
                    "matched": len(detections),
                }

            return {
                "objects": compress_detections_for_output(detections, is_video=False),
                "summary": summary,
            }
        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            return {"objects": [], "error": str(e)}

    # ------------------------------------------------------------------ #
    # Video
    # ------------------------------------------------------------------ #

    def process_video(
        self,
        video_path: str,
        query_params: Dict[str, Any],
        sample_rate: int = 3,
    ) -> Dict[str, Any]:
        self.logger.info(f"Processing video with Qwen VLM: {video_path}")
        start_time = time.time()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": f"Failed to open video: {video_path}"}

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # VLM inference is seconds per frame, so analyse ~1 keyframe/s (capped)
        # and interpolate boxes for the frames in between.
        max_frames = max(1, int(self.config.get("max_video_frames", 24)))
        per_second = float(self.config.get("keyframes_per_second", 1.0)) or 1.0
        step = max(
            1,
            sample_rate,
            int(round(fps / per_second)),
            math.ceil(total_frames / max_frames),
        )
        keyframe_ids = list(range(0, max(total_frames, 1), step))[:max_frames]
        if total_frames > 1 and keyframe_ids[-1] < total_frames - step // 2 - 1:
            keyframe_ids.append(total_frames - 1)

        n_scene = max(1, int(self.config.get("scene_frames", 4)))
        scene_ids = {
            keyframe_ids[round(i * (len(keyframe_ids) - 1) / max(1, n_scene - 1))]
            for i in range(min(n_scene, len(keyframe_ids)))
        }

        scene_images: List[Image.Image] = []
        try:
            self._ensure_model()
            frames: List[Tuple[int, Image.Image]] = []
            for frame_id in keyframe_ids:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ok, frame = cap.read()
                if not ok:
                    continue
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # Keep only the downscaled copy; full-res frames add up quickly.
                image = self._resize(image, int(self.config["max_image_size"]))
                if frame_id in scene_ids:
                    scene_images.append(self._resize(image, 640))
                frames.append((frame_id, image))
        finally:
            cap.release()

        grounding_params = query_params
        if (
            not self._targets(query_params)
            and self.config.get("auto_vocabulary", True)
            and scene_images
        ):
            vocab = self._discover_vocabulary(scene_images)
            if vocab:
                self.logger.info(f"Label hints for grounding: {', '.join(vocab)}")
                grounding_params = {**query_params, "label_hints": vocab}

        # Grounding works on the downscaled frames, so map boxes back up.
        def ground_frame(
            item: Tuple[int, Image.Image],
        ) -> Optional[List[Dict[str, Any]]]:
            frame_id, image = item
            try:
                detections = self._ground(image, grounding_params)
            except Exception as e:
                # One lost keyframe is survivable: tracks bridge the gap.
                self.logger.warning(f"Skipping keyframe {frame_id}: {e}")
                return None
            sx, sy = width / image.size[0], height / image.size[1]
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                det["bbox"] = [
                    int(x1 * sx),
                    int(y1 * sy),
                    min(width, int(x2 * sx)),
                    min(height, int(y2 * sy)),
                ]
            self.logger.info(
                f"Analysed keyframe {frame_id}/{total_frames} "
                f"({len(detections)} objects)"
            )
            return detections

        try:
            # Remote calls are I/O bound, so keyframes can be grounded in
            # parallel; tracking below still consumes them in frame order.
            workers = (
                max(1, int(self.config.get("api_concurrency", 2)))
                if self.api_client is not None
                else 1
            )
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(ground_frame, frames))
            failed = sum(1 for r in results if r is None)
            if frames and failed == len(frames):
                raise RuntimeError(
                    "every keyframe failed; see the warnings above (for the API "
                    "backend this is usually the free model's rate limit)"
                )
        except Exception as e:
            self.logger.error(f"Error processing video: {e}")
            return {"error": str(e)}

        detected = {
            frame_id: self._add_basic_attributes(detections, width, height)
            for (frame_id, _), detections in zip(frames, results)
            if detections is not None
        }
        tracker, keyframes, render = self._track(
            video_path, detected, total_frames, step
        )
        match_info = self._keep_matching_tracks(
            video_path, tracker, keyframes, render, query_params
        )

        scene = self._describe_scene(scene_images, query_params, is_video=True)
        internal_tracks = list(tracker.tracks.values())
        summary = build_video_summary(
            internal_tracks,
            keyframes,
            width,
            height,
            fps,
            total_frames,
            query_params,
            self.active_model_name,
            scene,
        )
        summary.update(match_info)

        # history holds live detection dicts; strip before returning/serialising.
        tracks_out = tracker.get_all_tracks()
        elapsed = time.time() - start_time
        self.logger.info(
            f"Video analysed: {len(keyframes)} keyframes, {len(internal_tracks)} "
            f"tracks in {elapsed:.1f}s"
        )

        return {
            "frame_detections": keyframes,
            "render_detections": render,
            "summary": summary,
            "tracks": tracks_out,
            "processing_info": {
                "frames_analyzed": len(keyframes),
                "frames_failed": failed,
                "total_frames": total_frames,
                "frame_step": step,
                "analysis_type": "vlm_keyframe_grounding",
                "detector": (
                    f"Qwen-VL API ({self.active_model_name})"
                    if self.api_client is not None
                    else "Qwen2.5-VL (local)"
                ),
                "model_name": self.active_model_name,
                "processing_seconds": round(elapsed, 1),
            },
            "metadata": {
                "width": width,
                "height": height,
                "fps": fps,
                "total_frames": total_frames,
            },
        }

    def _keep_matching_tracks(
        self,
        video_path: str,
        tracker: SparseTracker,
        keyframes: Dict[str, List[Dict[str, Any]]],
        render: Dict[str, List[Dict[str, Any]]],
        query_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Drop tracks that don't fit a descriptive query, in place.

        Returns ``{"query_match": ...}`` for the summary, or {} when the query
        names no description.
        """
        description = matching.target_description(query_params)
        if not description:
            return {}
        drawn = {d["track_id"] for dets in render.values() for d in dets}
        candidates = [t for tid, t in tracker.tracks.items() if tid in drawn]
        selected = self._select_tracks(video_path, candidates, description)
        info = {"description": description, "candidates": len(candidates)}
        if selected is None:
            self.logger.warning(
                f'Could not check tracks against "{description}"; keeping all'
            )
            return {"query_match": {**info, "matched": None}}

        self.logger.info(
            f'{len(selected)} of {len(candidates)} tracks match "{description}"'
        )
        for tid in list(tracker.tracks):
            if tid not in selected:
                del tracker.tracks[tid]
                tracker.active.pop(tid, None)
        for frames in (keyframes, render):
            for key, dets in frames.items():
                frames[key] = [d for d in dets if d.get("track_id") in selected]
        return {"query_match": {**info, "matched": len(selected)}}

    def _select_tracks(
        self,
        video_path: str,
        candidates: List[Dict[str, Any]],
        description: str,
    ) -> Optional[Set[int]]:
        """Ask the VLM which tracks fit the description (one call per batch).

        Returns the matching track ids, or None if the answer was unusable.
        """
        if not candidates:
            return set()
        n_views = max(1, int(self.config.get("match_views", 2)))
        views = {
            t["track_id"]: matching.pick_views(t["history"], n_views)
            for t in candidates
        }
        frames = matching.read_frames(
            video_path, (f for vs in views.values() for f, _ in vs)
        )
        crops: List[Tuple[int, List[Image.Image]]] = []
        for tid, track_views in views.items():
            images = [
                c
                for f, box in track_views
                if f in frames and (c := matching.crop(frames[f], box)) is not None
            ]
            if images:
                crops.append((tid, images))

        batch_size = max(1, int(self.config.get("match_batch_size", 12)))
        selected: Set[int] = set()
        for start in range(0, len(crops), batch_size):
            batch = crops[start : start + batch_size]
            images = [
                matching.candidate_image(imgs, k)
                for k, (_, imgs) in enumerate(batch, 1)
            ]
            prompt = matching.selection_prompt(
                description, len(batch), any(len(imgs) > 1 for _, imgs in batch)
            )
            try:
                text, _ = self._generate(images, prompt, 60)
            except Exception as e:
                self.logger.warning(f"Track selection call failed: {e}")
                return None
            picked = matching.parse_selection(text, len(batch))
            if picked is None:
                self.logger.warning(f"Unreadable track selection reply: {text!r}")
                return None
            selected.update(batch[k - 1][0] for k in picked)
        return selected

    def _new_tracker(self) -> SparseTracker:
        return SparseTracker(
            max_missed=int(self.config.get("track_max_missed", 2)),
            distance_gate=float(self.config.get("track_distance_gate", 1.2)),
        )

    def _track(
        self,
        video_path: str,
        detected: Dict[int, List[Dict[str, Any]]],
        total_frames: int,
        step: int,
    ) -> Tuple[
        SparseTracker,
        Dict[str, List[Dict[str, Any]]],
        Dict[str, List[Dict[str, Any]]],
    ]:
        """Link keyframe detections into tracks and build per-frame boxes.

        Returns (tracker, tracked keyframe detections, render detections).
        """
        if self.config.get("optical_flow", True):
            tracker = self._new_tracker()
            try:
                keyframes, render = propagate_tracks(
                    video_path,
                    detected,
                    tracker,
                    total_frames,
                    step,
                    int(self.config.get("flow_size", 640)),
                )
                return tracker, keyframes, render
            except Exception as e:
                self.logger.warning(
                    f"Optical-flow tracking failed ({e}); interpolating boxes"
                )
        tracker = self._new_tracker()
        keyframes = {
            str(frame_id): tracker.update(detected[frame_id], frame_id)
            for frame_id in sorted(detected)
        }
        return tracker, keyframes, interpolate_tracks(tracker, total_frames, step)

    @staticmethod
    def _add_basic_attributes(
        detections: List[Dict[str, Any]], width: int, height: int
    ) -> List[Dict[str, Any]]:
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            det["center"] = (center_x, center_y)

            relative_size = ((x2 - x1) * (y2 - y1)) / float(width * height)
            attributes = det.setdefault("attributes", {})
            attributes["size"] = (
                "small"
                if relative_size < 0.05
                else "medium" if relative_size < 0.25 else "large"
            )
            attributes["relative_size"] = round(relative_size, 4)

            rx, ry = center_x / width, center_y / height
            pos_v = "top" if ry < 0.33 else "middle" if ry < 0.66 else "bottom"
            pos_h = "left" if rx < 0.33 else "center" if rx < 0.66 else "right"
            attributes["position"] = f"{pos_v}-{pos_h}"
            det["relative_position"] = (rx, ry)
        return detections
