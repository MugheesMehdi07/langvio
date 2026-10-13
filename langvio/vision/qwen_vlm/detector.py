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
from typing import Any, Dict, List, Optional, Tuple

import cv2
from PIL import Image

from langvio.llm.qwen_api import QwenAPIClient, bbox_space, resolve_backend
from langvio.llm.qwen_loader import DEFAULT_VLM_MODEL, GENERATION_LOCK, load_qwen
from langvio.utils.detection import (
    add_unified_attributes,
    compress_detections_for_output,
)
from langvio.vision.base import BaseVisionProcessor
from langvio.vision.qwen_vlm.summary import build_image_summary, build_video_summary
from langvio.vision.qwen_vlm.tracking import SparseTracker, interpolate_tracks, iou

# Qwen2.5-VL patches are 14px and merged 2x2, so the model input is a multiple of 28.
_PATCH_SIZE = 14

_BBOX_PATTERN = re.compile(
    r'"bbox_2d"\s*:\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,'
    r'\s*([-\d.]+)\s*\]\s*,?\s*(?:"label"\s*:\s*"([^"]*)")?'
)

_GENERIC_TARGETS = {"object", "objects", "thing", "things", "item", "items"}


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
                api_key_env, api_extra_body, api_timeout).
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
                [{"role": "user", "content": self.api_client.user_content(images, text)}],
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
        trimmed = generated[:, inputs["input_ids"].shape[1]:]
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
        if bbox_space(self.model_name, str(self.config.get("bbox_format", "auto"))) == "norm1000":
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
        if targets:
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

        return self._parse_boxes(
            text, orig_w / input_w, orig_h / input_h, orig_w, orig_h
        )

    def _describe_scene(
        self, images: List[Image.Image], query_params: Dict[str, Any], is_video: bool
    ) -> str:
        if not self.config.get("scene_description", True) or not images:
            return ""
        query = query_params.get("query", "")
        focus = f' The user asked: "{query}".' if query else ""
        if is_video:
            intro = (
                f"These {len(images)} frames are sampled in order from one video."
            )
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

        detections: List[Dict[str, Any]] = []
        confidence = float(self.config.get("confidence", 0.9))
        min_side = max(4, int(0.004 * max(width, height)))
        for bbox, label in raw:
            try:
                x1, y1, x2, y2 = (float(v) for v in bbox)
            except (TypeError, ValueError):
                continue
            x1 = int(max(0, min(width - 1, x1 * scale_x)))
            y1 = int(max(0, min(height - 1, y1 * scale_y)))
            x2 = int(max(0, min(width, x2 * scale_x)))
            y2 = int(max(0, min(height, y2 * scale_y)))
            if x2 - x1 < min_side or y2 - y1 < min_side:
                continue
            label = label.strip().lower() or "object"
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
    def _nms(detections: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
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

            detections = self._ground(image, query_params)
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

            return {
                "objects": compress_detections_for_output(detections, is_video=False),
                "summary": build_image_summary(
                    detections, width, height, query_params, scene
                ),
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

        tracker = SparseTracker(
            max_missed=int(self.config.get("track_max_missed", 2)),
            distance_gate=float(self.config.get("track_distance_gate", 1.2)),
        )
        keyframes: Dict[str, List[Dict[str, Any]]] = {}
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

        # Grounding works on the downscaled frames, so map boxes back up.
        def ground_frame(
            item: Tuple[int, Image.Image],
        ) -> Optional[List[Dict[str, Any]]]:
            frame_id, image = item
            try:
                detections = self._ground(image, query_params)
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
            for (frame_id, _), detections in zip(frames, results):
                if detections is None:
                    continue
                detections = self._add_basic_attributes(detections, width, height)
                keyframes[str(frame_id)] = tracker.update(detections, frame_id)
        except Exception as e:
            self.logger.error(f"Error processing video: {e}")
            return {"error": str(e)}

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
        render = interpolate_tracks(tracker, total_frames, step)

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
