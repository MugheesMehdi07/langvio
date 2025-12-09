"""
YOLO-World vision processor - main coordinator
"""

import logging
from typing import Any, Dict

import torch
import cv2

from langvio.prompts.constants import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_VIDEO_SAMPLE_RATE,
)
from langvio.vision.base import BaseVisionProcessor
from langvio.vision.yolo_world.video_processor import YOLOWorldVideoProcessor
from langvio.vision.yolo_world.image_processor import YOLOWorldImageProcessor


class YOLOWorldProcessor(BaseVisionProcessor):
    """Main YOLO-World processor - coordinates image and video processing"""

    def __init__(
        self,
        name: str,
        model_name: str = "yolov8m-worldv2",
        confidence: float = DEFAULT_CONFIDENCE_THRESHOLD,
        **kwargs,
    ):
        """Initialize YOLO-World processor"""
        config = {
            "model_name": model_name,
            "confidence": confidence,
            **kwargs,
        }
        super().__init__(name, config)
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.model_name = model_name

    def initialize(self) -> bool:
        """Initialize the YOLO-World model with optimizations"""
        try:
            self.logger.info(f"Loading YOLO-World model: {self.model_name}")

            # Enable aggressive GPU optimizations
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.cuda.empty_cache()

                # Set memory fraction for faster processing
                torch.cuda.set_per_process_memory_fraction(0.8)

                # Enable memory efficient attention
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)

            # Disable gradients globally
            torch.set_grad_enabled(False)

            # Load YOLO-World model
            try:
                from ultralytics import YOLOWorld

                self.model = YOLOWorld(self.model_name)
            except ImportError:
                self.logger.error(
                    "YOLO-World not available. Install with: pip install ultralytics>=8.0.0"
                )
                return False

            # Move to GPU and enable half precision if available
            if torch.cuda.is_available():
                self.model.to("cuda")
                try:
                    self.model.half()  # Enable FP16 for 2x speed boost
                    self.logger.info("[OK] Half precision (FP16) enabled")
                except Exception:
                    self.logger.info("[WARN] Half precision not available, using FP32")

            # Warm up the model
            self._warmup_model()

            return True
        except Exception as e:
            self.logger.error(f"Error loading YOLO-World model: {e}")
            return False

    def _warmup_model(self):
        """Warm up model for consistent performance"""
        try:
            dummy_input = torch.randn(1, 3, 640, 640)
            if torch.cuda.is_available():
                dummy_input = dummy_input.cuda()
                if (
                    hasattr(self.model, "model")
                    and next(self.model.model.parameters()).dtype == torch.float16
                ):
                    dummy_input = dummy_input.half()

            with torch.no_grad():
                for _ in range(3):  # 3 warmup runs
                    self.model(dummy_input, verbose=False)
            self.logger.info(f"[OK] Model warmed up: {self.model_name}")
        except Exception as e:
            self.logger.warning(f"Warmup failed: {e}")

    def process_image(
        self, image_path: str, query_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process an image - delegate to image processor"""
        if not self.model:
            self.initialize()

        processor = YOLOWorldImageProcessor(self.model, self.config)
        return processor.process(image_path, query_params)

    def process_video(
        self,
        video_path: str,
        query_params: Dict[str, Any],
        sample_rate: int = 3,
    ) -> Dict[str, Any]:
        """Process a video - delegate to video processor"""
        if not self.model:
            self.initialize()

        processor = YOLOWorldVideoProcessor(self.model, self.config, self.model_name)
        return processor.process(video_path, query_params, sample_rate)

    def set_classes(self, classes: list):
        """Set classes for YOLO-World to detect"""
        if self.model:
            try:
                self.model.set_classes(classes)
                self.logger.info(f"Set classes: {classes}")
            except Exception as e:
                self.logger.error(f"Error setting classes: {e}")
