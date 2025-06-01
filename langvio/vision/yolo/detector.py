"""
Clean YOLO-based vision processor - main coordinator
"""

import logging
from typing import Any, Dict
from ultralytics import YOLO, YOLOE

from langvio.prompts.constants import DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_VIDEO_SAMPLE_RATE
from langvio.vision.base import BaseVisionProcessor
from langvio.vision.yolo.image_processor import YOLOImageProcessor
from langvio.vision.yolo.video_processor import YOLOVideoProcessor
from langvio.vision.yolo.yolo11_utils import check_yolo11_solutions_available


class YOLOProcessor(BaseVisionProcessor):
    """Main YOLO processor - coordinates image and video processing"""

    def __init__(self, name: str, model_path: str, confidence: float = DEFAULT_CONFIDENCE_THRESHOLD, **kwargs):
        """Initialize YOLO processor"""
        config = {
            "model_path": model_path,
            "confidence": confidence,
            **kwargs,
        }
        super().__init__(name, config)
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.model_type = kwargs.get("model_type", "yolo")

        # Check YOLO11 Solutions availability
        self.has_yolo11_solutions = check_yolo11_solutions_available()
        if self.has_yolo11_solutions:
            self.logger.info("YOLO11 Solutions is available for metrics")
        else:
            self.logger.info("YOLO11 Solutions not available - using basic detection only")

    def initialize(self) -> bool:
        """Initialize the YOLO model"""
        try:
            self.logger.info(f"Loading {self.model_type} model: {self.config['model_path']}")

            if self.model_type == "yoloe":
                self.model = YOLOE(self.config['model_path'])
            else:
                self.model = YOLO(self.config['model_path'])

            return True
        except Exception as e:
            self.logger.error(f"Error loading YOLO model: {e}")
            return False

    def process_image(self, image_path: str, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Process an image - delegate to image processor"""
        if not self.model:
            self.initialize()

        processor = YOLOImageProcessor(self.model, self.config)
        return processor.process(image_path, query_params)

    def process_video(
            self, video_path: str, query_params: Dict[str, Any],
            sample_rate: int = DEFAULT_VIDEO_SAMPLE_RATE
    ) -> Dict[str, Any]:
        """Process a video - delegate to video processor"""
        if not self.model:
            self.initialize()

        processor = YOLOVideoProcessor(self.model, self.config, self.has_yolo11_solutions)
        return processor.process(video_path, query_params, sample_rate)