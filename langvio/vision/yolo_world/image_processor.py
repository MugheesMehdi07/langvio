"""
YOLO-World image processing module
"""

import logging
from typing import Any, Dict, List

import torch
import cv2

from langvio.utils.detection import (
    optimize_for_memory,
    compress_detections_for_output,
    add_unified_attributes,
)


class YOLOWorldImageProcessor:
    """Handles image processing with YOLO-World models"""

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)

    def process(self, image_path: str, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Process an image with YOLO-World model"""
        self.logger.info(f"Processing image: {image_path}")

        with torch.no_grad():
            try:
                optimize_for_memory()

                # Get image dimensions
                image_dimensions = self._get_image_dimensions(image_path)
                if not image_dimensions:
                    return {"objects": [], "error": "Could not read image dimensions"}

                width, height = image_dimensions

                # Run detection with attributes
                detections = self._run_detection_with_attributes(
                    image_path, width, height, query_params
                )

                # Create compressed results
                compressed_objects = compress_detections_for_output(
                    detections, is_video=False
                )
                summary = self._create_image_summary(
                    detections, width, height, query_params
                )

                return {"objects": compressed_objects, "summary": summary}

            except Exception as e:
                self.logger.error(f"Error processing image: {e}")
                return {"objects": [], "error": str(e)}

    def _get_image_dimensions(self, image_path: str):
        """Get image dimensions"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            height, width = image.shape[:2]
            return width, height
        except Exception as e:
            self.logger.error(f"Error reading image dimensions: {e}")
            return None

    def _run_detection_with_attributes(
        self, image_path: str, width: int, height: int, query_params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Run YOLO-World detection with attributes"""
        try:
            # Run YOLO-World detection
            results = self.model(
                image_path, conf=self.config["confidence"], verbose=False
            )

            # Extract detections
            detections = self._extract_detections(results[0])

            # Determine what attributes are needed
            needs_color = any(
                attr.get("attribute") == "color"
                for attr in query_params.get("attributes", [])
            )
            needs_spatial = bool(query_params.get("spatial_relations", []))
            needs_size = True  # Always include size

            # Add unified attributes
            detections = add_unified_attributes(
                detections,
                width,
                height,
                image_path,  # Pass image path for color detection
                needs_color,
                needs_spatial,
                needs_size,
                is_video_frame=False,
            )

            return detections

        except Exception as e:
            self.logger.error(f"Error in detection: {e}")
            return []

    def _extract_detections(self, result) -> List[Dict[str, Any]]:
        """Extract detections from YOLO-World results"""
        detections = []

        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                x1, y1, x2, y2 = map(int, box)
                class_name = result.names[cls_id]

                detections.append(
                    {
                        "label": class_name,
                        "confidence": float(conf),
                        "bbox": [x1, y1, x2, y2],
                        "class_id": int(cls_id),
                    }
                )

        return detections

    def _create_image_summary(
        self,
        detections: List[Dict[str, Any]],
        width: int,
        height: int,
        query_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create summary for image analysis"""
        object_counts: Dict[str, int] = {}
        for det in detections:
            label = det["label"]
            object_counts[label] = object_counts.get(label, 0) + 1

        return {
            "total_objects": len(detections),
            "object_counts": object_counts,
            "image_dimensions": {"width": width, "height": height},
            "query_type": query_params.get("task_type", "identification"),
        }
