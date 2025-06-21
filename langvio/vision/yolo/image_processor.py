"""
YOLO image processing module
"""

import logging
from typing import Any, Dict, List

import torch

from langvio.utils.detection import (
    optimize_for_memory,
    compress_detections_for_output,
    extract_detections,
    identify_object_clusters,
    add_unified_attributes,
)


class YOLOImageProcessor:
    """Handles image processing with YOLO models"""

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)

    def process(self, image_path: str, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Process an image with YOLO model"""
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
            import cv2

            image = cv2.imread(image_path)
            if image is not None:
                height, width = image.shape[:2]
                return (width, height)
        except Exception:
            pass
        return None

    def _run_detection_with_attributes(
        self, image_path: str, width: int, height: int, query_params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Run YOLO detection with attributes for images"""
        # 1. Run YOLO detection
        results = self.model(image_path, conf=self.config["confidence"])
        detections = extract_detections(results)

        if not detections:
            return []

        # 2. For images, add all attributes (not video frames)
        detections = add_unified_attributes(
            detections,
            width,
            height,
            image_path,
            needs_color=True,
            needs_spatial=True,
            needs_size=True,
            is_video_frame=False,
        )

        return detections

    def _create_image_summary(
        self,
        detections: List[Dict[str, Any]],
        width: int,
        height: int,
        query_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create optimized image summary"""
        if not detections:
            return {
                "image_info": {"resolution": f"{width}x{height}", "total_objects": 0},
                "analysis": "No objects detected",
            }

        # Analyze patterns
        types = {}
        positions = {}
        sizes = {}
        colors = {}

        for det in detections:
            # Count types
            types[det["label"]] = types.get(det["label"], 0) + 1

            # Count positions (if available)
            pos = det.get("attributes", {}).get("position", "unknown")
            if pos != "unknown":
                positions[pos] = positions.get(pos, 0) + 1

            # Count sizes (if available)
            size = det.get("attributes", {}).get("size", "unknown")
            if size != "unknown":
                sizes[size] = sizes.get(size, 0) + 1

            # Count colors (if available)
            color = det.get("attributes", {}).get("color", "unknown")
            if color != "unknown":
                colors[color] = colors.get(color, 0) + 1

        # Identify notable patterns
        notable_patterns = []

        # Check for clusters
        if len(detections) > 3:
            clusters = identify_object_clusters(detections)
            if clusters:
                notable_patterns.append(f"Objects form {len(clusters)} main clusters")

        # Check for dominant type
        if types:
            dominant_type = max(types.items(), key=lambda x: x[1])
            if dominant_type[1] > len(detections) * 0.4:
                notable_patterns.append(
                    f"{dominant_type[0]} is dominant ({dominant_type[1]} instances)"
                )

        return {
            "image_info": {
                "resolution": f"{width}x{height}",
                "total_objects": len(detections),
                "unique_types": len(types),
            },
            "object_distribution": {
                "by_type": types,
                "by_position": positions if positions else None,
                "by_size": sizes if sizes else None,
                "by_color": colors if colors else None,
            },
            "notable_patterns": notable_patterns,
            "query_context": query_params.get("task_type", "identification"),
        }
