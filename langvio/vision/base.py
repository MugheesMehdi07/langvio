"""
Enhanced base classes for vision processors
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import cv2

from langvio.core.base import Processor
from langvio.prompts.constants import DEFAULT_VIDEO_SAMPLE_RATE
from langvio.vision.color_detection import ColorDetector


class BaseVisionProcessor(Processor):
    """Enhanced base class for all vision processors"""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize vision processor.

        Args:
            name: Processor name
            config: Configuration parameters
        """
        super().__init__(name, config)
        self.model = None

    @abstractmethod
    def process_image(
        self, image_path: str, query_params: Dict[str, Any]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process an image with the vision model.

        Args:
            image_path: Path to the input image
            query_params: Parameters from the query processor

        Returns:
            Dictionary with detection results
        """

    @abstractmethod
    def process_video(
        self,
        video_path: str,
        query_params: Dict[str, Any],
        sample_rate: int = DEFAULT_VIDEO_SAMPLE_RATE,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process a video with the vision model.

        Args:
            video_path: Path to the input video
            query_params: Parameters from the query processor
            sample_rate: Process every Nth frame

        Returns:
            Dictionary with detection results
        """

    def _get_image_dimensions(self, image_path: str) -> Optional[Tuple[int, int]]:
        """
        Get dimensions of an image.

        Args:
            image_path: Path to the image

        Returns:
            Tuple of (width, height) or None if failed
        """
        try:
            image = cv2.imread(image_path)
            if image is not None:
                height, width = image.shape[:2]
                return (width, height)
        except Exception:
            pass
        return None

    def _enhance_detections_with_attributes(
        self, detections: List[Dict[str, Any]], image_path: str
    ) -> List[Dict[str, Any]]:
        """
        Enhance detections with attribute information.
        Subclasses can override this to add more sophisticated attribute detection.

        Args:
            detections: List of detection dictionaries
            image_path: Path to the image

        Returns:
            Detections with added attributes
        """
        # Load image
        try:
            image = cv2.imread(image_path)
            if image is None:
                return detections

            image_height, image_width = image.shape[:2]

            for det in detections:
                # Extract bounding box
                x1, y1, x2, y2 = det["bbox"]

                # Skip invalid boxes
                if (
                    x1 >= x2
                    or y1 >= y2
                    or x1 < 0
                    or y1 < 0
                    or x2 > image_width
                    or y2 > image_height
                ):
                    continue

                # Get the object region
                obj_region = image[y1:y2, x1:x2]

                # Initialize attributes dictionary if not present
                if "attributes" not in det:
                    det["attributes"] = {}

                # Calculate basic size attribute
                area = (x2 - x1) * (y2 - y1)
                image_area = image_width * image_height
                relative_size = area / image_area

                if relative_size < 0.05:
                    det["attributes"]["size"] = "small"
                elif relative_size < 0.25:
                    det["attributes"]["size"] = "medium"
                else:
                    det["attributes"]["size"] = "large"

                # Extract dominant color (very basic implementation)
                if obj_region.size > 0:
                    # Get color information
                    color_info = ColorDetector.get_color_profile(obj_region)

                    # Add to detection attributes
                    if "attributes" not in det:
                        det["attributes"] = {}

                    det["attributes"]["color"] = color_info["dominant_color"]
                    det["attributes"]["is_multicolored"] = color_info["is_multicolored"]

                    # Optionally add all detected colors
                    det["attributes"]["colors"] = list(
                        color_info["color_percentages"].keys()
                    )

        except Exception:
            # In case of any errors, return original detections
            pass

        return detections
