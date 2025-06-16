"""
Image visualization module
"""

import logging
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np


class ImageVisualizer:
    """Handles image visualization with detection overlays"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def visualize_with_highlights(
            self,
            image_path: str,
            output_path: str,
            all_detections: List[Dict[str, Any]],
            highlighted_detections: List[Dict[str, Any]],
            original_box_color: Union[Tuple[int, int, int], List[int]] = (0, 255, 0),
            highlight_color: Union[Tuple[int, int, int], List[int]] = (0, 0, 255),
            text_color: Union[Tuple[int, int, int], List[int]] = (255, 255, 255),
            line_thickness: int = 2,
            show_attributes: bool = True,
            show_confidence: bool = True,
    ) -> None:
        """Visualize all detections on an image
        with highlighted objects in a different color"""
        self.logger.info(
            f"Visualizing {len(all_detections)} detections on image: {image_path}"
        )

        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")

            # Create set of highlighted detections for quick lookup
            highlighted_signatures = set()
            for det in highlighted_detections:
                if "bbox" in det and "label" in det:
                    signature = (
                        det["label"],
                        (
                            tuple(det["bbox"])
                            if isinstance(det["bbox"], list)
                            else det["bbox"]
                        ),
                    )
                    highlighted_signatures.add(signature)

            # Draw all detections with appropriate colors
            for det in all_detections:
                # Check if this detection is in the highlighted set
                is_highlighted = False
                if "bbox" in det and "label" in det:
                    signature = (
                        det["label"],
                        (
                            tuple(det["bbox"])
                            if isinstance(det["bbox"], list)
                            else det["bbox"]
                        ),
                    )
                    is_highlighted = signature in highlighted_signatures

                # Choose color based on whether the detection is highlighted
                box_color = highlight_color if is_highlighted else original_box_color

                # Use thicker lines for highlighted objects
                thickness = line_thickness + 1 if is_highlighted else line_thickness

                # Draw the detection with the chosen color and thickness
                image = self._draw_single_detection(
                    image,
                    det,
                    box_color,
                    text_color,
                    thickness,
                    show_attributes,
                    show_confidence,
                    is_highlighted,
                )

            # Save output
            cv2.imwrite(output_path, image)
            self.logger.info(f"Saved visualized image to: {output_path}")
        except Exception as e:
            self.logger.error(f"Error visualizing image: {e}")

    def _draw_single_detection(
            self,
            image: np.ndarray,
            det: Dict[str, Any],
            box_color: Union[Tuple[int, int, int], List[int]],
            text_color: Union[Tuple[int, int, int], List[int]],
            line_thickness: int,
            show_attributes: bool,
            show_confidence: bool,
            is_highlighted: bool = False,
    ) -> np.ndarray:
        """Draw a single detection on an image"""
        if "bbox" not in det:
            return image  # Skip detections without bounding boxes

        # Extract bounding box
        x1, y1, x2, y2 = det["bbox"]

        # Make sure coordinates are integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Check for valid box dimensions
        if x2 <= x1 or y2 <= y1:
            return image  # Skip invalid boxes

        # Create label based on configuration
        label_parts = [det["label"]]

        # Add confidence if requested
        if show_confidence and "confidence" in det:
            conf = det["confidence"]
            if isinstance(conf, (int, float)):
                label_parts.append(f"{conf:.2f}")

        # Add attributes if requested and present (limit to 2 most important)
        if show_attributes and "attributes" in det and det["attributes"]:
            # Prioritize color and size attributes
            priority_attrs = []
            for key in ["color", "size"]:
                if key in det["attributes"]:
                    priority_attrs.append(f"{key}:{det['attributes'][key]}")

            # Add up to 2 priority attributes to avoid cluttering
            if priority_attrs:
                label_parts.extend(priority_attrs[:2])

        # Add activities if present (limit to 1 most important)
        if "activities" in det and det["activities"] and len(det["activities"]) > 0:
            # Only add the first activity to avoid cluttering
            label_parts.append(f"[{det['activities'][0]}]")

        # Add "HIGHLIGHT" tag if this is a highlighted detection
        if is_highlighted:
            label_parts.append("*")

        # Combine into label
        label = " | ".join(label_parts)

        # Draw bounding box with line thickness scaled by image size
        thickness = max(
            1, min(line_thickness, int(min(image.shape[0], image.shape[1]) / 500))
        )
        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, thickness)

        # Calculate text size and scale font size based on image dimensions
        font_scale = max(0.3, min(0.5, min(image.shape[0], image.shape[1]) / 1000))
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(label, font, font_scale, 1)[0]

        # Draw text background with slight transparency
        alpha = 0.6  # Transparency factor
        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (x1, y1 - text_size[1] - 5),
            (x1 + text_size[0], y1),
            box_color,
            -1,
        )
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        # Draw text
        cv2.putText(image, label, (x1, y1 - 5), font, font_scale, text_color, 1)

        return image
