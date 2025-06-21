"""
Core detection extraction and enhancement utilities
"""

import gc
import os
from typing import Any, Dict, List

import cv2
import numpy as np
import torch

from langvio.llm.factory import logger


def optimize_for_memory():
    try:
        # Set environment variables for memory optimization
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
            "expandable_segments:True,max_split_size_mb:128"
        )

        # Check if CUDA is available and working
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            try:
                # Test CUDA functionality
                # test_tensor = torch.zeros(1, device="cuda")
                # del test_tensor

                # Clear cache
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                # Set memory management
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = False  # More stable for varying sizes
                torch.backends.cudnn.deterministic = True

                logger.info("CUDA memory optimization successful")

            except Exception as e:
                logger.warning(f"CUDA optimization failed: {e}, falling back to CPU")
                cuda_available = False
                force_cpu_mode()

        # CPU optimizations
        if not cuda_available:
            torch.set_num_threads(min(4, os.cpu_count()))  # Limit CPU threads
            torch.set_grad_enabled(False)

        # General PyTorch optimizations
        torch.set_grad_enabled(False)

        # Python garbage collection
        gc.collect()

    except Exception as e:
        logger.error(f"Memory optimization failed: {e}")


def force_cpu_mode():
    """Force CPU mode when CUDA fails"""
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        logger.info("Forced CPU mode")
    except Exception as e:
        logger.warning(f"Error forcing CPU mode: {e}")


def extract_detections(results) -> List[Dict[str, Any]]:
    """Extract detections from YOLO results with basic attributes"""
    detections = []

    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = result.names[cls_id]

            # Basic detection object
            detections.append(
                {
                    "label": label,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "class_id": cls_id,
                }
            )

    return detections


def add_unified_attributes(
    detections: List[Dict[str, Any]],
    width: int,
    height: int,
    input_data: Any,  # image_path (str) or frame (np.ndarray)
    needs_color: bool,
    needs_spatial: bool,
    needs_size: bool,
    is_video_frame: bool,
) -> List[Dict[str, Any]]:
    """
    Unified method to add attributes to detections.
    Works for both images and video frames.
    """
    if not detections:
        return detections

    # Get image data for color detection if needed

    needs_color = not is_video_frame  # setting color detection false
    image_data = None
    if needs_color:
        if is_video_frame:
            image_data = input_data  # input_data is already a frame
        else:
            try:
                image_data = cv2.imread(input_data)  # input_data is image path
            except Exception:
                pass

    # Process each detection
    enhanced_detections = []
    for i, det in enumerate(detections):
        if "bbox" not in det:
            enhanced_detections.append(det)
            continue

        x1, y1, x2, y2 = det["bbox"]

        # Skip invalid boxes
        if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            enhanced_detections.append(det)
            continue

        # Add basic position info (always needed for tracking)
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        det["center"] = (center_x, center_y)
        det["object_id"] = f"obj_{i}"

        # Initialize attributes
        attributes = {}

        # Add size attributes if needed
        if needs_size:
            area = (x2 - x1) * (y2 - y1)
            relative_size = area / (width * height)
            attributes["size"] = (
                "small"
                if relative_size < 0.05
                else "medium" if relative_size < 0.25 else "large"
            )
            attributes["relative_size"] = relative_size

        # Add position attributes if needed (for spatial queries)
        if needs_spatial:
            rx, ry = center_x / width, center_y / height
            pos_v = "top" if ry < 0.33 else "middle" if ry < 0.66 else "bottom"
            pos_h = "left" if rx < 0.33 else "center" if rx < 0.66 else "right"
            attributes["position"] = f"{pos_v}-{pos_h}"

            # Add relative position for advanced spatial analysis
            det["relative_position"] = (rx, ry)

        # Add color attributes if needed (expensive)
        if needs_color and image_data is not None:
            try:
                obj_region = image_data[int(y1) : int(y2), int(x1) : int(x2)]
                if obj_region.size > 0:
                    from langvio.vision.color_detection import ColorDetector

                    color_info = ColorDetector.get_color_profile(obj_region)
                    attributes["color"] = color_info["dominant_color"]
                    attributes["is_multicolored"] = color_info["is_multicolored"]
            except Exception:
                attributes["color"] = "unknown"

        det["attributes"] = attributes
        enhanced_detections.append(det)

    # Add spatial relationships if needed (expensive)
    if needs_spatial and len(enhanced_detections) > 1:
        from langvio.utils.spatial.relationships import add_spatial_relationships

        enhanced_detections = add_spatial_relationships(enhanced_detections)

    return enhanced_detections


def add_tracking_info(
    detections: List[Dict[str, Any]], frame_idx: int
) -> List[Dict[str, Any]]:
    """Add tracking information to detections"""
    for i, det in enumerate(detections):
        if "track_id" not in det:
            det["track_id"] = f"track_{frame_idx}_{i}"
        if "object_id" not in det:
            det["object_id"] = f"obj_{frame_idx}_{i}"
    return detections


def add_color_attributes(
    detections: List[Dict[str, Any]], frame: np.ndarray, needs_color: bool
) -> List[Dict[str, Any]]:
    """Add color attributes to detections (optimized for video)"""
    if not needs_color or frame is None:
        return detections

    from langvio.vision.color_detection import ColorDetector

    for det in detections:
        if "bbox" not in det:
            continue

        try:
            x1, y1, x2, y2 = map(int, det["bbox"])
            if x1 >= x2 or y1 >= y2:
                continue

            # Extract object region
            obj_region = frame[y1:y2, x1:x2]
            if obj_region.size > 0:
                # Get dominant color only (faster than full profile)
                dominant_color = ColorDetector.detect_color(
                    obj_region, return_all=False
                )

                if "attributes" not in det:
                    det["attributes"] = {}
                det["attributes"]["color"] = dominant_color
        except Exception:
            continue  # Skip on error

    return detections


def add_size_and_position_attributes(
    detections: List[Dict[str, Any]], width: int, height: int
) -> List[Dict[str, Any]]:
    """Add size and position attributes (fast computation)"""
    image_area = width * height

    for det in detections:
        if "bbox" not in det:
            continue

        x1, y1, x2, y2 = det["bbox"]

        # Calculate center and size
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        det["center"] = (center_x, center_y)

        # Size attribute
        area = (x2 - x1) * (y2 - y1)
        relative_size = area / image_area

        if "attributes" not in det:
            det["attributes"] = {}

        det["attributes"]["size"] = (
            "small"
            if relative_size < 0.05
            else "medium" if relative_size < 0.25 else "large"
        )

        # Position attribute
        rx, ry = center_x / width, center_y / height
        pos_v = "top" if ry < 0.33 else "middle" if ry < 0.66 else "bottom"
        pos_h = "left" if rx < 0.33 else "center" if rx < 0.66 else "right"
        det["attributes"]["position"] = f"{pos_v}-{pos_h}"

    return detections
