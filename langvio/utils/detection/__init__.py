# langvio/utils/detection/__init__.py
"""
Detection processing utilities
"""

from langvio.utils.detection.compression import (
    compress_detections_for_output,
    identify_object_clusters,
)
from langvio.utils.detection.extraction import (
    add_color_attributes,
    add_size_and_position_attributes,
    add_tracking_info,
    add_unified_attributes,
    extract_detections,
    optimize_for_memory,
)

__all__ = [
    "extract_detections",
    "optimize_for_memory",
    "add_unified_attributes",
    "add_tracking_info",
    "add_color_attributes",
    "add_size_and_position_attributes",
    "compress_detections_for_output",
    "identify_object_clusters",
]
