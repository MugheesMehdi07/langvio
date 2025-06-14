"""
Constants for langvio package
"""

# Task types that the system can handle
TASK_TYPES = [
    "identification",  # Basic object detection
    "counting",  # Counting specific objects
    "verification",  # Verifying existence of objects
    "analysis",  # Detailed analysis with attributes and relationships
    "tracking",  # For tracking objects across video frames
    "activity",  # For detecting activities/actions
]

# Default detection confidence threshold
DEFAULT_CONFIDENCE_THRESHOLD = 0.25

# Default IoU threshold for NMS
DEFAULT_IOU_THRESHOLD = 0.5

# Enhanced configuration for video processing
DEFAULT_VIDEO_SAMPLE_RATE = 2  # Process every 2nd frame for speed
DEFAULT_COLOR_ANALYSIS_INTERVAL = 3  # Color analysis every 3rd frame
DEFAULT_SPATIAL_UPDATE_INTERVAL = 2  # Spatial analysis every 2nd frame

# YOLO11 optimized configuration
YOLO11_CONFIG = {
    "model_path": "yolo11n.pt",
    "confidence": 0.3,
    "show": False,
    "verbose": False,
}
