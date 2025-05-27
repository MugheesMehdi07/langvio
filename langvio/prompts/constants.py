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

# Default sample rate for video processing (every N frames)
DEFAULT_VIDEO_SAMPLE_RATE = 5

YOLO11_CONFIG = {
"model_path": "yolo11n.pt" ,
"confidence" : 0.3
}
