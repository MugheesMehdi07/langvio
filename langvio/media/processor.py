"""
Main media processing coordinator
"""

import logging
import os
from typing import Any, Dict, List, Union

from langvio.media.image_visualizer import ImageVisualizer
from langvio.media.video_visualizer import VideoVisualizer


class MediaProcessor:
    """Main media processor - coordinates image and video visualization"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize media processor"""
        self.config = config or {
            "output_dir": "./output",
            "temp_dir": "./temp",
            "visualization": {
                "box_color": [0, 255, 0],
                "text_color": [255, 255, 255],
                "line_thickness": 2,
                "show_attributes": True,
                "show_confidence": True,
            },
        }

        self.logger = logging.getLogger(__name__)

        # Create output and temp directories
        os.makedirs(self.config["output_dir"], exist_ok=True)
        os.makedirs(self.config["temp_dir"], exist_ok=True)

        # Initialize visualizers
        self.image_visualizer = ImageVisualizer(self.config)
        self.video_visualizer = VideoVisualizer(self.config)

    def update_config(self, config: Dict[str, Any]) -> None:
        """Update configuration parameters"""
        self.config.update(config)

        # Ensure directories exist
        os.makedirs(self.config["output_dir"], exist_ok=True)
        os.makedirs(self.config["temp_dir"], exist_ok=True)

        # Update visualizers
        self.image_visualizer.config = self.config
        self.video_visualizer.config = self.config

    def is_video(self, file_path: str) -> bool:
        """Check if a file is a video based on extension"""
        video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
        _, ext = os.path.splitext(file_path.lower())
        return ext in video_extensions

    def get_output_path(self, input_path: str, suffix: str = "_processed") -> str:
        """Generate an output path for processed media"""
        filename = os.path.basename(input_path)
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}{suffix}{ext}"

        return os.path.join(self.config["output_dir"], output_filename)

    def visualize_image_with_highlights(
            self,
            image_path: str,
            output_path: str,
            all_detections: List[Dict[str, Any]],
            highlighted_detections: List[Dict[str, Any]],
            **kwargs
    ) -> None:
        """Delegate image visualization to ImageVisualizer"""
        return self.image_visualizer.visualize_with_highlights(
            image_path, output_path, all_detections, highlighted_detections, **kwargs
        )

    def visualize_video_with_highlights(
            self,
            video_path: str,
            output_path: str,
            all_frame_detections: Dict[str, List[Dict[str, Any]]],
            highlighted_objects: List[Dict[str, Any]],
            **kwargs
    ) -> None:
        """Delegate video visualization to VideoVisualizer"""
        return self.video_visualizer.visualize_with_highlights(
            video_path, output_path, all_frame_detections, highlighted_objects, **kwargs
        )