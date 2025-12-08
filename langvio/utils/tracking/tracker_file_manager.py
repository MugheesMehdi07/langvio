"""
Tracker file manager for saving and loading detection and tracking data
"""

import json
import os
import logging
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class TrackerFileManager:
    """Manages tracker file I/O for YOLO-World + ByteTracker integration"""

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def create_tracker_file_path(self, video_path: str) -> str:
        """Create tracker file path based on video path"""
        video_name = Path(video_path).stem
        tracker_filename = f"{video_name}_tracker.json"
        return str(self.output_dir / tracker_filename)

    def save_tracker_data(
        self,
        video_path: str,
        detections: List[Dict[str, Any]],
        tracks: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        query: str = "",
    ) -> str:
        """
        Save tracker data to JSON file

        Args:
            video_path: Original video path
            detections: List of detections per frame
            tracks: List of track objects
            metadata: Video and model metadata
            query: Original query for context

        Returns:
            Path to saved tracker file
        """
        tracker_file_path = self.create_tracker_file_path(video_path)

        # Prepare tracker data structure
        tracker_data = {
            "metadata": {
                "video_path": video_path,
                "query": query,
                "timestamp": metadata.get("timestamp"),
                "model_info": metadata.get("model_info", {}),
                "tracker_info": metadata.get("tracker_info", {}),
                **metadata,
            },
            "detections": detections,
            "tracks": tracks,
        }

        try:
            with open(tracker_file_path, "w") as f:
                json.dump(tracker_data, f, indent=2, default=str)

            self.logger.info(f"Tracker data saved to: {tracker_file_path}")
            return tracker_file_path

        except Exception as e:
            self.logger.error(f"Error saving tracker data: {e}")
            raise

    def load_tracker_data(self, tracker_file_path: str) -> Dict[str, Any]:
        """
        Load tracker data from JSON file

        Args:
            tracker_file_path: Path to tracker file

        Returns:
            Loaded tracker data
        """
        try:
            with open(tracker_file_path, "r") as f:
                tracker_data = json.load(f)

            self.logger.info(f"Tracker data loaded from: {tracker_file_path}")
            return tracker_data

        except Exception as e:
            self.logger.error(f"Error loading tracker data: {e}")
            raise

    def convert_to_legacy_format(self, tracker_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert tracker file format to legacy detection format for compatibility

        Args:
            tracker_data: Loaded tracker data

        Returns:
            Legacy format detection data
        """
        detections = tracker_data.get("detections", [])
        metadata = tracker_data.get("metadata", {})

        # Convert to frame_detections format expected by existing code
        frame_detections = {}

        for frame_data in detections:
            frame_id = str(frame_data["frame_id"])
            frame_detections[frame_id] = frame_data["objects"]

        # Create legacy format result
        legacy_result = {
            "frame_detections": frame_detections,
            "summary": {
                "total_frames": len(detections),
                "video_metadata": metadata,
                "tracking_enabled": True,
            },
            "tracks": tracker_data.get("tracks", []),
            "metadata": metadata,
        }

        return legacy_result

    def get_tracker_file_if_exists(self, video_path: str) -> Optional[str]:
        """Check if tracker file already exists for video"""
        tracker_file_path = self.create_tracker_file_path(video_path)
        return tracker_file_path if os.path.exists(tracker_file_path) else None

    def cleanup_old_tracker_files(self, max_age_days: int = 7):
        """Clean up old tracker files"""
        try:
            import time

            current_time = time.time()
            max_age_seconds = max_age_days * 24 * 60 * 60

            for file_path in self.output_dir.glob("*_tracker.json"):
                if current_time - file_path.stat().st_mtime > max_age_seconds:
                    file_path.unlink()
                    self.logger.info(f"Cleaned up old tracker file: {file_path}")

        except Exception as e:
            self.logger.warning(f"Error cleaning up tracker files: {e}")
