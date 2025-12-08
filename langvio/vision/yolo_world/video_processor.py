"""
YOLO-World video processing module with ByteTracker integration
"""

import logging
import time
from typing import Any, Dict, List, Tuple

import cv2
import torch
import numpy as np

from langvio.utils.tracking import TrackerFileManager, ByteTrackerManager
from langvio.utils.detection import optimize_for_memory


class YOLOWorldVideoProcessor:
    """Handles video processing with YOLO-World models and ByteTracker"""

    def __init__(self, model, config, model_name: str):
        self.model = model
        self.config = config
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)

        # Initialize tracker components
        self.tracker_file_manager = TrackerFileManager()
        self.byte_tracker = ByteTrackerManager(
            track_thresh=config.get("track_thresh", 0.3),
            track_buffer=config.get("track_buffer", 70),
            match_thresh=config.get("match_thresh", 0.6),
        )

    def process(
        self, video_path: str, query_params: Dict[str, Any], sample_rate: int
    ) -> Dict[str, Any]:
        """Process video with YOLO-World + ByteTracker integration"""
        self.logger.info(f"Processing video: {video_path}")

        # Check if tracker file already exists
        existing_tracker_file = self.tracker_file_manager.get_tracker_file_if_exists(video_path)
        if existing_tracker_file:
            self.logger.info(f"Using existing tracker file: {existing_tracker_file}")
            return self._load_and_convert_tracker_file(existing_tracker_file)

        # Process video with YOLO-World + ByteTracker
        return self._process_video_with_tracking(video_path, query_params, 2)

    def _process_video_with_tracking(
        self, video_path: str, query_params: Dict[str, Any], sample_rate: int
    ) -> Dict[str, Any]:
        """Process video with YOLO-World detection and ByteTracker"""
        start_time = time.time()

        # Initialize video capture
        cap, video_props = self._initialize_video_capture(video_path)
        width, height, fps, total_frames = video_props

        # Storage for tracking data
        all_detections = []
        frame_count = 0
        processed_frames = 0

        try:
            with torch.no_grad():
                optimize_for_memory()

                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        break

                    # Sample frames based on sample_rate
                    if sample_rate > 1 and frame_count % sample_rate != 0:
                        frame_count += 1
                        continue

                    # Process frame with YOLO-World
                    detections = self._run_detection(frame, width, height)

                    # Update ByteTracker
                    tracked_detections = self.byte_tracker.update(detections, frame_count)

                    # Store frame data
                    frame_data = {
                        "frame_id": frame_count,
                        "timestamp": frame_count / fps,
                        "objects": tracked_detections
                    }
                    all_detections.append(frame_data)

                    frame_count += 1
                    processed_frames += 1

                    # Log progress
                    if frame_count % 50 == 0:
                        self.logger.info(f"Processed {frame_count}/{total_frames} frames")

            # Get all tracks
            all_tracks = self.byte_tracker.get_all_tracks()

            # Create metadata
            metadata = {
                "width": width,
                "height": height,
                "fps": fps,
                "total_frames": total_frames,
                "processed_frames": processed_frames,
                "model_info": {
                    "model_name": self.model_name,
                    "confidence": self.config["confidence"]
                },
                "tracker_info": {
                    "track_thresh": self.byte_tracker.track_thresh,
                    "track_buffer": self.byte_tracker.track_buffer,
                    "match_thresh": self.byte_tracker.match_thresh
                },
                "timestamp": time.time()
            }

            # Save tracker data
            tracker_file_path = self.tracker_file_manager.save_tracker_data(
                video_path, all_detections, all_tracks, metadata, query_params.get("query", "")
            )

            # Convert to legacy format for compatibility
            legacy_result = self.tracker_file_manager.convert_to_legacy_format({
                "detections": all_detections,
                "tracks": all_tracks,
                "metadata": metadata
            })

            # Add tracker file path to result
            legacy_result["tracker_file_path"] = tracker_file_path

            # Performance summary
            total_time = time.time() - start_time
            self.logger.info("=== PERFORMANCE SUMMARY ===")
            self.logger.info(f"Total frames processed: {processed_frames}")
            self.logger.info(f"Total processing time: {total_time:.2f} seconds")
            self.logger.info(f"Processing FPS: {processed_frames/total_time:.2f}")
            self.logger.info(f"Tracker file saved: {tracker_file_path}")

            return legacy_result

        except Exception as e:
            self.logger.error(f"Error processing video: {e}")
            return {"error": str(e)}
        finally:
            if cap:
                cap.release()

    def _load_and_convert_tracker_file(self, tracker_file_path: str) -> Dict[str, Any]:
        """Load existing tracker file and convert to legacy format"""
        try:
            tracker_data = self.tracker_file_manager.load_tracker_data(tracker_file_path)
            legacy_result = self.tracker_file_manager.convert_to_legacy_format(tracker_data)
            legacy_result["tracker_file_path"] = tracker_file_path
            return legacy_result
        except Exception as e:
            self.logger.error(f"Error loading tracker file: {e}")
            return {"error": str(e)}

    def _initialize_video_capture(
        self, video_path: str
    ) -> Tuple[cv2.VideoCapture, Tuple[int, int, float, int]]:
        """Initialize video capture and extract video properties"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        return cap, (width, height, fps, total_frames)

    def _run_detection(self, frame: np.ndarray, width: int, height: int) -> List[Dict[str, Any]]:
        """Run YOLO-World detection on frame"""
        try:
            # Run YOLO-World detection
            results = self.model(frame, conf=self.config["confidence"], verbose=False)

            # Extract detections
            detections = self._extract_detections(results[0])

            # Add basic attributes
            detections = self._add_basic_attributes(detections, width, height)

            return detections

        except Exception as e:
            self.logger.warning(f"Error in detection: {e}")
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

                detections.append({
                    "label": class_name,
                    "confidence": float(conf),
                    "bbox": [x1, y1, x2, y2],
                    "class_id": int(cls_id),
                })

        return detections

    def _add_basic_attributes(self, detections: List[Dict[str, Any]], width: int, height: int) -> List[Dict[str, Any]]:
        """Add basic attributes to detections"""
        for det in detections:
            if "bbox" not in det:
                continue

            x1, y1, x2, y2 = det["bbox"]

            # Add center coordinates
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            det["center"] = (center_x, center_y)

            # Add size attributes
            area = (x2 - x1) * (y2 - y1)
            relative_size = area / (width * height)

            if "attributes" not in det:
                det["attributes"] = {}

            det["attributes"]["size"] = (
                "small"
                if relative_size < 0.05
                else "medium"
                if relative_size < 0.25
                else "large"
            )
            det["attributes"]["relative_size"] = relative_size

            # Add position attributes
            rx, ry = center_x / width, center_y / height
            pos_v = "top" if ry < 0.33 else "middle" if ry < 0.66 else "bottom"
            pos_h = "left" if rx < 0.33 else "center" if rx < 0.66 else "right"
            det["attributes"]["position"] = f"{pos_v}-{pos_h}"
            det["relative_position"] = (rx, ry)

        return detections
