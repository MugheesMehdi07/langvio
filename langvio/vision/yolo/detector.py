"""
Clean YOLO-based vision processor focused only on processing logic
"""

import logging
from typing import Any, Dict, List, Tuple

import cv2
import torch
from ultralytics import YOLO, YOLOE

from langvio.prompts.constants import DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_VIDEO_SAMPLE_RATE
from langvio.vision.base import BaseVisionProcessor
from langvio.vision.utils import (
    optimize_for_memory,
    extract_detections,
    add_unified_attributes,
    compress_detections_for_output,
    update_object_tracker,
    update_time_window,
    analyze_movement_patterns,
    create_temporal_analysis,
    identify_object_clusters
)
from langvio.vision.yolo.yolo11_utils import (
    check_yolo11_solutions_available,
    process_frame_with_yolo11,
    initialize_yolo11_tools
)


class YOLOProcessor(BaseVisionProcessor):
    """Clean vision processor using YOLO models - focused on processing logic only"""

    def __init__(self, name: str, model_path: str, confidence: float = DEFAULT_CONFIDENCE_THRESHOLD, **kwargs):
        """Initialize YOLO processor."""
        config = {
            "model_path": model_path,
            "confidence": confidence,
            **kwargs,
        }
        super().__init__(name, config)
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.model_type = kwargs.get("model_type", "yolo")

        # Check YOLO11 Solutions availability
        self.has_yolo11_solutions = check_yolo11_solutions_available()
        if self.has_yolo11_solutions:
            self.logger.info("YOLO11 Solutions is available for metrics")
        else:
            self.logger.info("YOLO11 Solutions not available - using basic detection only")

    def initialize(self) -> bool:
        """Initialize the YOLO model."""
        try:
            self.logger.info(f"Loading {self.model_type} model: {self.config['model_path']}")

            if self.model_type == "yoloe":
                self.model = YOLOE(self.config['model_path'])
            else:
                self.model = YOLO(self.config['model_path'])

            return True
        except Exception as e:
            self.logger.error(f"Error loading YOLO model: {e}")
            return False

    def initialize_video_capture(self, video_path: str) -> Tuple[cv2.VideoCapture, Tuple[int, int, float, int]]:
        """Initialize video capture and extract video properties."""
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

    # =============================================================================
    # CORE PROCESSING METHODS
    # =============================================================================

    def _run_detection_with_attributes(
            self,
            input_data: Any,  # Can be image_path (str) or frame (np.ndarray)
            width: int,
            height: int,
            query_params: Dict[str, Any],
            is_video_frame: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Core detection method for both images and video frames.

        Args:
            input_data: Image path (str) for images, frame array for videos
            width: Image/frame width
            height: Image/frame height
            query_params: Query parameters to determine what attributes to compute
            is_video_frame: Whether this is a video frame

        Returns:
            List of enhanced detections
        """
        # 1. Run YOLO detection
        results = self.model(input_data, conf=self.config["confidence"])
        detections = extract_detections(results)

        if not detections:
            return []

        # 2. Determine what attributes to compute based on query
        needs_color = any(attr.get("attribute") == "color" for attr in query_params.get("attributes", []))
        needs_spatial = bool(query_params.get("spatial_relations", []))
        needs_size = any(attr.get("attribute") == "size" for attr in query_params.get("attributes", []))

        # Skip expensive operations for videos unless specifically needed
        if is_video_frame:
            needs_color = False  # Too expensive for video
            needs_spatial = False  # Too expensive for video

        # 3. Add attributes using utility function
        detections = add_unified_attributes(
            detections, width, height, input_data,
            needs_color, needs_spatial, needs_size, is_video_frame
        )

        return detections

    def _process_single_frame(
            self,
            frame,
            frame_idx: int,
            width: int,
            height: int,
            query_params: Dict[str, Any],
            counter: Any = None,
            speed_estimator: Any = None
    ) -> Dict[str, Any]:
        """
        Process a single video frame.

        Args:
            frame: Video frame
            frame_idx: Frame index
            width: Frame width
            height: Frame height
            query_params: Query parameters
            counter: YOLO11 counter (optional)
            speed_estimator: YOLO11 speed estimator (optional)

        Returns:
            Frame processing results
        """
        result = {
            "detections": [],
            "counter_result": None,
            "speed_result": None
        }

        try:
            # 1. Run detection with attributes
            detections = self._run_detection_with_attributes(
                frame, width, height, query_params, is_video_frame=True
            )
            result["detections"] = detections

            # 2. Run YOLO11 metrics if available
            if self.has_yolo11_solutions and (counter or speed_estimator):
                try:
                    counter_result, speed_result = process_frame_with_yolo11(
                        frame, counter, speed_estimator
                    )
                    result["counter_result"] = counter_result
                    result["speed_result"] = speed_result
                except Exception as e:
                    self.logger.warning(f"YOLO11 processing error on frame {frame_idx}: {e}")

        except Exception as e:
            self.logger.warning(f"Error processing frame {frame_idx}: {e}")

        return result

    # =============================================================================
    # IMAGE PROCESSING
    # =============================================================================

    def process_image(self, image_path: str, query_params: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Optimized image processing with compressed output."""
        self.logger.info(f"Processing image: {image_path}")

        if not self.model:
            self.initialize()

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
                    image_path, width, height, query_params, is_video_frame=False
                )

                # Create compressed results
                compressed_objects = compress_detections_for_output(detections, is_video=False)
                summary = self._create_image_summary(detections, width, height, query_params)

                return {
                    "objects": compressed_objects,
                    "summary": summary
                }

            except Exception as e:
                self.logger.error(f"Error processing image: {e}")
                return {"objects": [], "error": str(e)}

    def _create_image_summary(
            self,
            detections: List[Dict[str, Any]],
            width: int,
            height: int,
            query_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create optimized image summary."""
        if not detections:
            return {
                "image_info": {"resolution": f"{width}x{height}", "total_objects": 0},
                "analysis": "No objects detected"
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
                notable_patterns.append(f"{dominant_type[0]} is dominant ({dominant_type[1]} instances)")

        return {
            "image_info": {
                "resolution": f"{width}x{height}",
                "total_objects": len(detections),
                "unique_types": len(types)
            },
            "object_distribution": {
                "by_type": types,
                "by_position": positions if positions else None,
                "by_size": sizes if sizes else None,
                "by_color": colors if colors else None
            },
            "notable_patterns": notable_patterns,
            "query_context": query_params.get("task_type", "identification")
        }

    # =============================================================================
    # VIDEO PROCESSING - OPTIMIZED FOR SPEED AND COMPRESSED OUTPUT
    # =============================================================================

    def process_video(
            self,
            video_path: str,
            query_params: Dict[str, Any],
            sample_rate: int = DEFAULT_VIDEO_SAMPLE_RATE,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Optimized video processing - fast with compressed output.
        """
        self.logger.info(f"Processing video: {video_path}")

        if not self.model:
            self.initialize()

        # Lightweight containers - no per-frame storage
        object_tracker = {}  # Track objects across video
        time_windows = {}    # Aggregate into time windows
        counter_results = None
        speed_results = None
        cap = None

        try:
            # Initialize video
            cap, video_props = self.initialize_video_capture(video_path)
            width, height, fps, total_frames = video_props
            counter, speed_estimator = initialize_yolo11_tools(width, height)

            frame_idx = 0
            with torch.no_grad():
                optimize_for_memory()

                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        break

                    # Process every Nth frame for speed
                    if frame_idx % sample_rate == 0:

                        # Process single frame
                        result = self._process_single_frame(
                            frame, frame_idx, width, height, query_params, counter, speed_estimator
                        )

                        # Update trackers (no per-frame storage)
                        detections = result.get("detections", [])
                        if detections:
                            update_object_tracker(object_tracker, detections, frame_idx, fps)

                            # Aggregate into time windows
                            time_window = frame_idx // (fps * 2)  # 2-second windows
                            update_time_window(time_windows, detections, time_window)

                        # Update YOLO11 results (only keep latest)
                        if result.get("counter_result"):
                            counter_results = result["counter_result"]
                        if result.get("speed_result"):
                            speed_results = result["speed_result"]

                    frame_idx += 1

            # Create compressed final results
            return self._create_compressed_video_results(
                object_tracker, time_windows, counter_results, speed_results, video_props
            )

        except Exception as e:
            self.logger.error(f"Error processing video: {e}")
            return {"error": str(e)}
        finally:
            if cap:
                cap.release()

    def _create_compressed_video_results(
            self,
            object_tracker: Dict[str, Dict],
            time_windows: Dict[int, Dict],
            counter_results: Any,
            speed_results: Any,
            video_props: Tuple
    ) -> Dict[str, Any]:
        """Create compressed video results - no verbose per-frame data."""
        width, height, fps, total_frames = video_props
        duration = total_frames / fps

        # Analyze object patterns using utility functions
        movement_patterns = analyze_movement_patterns(object_tracker)
        temporal_analysis = create_temporal_analysis(time_windows, fps)

        # Analyze object lifecycles
        object_analysis = {}
        for obj_key, obj_data in object_tracker.items():
            obj_type = obj_data["type"]
            if obj_type not in object_analysis:
                object_analysis[obj_type] = {
                    "count": 0,
                    "avg_duration": 0,
                    "avg_confidence": 0
                }

            object_analysis[obj_type]["count"] += 1
            obj_duration = obj_data["last_seen"] - obj_data["first_seen"]
            object_analysis[obj_type]["avg_duration"] += obj_duration
            object_analysis[obj_type]["avg_confidence"] += (
                obj_data["total_confidence"] / obj_data["appearances"]
            )

        # Calculate averages
        for obj_type in object_analysis:
            count = object_analysis[obj_type]["count"]
            if count > 0:
                object_analysis[obj_type]["avg_duration"] = round(
                    object_analysis[obj_type]["avg_duration"] / count, 1
                )
                object_analysis[obj_type]["avg_confidence"] = round(
                    object_analysis[obj_type]["avg_confidence"] / count, 2
                )

        # Parse YOLO11 metrics
        metrics = {}
        if counter_results:
            metrics["counting"] = parse_solution_results(counter_results)
        if speed_results:
            metrics["speed"] = parse_solution_results(speed_results)

        # COMPRESSED RESULT - no per-frame data!
        return {
            "summary": {
                "video_info": {
                    "duration_seconds": round(duration, 1),
                    "resolution": f"{width}x{height}",
                    "unique_objects_tracked": len(object_tracker),
                    "frames_analyzed": len([k for k in time_windows.keys()])
                },
                "object_analysis": object_analysis,
                "movement_patterns": movement_patterns,
                "temporal_analysis": temporal_analysis,
                "yolo11_metrics": metrics if metrics else None
            },
            "metrics": metrics
        }

    # =============================================================================
    # LEGACY SUPPORT METHOD (for backward compatibility)
    # =============================================================================

    def _create_simple_summary(self, detections) -> Dict[str, Any]:
        """Simplified version for backward compatibility."""
        if isinstance(detections, list):
            types = [det["label"] for det in detections if "label" in det]
            return {
                "object_types": list(set(types)),
                "total_objects": len(detections)
            }
        elif isinstance(detections, dict) and "summary" in detections:
            return detections["summary"]

        return {"object_types": [], "total_objects": 0}