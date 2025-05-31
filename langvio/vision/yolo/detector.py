"""
Clean YOLO-based vision processor focused only on processing logic
"""

import logging
from ultralytics import YOLO, YOLOE

from langvio.prompts.constants import DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_VIDEO_SAMPLE_RATE
from langvio.vision.base import BaseVisionProcessor
from langvio.vision.utils import *
from langvio.vision.yolo.yolo11_utils import (
    check_yolo11_solutions_available,
    process_frame_with_yolo11,
    initialize_yolo11_tools, parse_solutions_results
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



        # Skip expensive operations for videos unless specifically needed
        if is_video_frame:
            # 2. Determine what attributes to compute based on query
            needs_color = any(attr.get("attribute") == "color" for attr in query_params.get("attributes", []))
            needs_spatial = bool(query_params.get("spatial_relations", []))
            needs_size = any(attr.get("attribute") == "size" for attr in query_params.get("attributes", []))
        else:
            needs_color = needs_spatial =  needs_size = True

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
        """Create comprehensive compressed video results with integrated YOLO11 metrics."""
        width, height, fps, total_frames = video_props
        duration = total_frames / fps

        # Use existing utility functions
        temporal_analysis = create_temporal_analysis(time_windows, fps)

        # Parse YOLO11 metrics using existing function

        advanced_stats = parse_solutions_results(counter_results,speed_results)

        # Spatial distribution insights - object sizes and positions
        spatial_insights = {}
        for obj_key, obj_data in object_tracker.items():
            obj_type = obj_data["type"]
            if obj_type not in spatial_insights:
                spatial_insights[obj_type] = {
                    "count": 0,
                    "avg_size": 0,
                    "positions": [],
                    "size_distribution": {"small": 0, "medium": 0, "large": 0}
                }

            spatial_insights[obj_type]["count"] += 1

            # Get last known position for spatial context
            if obj_data["positions"]:
                last_pos = obj_data["positions"][-1]
                # Convert to relative position (0-1)
                rel_x = last_pos[0] / width if width > 0 else 0
                rel_y = last_pos[1] / height if height > 0 else 0

                # Categorize position zones
                zone_x = "left" if rel_x < 0.33 else "center" if rel_x < 0.67 else "right"
                zone_y = "top" if rel_y < 0.33 else "middle" if rel_y < 0.67 else "bottom"
                zone = f"{zone_y}_{zone_x}"

                spatial_insights[obj_type]["positions"].append(zone)

                # Estimate object size based on bounding box area (if available in positions data)
                # This is a simplified approach - could be enhanced with actual bbox data
                estimated_size = "medium"  # Default
                spatial_insights[obj_type]["size_distribution"][estimated_size] += 1

        # Process position distributions
        for obj_type in spatial_insights:
            positions = spatial_insights[obj_type]["positions"]
            if positions:
                # Get most common positions
                from collections import Counter
                position_counts = Counter(positions)
                spatial_insights[obj_type]["common_positions"] = dict(position_counts.most_common(3))
            del spatial_insights[obj_type]["positions"]  # Remove raw positions list

        # Get most common objects for context
        most_common = sorted(spatial_insights.items(), key=lambda x: x[1]["count"], reverse=True)[:3]
        primary_objects = [obj_type for obj_type, _ in most_common]

        # Activity level based on objects per minute
        objects_per_minute = len(object_tracker) / (duration / 60) if duration > 0 else 0
        if objects_per_minute < 5:
            activity_level = "low"
        elif objects_per_minute < 20:
            activity_level = "moderate"
        else:
            activity_level = "high"

        return {
            "summary": {
                "video_info": {
                    "duration_seconds": round(duration, 1),
                    "resolution": f"{width}x{height}",
                    "fps": round(fps, 1),
                    "total_objects_tracked": len(object_tracker),
                    "time_windows_analyzed": len(time_windows),
                    "activity_level": activity_level,
                    "primary_objects": primary_objects
                },

                "spatial_distribution": spatial_insights,
                "temporal_analysis": temporal_analysis,
                "advanced_stats": advanced_stats
            }
        }

