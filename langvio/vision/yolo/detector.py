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

    def _run_basic_detection(self, input_data: Any) -> List[Dict[str, Any]]:
        """
        Run basic YOLO detection without attributes.

        Args:
            input_data: Image path (str) or frame (np.ndarray)

        Returns:
            List of basic detection dictionaries
        """
        try:
            # Run YOLO detection
            results = self.model(input_data, conf=self.config["confidence"])

            # Extract basic detections using existing utility
            detections = extract_detections(results)

            return detections
        except Exception as e:
            self.logger.warning(f"Error in basic detection: {e}")
            return []

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
    ) -> Dict[str, Any]:
        """
        Enhanced video processing with optimized frame analysis strategy.
        - Every frame: YOLO11 counting/speed + basic detection
        - Every N frames: Color/attribute analysis
        - Continuous: Temporal relationship tracking
        """
        self.logger.info(f"Processing video: {video_path}")

        if not self.model:
            self.initialize()

        # Analysis configuration based on query
        analysis_config = self._determine_analysis_needs(query_params)

        # Storage for different types of data
        frame_detections = {}  # For visualization - store every processed frame
        temporal_tracker = TemporalObjectTracker()
        spatial_analyzer = SpatialRelationshipAnalyzer()

        # YOLO11 results storage
        final_counter_results = None
        final_speed_results = None

        cap = None

        try:
            # Initialize video and YOLO11 tools
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

                    # Process every frame for counting/speed (required for YOLO11)
                    frame_result = self._process_frame_with_strategy(
                        frame, frame_idx, width, height,
                        analysis_config, counter, speed_estimator
                    )

                    # Store for visualization if frame was processed
                    if frame_result["detections"]:
                        frame_detections[str(frame_idx)] = frame_result["detections"]

                    # Update temporal tracking (every frame)
                    temporal_tracker.update_frame(frame_idx, frame_result["detections"], fps)

                    # Update spatial relationships (every N frames for performance)
                    if frame_idx % analysis_config["spatial_update_interval"] == 0:
                        spatial_analyzer.update_relationships(frame_result["detections"])

                    # Update YOLO11 results
                    if frame_result.get("counter_result"):
                        final_counter_results = frame_result["counter_result"]
                    if frame_result.get("speed_result"):
                        final_speed_results = frame_result["speed_result"]

                    frame_idx += 1

            # Generate comprehensive results
            return self._create_enhanced_video_results(
                frame_detections, temporal_tracker, spatial_analyzer,
                final_counter_results, final_speed_results, video_props, query_params
            )

        except Exception as e:
            self.logger.error(f"Error processing video: {e}")
            return {"error": str(e)}
        finally:
            if cap:
                cap.release()

    def _determine_analysis_needs(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Determine what analysis is needed based on query to optimize processing."""
        task_type = query_params.get("task_type", "identification")

        # Color analysis frequency based on query
        needs_color = any(attr.get("attribute") == "color" for attr in query_params.get("attributes", []))
        color_analysis_interval = 3 if needs_color else 10  # Every 3rd frame if color needed

        # Spatial analysis frequency
        needs_spatial = bool(query_params.get("spatial_relations", []))
        spatial_update_interval = 2 if needs_spatial else 5

        return {
            "needs_color": needs_color,
            "needs_spatial": needs_spatial,
            "needs_temporal": task_type in ["tracking", "activity"],
            "color_analysis_interval": color_analysis_interval,
            "spatial_update_interval": spatial_update_interval,
            "frame_processing_interval": 1 if task_type == "counting" else 2  # Every frame for counting
        }

    def _process_frame_with_strategy(
            self,
            frame: np.ndarray,
            frame_idx: int,
            width: int,
            height: int,
            analysis_config: Dict[str, Any],
            counter: Any = None,
            speed_estimator: Any = None
    ) -> Dict[str, Any]:
        """
        Process frame with optimized strategy based on analysis needs.
        """
        result = {
            "detections": [],
            "counter_result": None,
            "speed_result": None
        }

        try:
            # Always run basic detection
            detections = self._run_basic_detection(frame)

            # Add tracking IDs and basic spatial info (fast)
            detections = add_tracking_info(detections, frame_idx)

            # Color analysis on selected frames only
            if frame_idx % analysis_config["color_analysis_interval"] == 0:
                detections = add_color_attributes(detections, frame, analysis_config["needs_color"])

            # Size and position attributes (always, as they're fast)
            detections = add_size_and_position_attributes(detections, width, height)

            result["detections"] = detections

            # YOLO11 processing (every frame for accuracy)
            if counter or speed_estimator:
                counter_result, speed_result = process_frame_with_yolo11(
                    frame, counter, speed_estimator
                )
                result["counter_result"] = counter_result
                result["speed_result"] = speed_result

        except Exception as e:
            self.logger.warning(f"Error processing frame {frame_idx}: {e}")

        return result

    def _create_enhanced_video_results(
            self,
            frame_detections: Dict[str, List[Dict[str, Any]]],
            temporal_tracker: Any,  # TemporalObjectTracker
            spatial_analyzer: Any,  # SpatialRelationshipAnalyzer
            counter_results: Any,
            speed_results: Any,
            video_props: Tuple,
            query_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create comprehensive video results with temporal, spatial, and YOLO11 analysis.
        Returns both compressed results (for LLM) and detailed results (for visualization).
        """
        width, height, fps, total_frames = video_props
        duration = total_frames / fps

        # Get temporal analysis
        movement_patterns = temporal_tracker.get_movement_patterns()
        temporal_relationships = temporal_tracker.get_temporal_relationships()

        # Get spatial analysis
        spatial_summary = spatial_analyzer.get_relationship_summary()

        # Parse YOLO11 metrics using enhanced parser
        yolo11_metrics = self._parse_enhanced_yolo11_results(counter_results, speed_results)

        # Create object summary with temporal and spatial context
        object_analysis = self._create_comprehensive_object_analysis(
            temporal_tracker, movement_patterns, spatial_summary
        )

        # Determine primary insights based on query type
        primary_insights = self._extract_primary_insights(
            query_params, yolo11_metrics, movement_patterns, spatial_summary
        )

        return {
            # For LLM processing - compressed and focused
            "summary": {
                "video_info": {
                    "duration_seconds": round(duration, 1),
                    "resolution": f"{width}x{height}",
                    "fps": round(fps, 1),
                    "activity_level": self._assess_activity_level(movement_patterns, duration),
                    "primary_objects": object_analysis.get("most_common_types", [])[:3]
                },

                # YOLO11 metrics (counting and speed) - PRIMARY SOURCE OF TRUTH
                "counting_analysis": yolo11_metrics.get("counting", {}),
                "speed_analysis": yolo11_metrics.get("speed", {}),

                # Temporal analysis
                "temporal_relationships": {
                    "movement_patterns": {
                        "stationary_count": len(movement_patterns.get("stationary_objects", [])),
                        "moving_count": len(movement_patterns.get("moving_objects", [])),
                        "fast_moving_count": len(movement_patterns.get("fast_moving_objects", [])),
                        "primary_directions": dict(list(movement_patterns.get("directional_movements", {}).items())[:3])
                    },
                    "co_occurrence_events": len(temporal_relationships),
                    "interaction_summary": temporal_relationships[:5]  # Top 5 interactions
                },

                # Spatial analysis
                "spatial_relationships": {
                    "common_relations": spatial_summary.get("most_common_relations", {}),
                    "frequent_pairs": spatial_summary.get("frequent_object_pairs", {}),
                    "spatial_patterns": spatial_summary.get("spatial_patterns", {})
                },

                # Object analysis with attributes
                "object_analysis": object_analysis,

                # Query-specific insights
                "primary_insights": primary_insights
            },

            # For visualization - detailed frame data
            "frame_detections": frame_detections,

            # Metadata
            "processing_info": {
                "frames_analyzed": len(frame_detections),
                "total_frames": total_frames,
                "analysis_type": query_params.get("task_type", "identification"),
                "yolo11_enabled": bool(counter_results or speed_results)
            }
        }

    def _parse_enhanced_yolo11_results(self, counter_results: Any, speed_results: Any) -> Dict[str, Any]:
        """Enhanced parsing of YOLO11 results with better structure."""
        metrics = {}

        # Enhanced counting analysis
        if counter_results and hasattr(counter_results, 'in_count'):
            counting_data = {
                "objects_entered": counter_results.in_count,
                "objects_exited": counter_results.out_count,
                "net_flow": counter_results.in_count - counter_results.out_count,
                "total_crossings": counter_results.in_count + counter_results.out_count,
                "flow_direction": "inward" if counter_results.in_count > counter_results.out_count else "outward"
            }

            # Class-wise analysis
            if hasattr(counter_results, 'classwise_count'):
                class_analysis = {}
                for obj_type, directions in counter_results.classwise_count.items():
                    if directions["IN"] > 0 or directions["OUT"] > 0:
                        class_analysis[obj_type] = {
                            "entered": directions["IN"],
                            "exited": directions["OUT"],
                            "net_flow": directions["IN"] - directions["OUT"],
                            "dominance": "entering" if directions["IN"] > directions["OUT"] else "exiting"
                        }

                counting_data["by_object_type"] = class_analysis
                counting_data["most_active_type"] = max(class_analysis.items(),
                                                        key=lambda x: x[1]["entered"] + x[1]["exited"])[
                    0] if class_analysis else None

            metrics["counting"] = counting_data

        # Enhanced speed analysis
        if speed_results and hasattr(speed_results, 'total_tracks'):
            speed_data = {
                "objects_with_speed": speed_results.total_tracks,
                "speed_available": speed_results.total_tracks > 0
            }

            if hasattr(speed_results, 'avg_speed') and speed_results.avg_speed:
                speed_data["average_speed_kmh"] = round(speed_results.avg_speed, 1)
                speed_data["speed_category"] = self._categorize_speed(speed_results.avg_speed)

            # Class-wise speed analysis
            if hasattr(speed_results, 'class_speeds'):
                class_speeds = {}
                for obj_type, speeds in speed_results.class_speeds.items():
                    if speeds:
                        avg_speed = sum(speeds) / len(speeds)
                        class_speeds[obj_type] = {
                            "average_speed": round(avg_speed, 1),
                            "sample_count": len(speeds),
                            "speed_range": {
                                "min": round(min(speeds), 1),
                                "max": round(max(speeds), 1)
                            } if len(speeds) > 1 else None,
                            "speed_category": self._categorize_speed(avg_speed)
                        }

                speed_data["by_object_type"] = class_speeds

                # Find fastest object type
                if class_speeds:
                    speed_data["fastest_type"] = max(class_speeds.items(),
                                                     key=lambda x: x[1]["average_speed"])[0]

            metrics["speed"] = speed_data

        return metrics

    def _create_comprehensive_object_analysis(
            self,
            temporal_tracker: Any,
            movement_patterns: Dict[str, Any],
            spatial_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create comprehensive object analysis from temporal and spatial data."""

        # Extract object types and their characteristics
        object_characteristics = defaultdict(lambda: {
            "total_instances": 0,
            "movement_behavior": "unknown",
            "common_attributes": defaultdict(int),
            "spatial_preferences": defaultdict(int)
        })

        # Analyze from temporal tracker
        for obj_key, history in temporal_tracker.object_histories.items():
            obj_type = obj_key.split('_')[0]  # Extract object type
            characteristics = object_characteristics[obj_type]
            characteristics["total_instances"] += 1

            # Analyze movement behavior
            if len(history["positions"]) >= 3:
                movement_distance = temporal_tracker._calculate_total_movement(list(history["positions"]))
                if movement_distance < 50:
                    characteristics["movement_behavior"] = "stationary"
                elif movement_distance > 200:
                    characteristics["movement_behavior"] = "highly_mobile"
                else:
                    characteristics["movement_behavior"] = "moderately_mobile"

            # Collect attributes from latest frames
            for attrs in list(history["attributes"])[-5:]:  # Last 5 frames
                for attr_name, attr_value in attrs.items():
                    if attr_value and attr_value != "unknown":
                        characteristics["common_attributes"][f"{attr_name}:{attr_value}"] += 1

        # Convert to regular dict and find most common attributes
        final_analysis = {}
        most_common_types = sorted(object_characteristics.items(),
                                   key=lambda x: x[1]["total_instances"], reverse=True)

        for obj_type, chars in most_common_types:
            # Get most common attributes
            top_attributes = dict(sorted(chars["common_attributes"].items(),
                                         key=lambda x: x[1], reverse=True)[:3])

            final_analysis[obj_type] = {
                "total_instances": chars["total_instances"],
                "movement_behavior": chars["movement_behavior"],
                "common_attributes": top_attributes
            }

        return {
            "object_characteristics": final_analysis,
            "most_common_types": [obj_type for obj_type, _ in most_common_types[:5]],
            "total_unique_objects": len(object_characteristics)
        }

    def _extract_primary_insights(
            self,
            query_params: Dict[str, Any],
            yolo11_metrics: Dict[str, Any],
            movement_patterns: Dict[str, Any],
            spatial_summary: Dict[str, Any]
    ) -> List[str]:
        """Extract key insights based on query type and analysis results."""
        insights = []
        task_type = query_params.get("task_type", "identification")

        # Counting-specific insights (PRIMARY)
        if task_type == "counting" and "counting" in yolo11_metrics:
            counting = yolo11_metrics["counting"]
            insights.append(f"YOLO11 counted {counting.get('total_crossings', 0)} total object crossings")

            if counting.get("net_flow", 0) != 0:
                flow_type = "net inward" if counting["net_flow"] > 0 else "net outward"
                insights.append(f"Overall flow: {abs(counting['net_flow'])} objects {flow_type}")

            if counting.get("most_active_type"):
                insights.append(f"Most active object type: {counting['most_active_type']}")

        # Speed-specific insights
        if "speed" in yolo11_metrics and yolo11_metrics["speed"].get("speed_available"):
            speed = yolo11_metrics["speed"]
            if speed.get("average_speed_kmh"):
                insights.append(
                    f"Average speed: {speed['average_speed_kmh']} km/h ({speed.get('speed_category', 'unknown')} pace)")

            if speed.get("fastest_type"):
                insights.append(f"Fastest object type: {speed['fastest_type']}")

        # Movement pattern insights
        if movement_patterns:
            stationary_count = len(movement_patterns.get("stationary_objects", []))
            moving_count = len(movement_patterns.get("moving_objects", []))

            if stationary_count > moving_count:
                insights.append(f"Scene is mostly static with {stationary_count} stationary objects")
            elif moving_count > 0:
                insights.append(f"Active scene with {moving_count} moving objects")

        # Spatial relationship insights
        if spatial_summary.get("most_common_relations"):
            top_relation = list(spatial_summary["most_common_relations"].keys())[0]
            insights.append(f"Most common spatial relationship: {top_relation}")

        return insights[:4]  # Limit to top 4 insights

    def _categorize_speed(self, speed_kmh: float) -> str:
        """Categorize speed into human-readable categories."""
        if speed_kmh < 5:
            return "very_slow"
        elif speed_kmh < 15:
            return "slow"
        elif speed_kmh < 40:
            return "moderate"
        elif speed_kmh < 80:
            return "fast"
        else:
            return "very_fast"

    def _assess_activity_level(self, movement_patterns: Dict[str, Any], duration: float) -> str:
        """Assess overall activity level of the video."""
        total_moving = (len(movement_patterns.get("moving_objects", [])) +
                        len(movement_patterns.get("fast_moving_objects", [])))
        total_stationary = len(movement_patterns.get("stationary_objects", []))

        if total_moving == 0:
            return "static"
        elif total_moving < total_stationary:
            return "low_activity"
        elif total_moving > total_stationary * 2:
            return "high_activity"
        else:
            return "moderate_activity"