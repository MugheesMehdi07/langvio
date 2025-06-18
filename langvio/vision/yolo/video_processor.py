"""
YOLO video processing module
"""

import logging
from typing import Any, Dict, List, Tuple

import cv2
import torch

from langvio.vision.utils import (SpatialRelationshipAnalyzer,
                                  TemporalObjectTracker, add_color_attributes,
                                  add_size_and_position_attributes,
                                  add_tracking_info, extract_detections,
                                  optimize_for_memory)
from langvio.vision.yolo.yolo11_utils import (initialize_yolo11_tools,
                                              process_frame_with_yolo11)


class YOLOVideoProcessor:
    """Handles video processing with YOLO models and YOLO11 integration"""

    def __init__(self, model, config, has_yolo11_solutions):
        self.model = model
        self.config = config
        self.has_yolo11_solutions = has_yolo11_solutions
        self.logger = logging.getLogger(__name__)

    def process(
            self, video_path: str, query_params: Dict[str, Any], sample_rate: int
    ) -> Dict[str, Any]:
        """Process video with enhanced analysis strategy"""
        self.logger.info(f"Processing video: {video_path}")

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
            cap, video_props = self._initialize_video_capture(video_path)
            width, height, fps, total_frames = video_props
            counter, speed_estimator = initialize_yolo11_tools(width, height)

            frame_idx = 0
            with torch.no_grad():
                optimize_for_memory()

                while cap.isOpened():
                    self.logger.info(f"Processing frame {frame_idx}/{total_frames}")

                    success, frame = cap.read()
                    if not success:
                        break

                    # Process every frame for counting/speed (required for YOLO11)
                    frame_result = self._process_frame_with_strategy(
                        frame,
                        frame_idx,
                        width,
                        height,
                        analysis_config,
                        counter,
                        speed_estimator,
                    )

                    # Store for visualization if frame was processed
                    if frame_result["detections"]:
                        frame_detections[str(frame_idx)] = frame_result["detections"]

                    # Update temporal tracking (every frame)
                    temporal_tracker.update_frame(
                        frame_idx, frame_result["detections"], fps
                    )

                    # Update spatial relationships (every N frames for performance)
                    if frame_idx % analysis_config["spatial_update_interval"] == 0:
                        spatial_analyzer.update_relationships(
                            frame_result["detections"]
                        )

                    # Update YOLO11 results
                    if frame_result.get("counter_result"):
                        final_counter_results = frame_result["counter_result"]
                    if frame_result.get("speed_result"):
                        final_speed_results = frame_result["speed_result"]

                    frame_idx += 1

            # Generate comprehensive results
            return self._create_enhanced_video_results(
                frame_detections,
                temporal_tracker,
                spatial_analyzer,
                final_counter_results,
                final_speed_results,
                video_props,
                query_params,
            )

        except Exception as e:
            self.logger.error(f"Error processing video: {e}")
            return {"error": str(e)}
        finally:
            if cap:
                cap.release()

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

    def _determine_analysis_needs(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Determine what analysis is needed based on query to optimize processing"""
        task_type = query_params.get("task_type", "identification")

        # Color analysis frequency based on query
        needs_color = any(
            attr.get("attribute") == "color"
            for attr in query_params.get("attributes", [])
        )
        color_analysis_interval = (
            3 if needs_color else 10
        )  # Every 3rd frame if color needed

        # Spatial analysis frequency
        needs_spatial = bool(query_params.get("spatial_relations", []))
        spatial_update_interval = 2 if needs_spatial else 5

        return {
            "needs_color": needs_color,
            "needs_spatial": needs_spatial,
            "needs_temporal": task_type in ["tracking", "activity"],
            "color_analysis_interval": color_analysis_interval,
            "spatial_update_interval": spatial_update_interval,
            "frame_processing_interval": (
                1 if task_type == "counting" else 2
            ),  # Every frame for counting
        }

    def _process_frame_with_strategy(
            self,
            frame,
            frame_idx: int,
            width: int,
            height: int,
            analysis_config: Dict[str, Any],
            counter: Any = None,
            speed_estimator: Any = None,
    ) -> Dict[str, Any]:
        """Process frame with optimized strategy based on analysis needs"""
        result = {"detections": [], "counter_result": None, "speed_result": None}

        try:
            # Always run basic detection
            detections = self._run_basic_detection(frame)

            # Add tracking IDs and basic spatial info (fast)
            detections = add_tracking_info(detections, frame_idx)

            # Color analysis on selected frames only
            if frame_idx % analysis_config["color_analysis_interval"] == 0:
                detections = add_color_attributes(
                    detections, frame, analysis_config["needs_color"]
                )

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

    def _run_basic_detection(self, frame) -> List[Dict[str, Any]]:
        """Run basic YOLO detection without attributes"""
        try:
            # Run YOLO detection
            optimized_settings = {
                "conf": self.config["confidence"],
                "iou": 0.5,
                "max_det": 100,  # Limit max detections
                "verbose": False,
                "save": False,
                "show": False,
                "half": torch.cuda.is_available(),  # Use FP16 if available
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "augment": False,  # Disable test-time augmentation
            }

            results = self.model(frame, **optimized_settings)
            detections = extract_detections(results)
            return detections
        except Exception as e:
            self.logger.warning(f"Error in basic detection: {e}")
            return []

    def _create_enhanced_video_results(
            self,
            frame_detections: Dict[str, List[Dict[str, Any]]],
            temporal_tracker: TemporalObjectTracker,
            spatial_analyzer: SpatialRelationshipAnalyzer,
            counter_results: Any,
            speed_results: Any,
            video_props: Tuple,
            query_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
    Create comprehensive video results with temporal, spatial,
    and YOLO11 analysis.
        """

        from langvio.vision.yolo.result_formatter import YOLOResultFormatter

        formatter = YOLOResultFormatter()
        return formatter.create_enhanced_video_results(
            frame_detections,
            temporal_tracker,
            spatial_analyzer,
            counter_results,
            speed_results,
            video_props,
            query_params,
        )
