"""
YOLO video processing module
"""

import logging
from typing import Any, Dict, List, Tuple

import cv2
import torch
import numpy as np

from langvio.utils.detection import (
    optimize_for_memory,
    add_tracking_info,
    add_color_attributes,
    add_size_and_position_attributes,
    extract_detections,
)
from langvio.vision.utils import (
    SpatialRelationshipAnalyzer,
    TemporalObjectTracker,
)
from langvio.vision.yolo.yolo11_utils import (
    initialize_yolo11_tools,
    process_frame_with_yolo11,
)


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

        # Performance monitoring
        import time

        start_time = time.time()
        frame_count = 0
        processed_frames = 0

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

            # Get optimal processing resolution
            target_width, target_height = self._get_optimal_resolution(width, height)
            self.logger.info(f"Original resolution: {width}x{height}")
            self.logger.info(f"Processing resolution: {target_width}x{target_height}")

            # Initialize YOLO11 tools with processing resolution
            counter, speed_estimator = initialize_yolo11_tools(
                target_width, target_height
            )

            frame_idx = 0
            processed_frames = 0

            # Memory management
            memory_cleanup_interval = 50  # Clean memory every 50 frames

            with torch.no_grad():
                optimize_for_memory()

                while cap.isOpened():
                    frame_start_time = time.time()

                    # Log progress less frequently
                    if frame_idx % 10 == 0:
                        self.logger.info(f"Processing frame {frame_idx}/{total_frames}")

                    success, frame = cap.read()
                    if not success:
                        break

                    # Ultra-fast mode: skip frames for speed
                    should_process_frame = True
                    if analysis_config.get("ultra_fast_mode") and frame_idx % 2 == 1:
                        should_process_frame = False
                        # Still update YOLO11 counters on skipped frames
                        if counter or speed_estimator:
                            self._update_counters_only(frame, counter, speed_estimator)
                        frame_idx += 1
                        continue

                    # Resize frame for processing
                    resized_frame, scale_factor = self._resize_frame_if_needed(
                        frame, target_width, target_height
                    )

                    # Process frame for YOLO11 counting/speed
                    frame_result = self._process_frame_with_strategy(
                        resized_frame,  # Use resized frame
                        frame_idx,
                        target_width,  # Use target dimensions
                        target_height,
                        analysis_config,
                        counter,
                        speed_estimator,
                    )

                    # Scale bounding boxes back to original resolution for visualization
                    if frame_result["detections"]:
                        scaled_detections = self._scale_detections_to_original(
                            frame_result["detections"], scale_factor, width, height
                        )
                        frame_detections[str(frame_idx)] = scaled_detections
                    else:
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

                    # Memory cleanup
                    if frame_idx % memory_cleanup_interval == 0:
                        self._cleanup_memory()

                    frame_idx += 1
                    processed_frames += 1

            # Performance summary
            total_time = time.time() - start_time
            self.logger.info("=== PERFORMANCE SUMMARY ===")
            self.logger.info(f"Total frames processed: {processed_frames}")
            self.logger.info(f"Total processing time: {total_time:.2f} seconds")
            self.logger.info(f"Processing FPS: {processed_frames/total_time:.2f}")
            self.logger.info(
                f"Average frame time: {(total_time/processed_frames)*1000:.2f} ms"
            )
            self.logger.info(
                f"Resolution scaling: {width}x{height} → {target_width}x{target_height}"
            )

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
            # Final memory cleanup
            self._cleanup_memory()

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

    def _get_optimal_resolution(
        self, original_width: int, original_height: int
    ) -> Tuple[int, int]:
        """Get optimal resolution for processing while maintaining aspect ratio"""
        # More aggressive target resolutions for maximum speed
        target_resolutions = [
            (640, 360),  # 360p - fastest (4x speedup for 1080p)
            (854, 480),  # 480p - fast
            (1280, 720),  # 720p - balanced
        ]

        # Find the best resolution that's smaller than original
        for target_w, target_h in target_resolutions:
            if target_w <= original_width and target_h <= original_height:
                return target_w, target_h

        # If original is smaller than our targets, keep original
        return original_width, original_height

    def _resize_frame_if_needed(
        self, frame, target_width: int, target_height: int
    ) -> Tuple[np.ndarray, float]:
        """Resize frame if needed and return scale factor"""
        original_height, original_width = frame.shape[:2]

        if original_width == target_width and original_height == target_height:
            return frame, 1.0

        # Calculate scale factor
        scale_x = target_width / original_width
        scale_y = target_height / original_height
        scale_factor = min(scale_x, scale_y)  # Maintain aspect ratio

        # Calculate new dimensions
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)

        # Resize frame
        resized_frame = cv2.resize(
            frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR
        )

        return resized_frame, scale_factor

    def _determine_analysis_needs(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Determine what analysis is needed based on query to optimize processing"""
        task_type = query_params.get("task_type", "identification")

        # Color analysis frequency based on query
        needs_color = any(
            attr.get("attribute") == "color"
            for attr in query_params.get("attributes", [])
        )
        color_analysis_interval = (
            10 if needs_color else 30  # Much less frequent for speed
        )

        # Spatial analysis frequency
        needs_spatial = bool(query_params.get("spatial_relations", []))
        spatial_update_interval = 10 if needs_spatial else 20  # Much less frequent

        # Temporal analysis based on task
        needs_temporal = task_type in ["tracking", "activity"]
        temporal_update_interval = 5 if needs_temporal else 10

        # Ultra-fast mode: skip some frames for YOLO11 (trade accuracy for speed)
        ultra_fast_mode = query_params.get("ultra_fast", False)
        if ultra_fast_mode:
            self.logger.warning(
                "⚠️ Ultra-fast mode enabled - may reduce counting accuracy"
            )
            frame_processing_interval = 2  # Process every 2nd frame
        else:
            frame_processing_interval = 1  # Process every frame for accuracy

        return {
            "needs_color": needs_color,
            "needs_spatial": needs_spatial,
            "needs_temporal": needs_temporal,
            "color_analysis_interval": color_analysis_interval,
            "spatial_update_interval": spatial_update_interval,
            "temporal_update_interval": temporal_update_interval,
            "frame_processing_interval": frame_processing_interval,
            "ultra_fast_mode": ultra_fast_mode,
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
            # Run YOLO detection with ultra-fast settings
            optimized_settings = {
                "conf": self.config["confidence"],
                "iou": 0.4,  # Even lower IoU for speed
                "max_det": 30,  # Further reduced max detections
                "verbose": False,
                "save": False,
                "show": False,
                "half": torch.cuda.is_available(),  # Use FP16 if available
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "augment": False,  # Disable test-time augmentation
                "classes": None,  # Detect all classes for counting
                "agnostic_nms": True,  # Use agnostic NMS for speed
                "retina_masks": False,  # Disable retina masks for speed
                "max_det": 30,  # Ultra-low max detections
            }

            results = self.model(frame, **optimized_settings)
            detections = extract_detections(results)
            return detections
        except Exception as e:
            self.logger.warning(f"Error in basic detection: {e}")
            return []

    def _scale_detections_to_original(
        self,
        detections: List[Dict[str, Any]],
        scale_factor: float,
        original_width: int,
        original_height: int,
    ) -> List[Dict[str, Any]]:
        """Scale detections back to original resolution for visualization"""
        if scale_factor == 1.0:
            return detections

        scaled_detections = []
        for det in detections:
            scaled_det = det.copy()

            # Scale bounding box coordinates
            if "bbox" in scaled_det:
                x1, y1, x2, y2 = scaled_det["bbox"]
                scaled_det["bbox"] = [
                    int(x1 / scale_factor),
                    int(y1 / scale_factor),
                    int(x2 / scale_factor),
                    int(y2 / scale_factor),
                ]

            # Scale center coordinates if present
            if "center" in scaled_det:
                cx, cy = scaled_det["center"]
                scaled_det["center"] = (int(cx / scale_factor), int(cy / scale_factor))

            # Scale relative position if present
            if "relative_position" in scaled_det:
                rx, ry = scaled_det["relative_position"]
                scaled_det["relative_position"] = (
                    rx,
                    ry,
                )  # Relative positions don't change

            scaled_detections.append(scaled_det)

        return scaled_detections

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

    def _cleanup_memory(self):
        """Clean up memory to prevent OOM issues"""
        try:
            import gc
            import torch

            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Python garbage collection
            gc.collect()

        except Exception as e:
            # Silent cleanup failure
            pass
