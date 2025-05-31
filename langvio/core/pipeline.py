"""
Enhanced core pipeline for connecting LLMs with vision models
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional


from langvio.config import Config
from langvio.media.processor import MediaProcessor
from langvio.utils.file_utils import is_video_file
from langvio.utils.logging import setup_logging


class Pipeline:
    """Enhanced main pipeline for processing queries with LLMs and vision models"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize pipeline.

        Args:
            config_path: Path to configuration file
        """
        # Initialize configuration
        self.config = Config(config_path)

        # Set up logging
        setup_logging(self.config.get_logging_config())
        self.logger = logging.getLogger(__name__)

        # Initialize processors
        self.llm_processor = None
        self.vision_processor = None
        self.media_processor = MediaProcessor(self.config.get_media_config())

        self.logger.info("Enhanced Pipeline initialized")

    def load_config(self, config_path: str) -> None:
        """
        Load configuration from file.

        Args:
            config_path: Path to configuration file
        """
        self.config.load_config(config_path)
        self.logger.info(f"Loaded configuration from {config_path}")

        # Reinitialize processors with new config
        if self.llm_processor:
            self.set_llm_processor(self.llm_processor.name)

        if self.vision_processor:
            self.set_vision_processor(self.vision_processor.name)

        # Update media processor
        self.media_processor.update_config(self.config.get_media_config())

    def set_llm_processor(self, processor_name: str) -> None:
        """
        Set the LLM processor.

        Args:
            processor_name: Name of the processor to use
        """
        from langvio import registry

        self.logger.info(f"Setting LLM processor to {processor_name}")

        # Get processor config
        processor_config = self.config.get_llm_config(processor_name)

        # Check if the requested processor is available
        if processor_name not in registry.list_llm_processors():
            error_msg = (
                f"ERROR: LLM processor '{processor_name}' not found. "
                "You may need to install additional dependencies:\n"
                "- For OpenAI: pip install langvio[openai]\n"
                "- For Google Gemini: pip install langvio[google]\n"
                "- For all providers: pip install langvio[all-llm]"
            )
            self.logger.error(error_msg)
            print(error_msg, file=sys.stderr)
            sys.exit(1)

        # Create processor
        try:
            self.llm_processor = registry.get_llm_processor(
                processor_name, **processor_config
            )

            # Explicitly initialize the processor
            self.llm_processor.initialize()

        except Exception as e:
            error_msg = (
                f"ERROR: Failed to initialize LLM processor '{processor_name}': {e}"
            )
            self.logger.error(error_msg)
            print(error_msg, file=sys.stderr)
            sys.exit(1)

    def set_vision_processor(self, processor_name: str) -> None:
        """
        Set the vision processor.

        Args:
            processor_name: Name of the processor to use
        """
        from langvio import registry

        self.logger.info(f"Setting vision processor to {processor_name}")

        # Get processor config
        processor_config = self.config.get_vision_config(processor_name)

        # Check if the requested processor is available
        if processor_name not in registry.list_vision_processors():
            error_msg = f"ERROR: Vision processor '{processor_name}' not found."
            self.logger.error(error_msg)
            print(error_msg, file=sys.stderr)
            sys.exit(1)

        # Create processor
        try:
            self.vision_processor = registry.get_vision_processor(
                processor_name, **processor_config
            )
        except Exception as e:
            error_msg = (
                f"ERROR: Failed to initialize vision processor '{processor_name}': {e}"
            )
            self.logger.error(error_msg)
            print(error_msg, file=sys.stderr)
            sys.exit(1)

    def process(self, query: str, media_path: str) -> Dict[str, Any]:
        """
        Process a query on media with enhanced capabilities.
        Integrates YOLO detections with YOLO11 metrics, then passes to LLM.

        Args:
            query: Natural language query
            media_path: Path to media file (image or video)

        Returns:
            Dictionary with results

        Raises:
            ValueError: If processors are not set or media file doesn't exist
        """
        self.logger.info(f"Processing query: {query}")

        # Check if processors are set
        if not self.llm_processor:
            error_msg = "ERROR: LLM processor not set"
            self.logger.error(error_msg)
            print(error_msg, file=sys.stderr)
            sys.exit(1)

        if not self.vision_processor:
            error_msg = "ERROR: Vision processor not set"
            self.logger.error(error_msg)
            print(error_msg, file=sys.stderr)
            sys.exit(1)

        # Check if media file exists
        if not os.path.exists(media_path):
            error_msg = f"ERROR: Media file not found: {media_path}"
            self.logger.error(error_msg)
            print(error_msg, file=sys.stderr)
            sys.exit(1)

        # Check media type
        is_video = is_video_file(media_path)
        media_type = "video" if is_video else "image"

        # Process query with LLM to get structured parameters
        query_params = self.llm_processor.parse_query(query)
        self.logger.info(f"Parsed query params: {query_params}")

        # Run detection with vision processor
        if is_video:
            # For video processing, check if we need to adjust sample rate based on task
            sample_rate = 5  # Default
            if query_params.get("task_type") in ["tracking", "activity"]:
                # Use a more frequent sampling for tracking and activity detection
                sample_rate = 2

            # Get all detections with YOLO11 metrics integrated
            all_detections = self.vision_processor.process_video(
                media_path, query_params, sample_rate
            )
        else:
            # Get all detections with YOLO11 metrics integrated for image
            all_detections = self.vision_processor.process_image(
                media_path, query_params
            )

        # Generate explanation using all detected objects and metrics
        explanation = self.llm_processor.generate_explanation(query, all_detections,is_video)

        # Get highlighted objects from the LLM processor
        highlighted_objects = self.llm_processor.get_highlighted_objects()

        # Create visualization with highlighted objects
        output_path = self._create_visualization(
            media_path, all_detections, highlighted_objects, query_params, is_video
        )

        # Prepare result
        result = {
            "query": query,
            "media_path": media_path,
            "media_type": media_type,
            "output_path": output_path,
            "explanation": explanation,
            "detections": all_detections,
            "query_params": query_params,
            "highlighted_objects": highlighted_objects,
        }

        self.logger.info(f"Processed query successfully")
        return result

    def _get_visualization_config(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get visualization configuration based on query parameters.

        Args:
            query_params: Query parameters from LLM processor

        Returns:
            Visualization configuration parameters
        """
        # Get default visualization config
        viz_config = self.config.config["media"]["visualization"].copy()

        # Customize based on task type
        task_type = query_params.get("task_type", "identification")

        if task_type == "counting":
            # For counting tasks, use a different color
            viz_config["box_color"] = [255, 0, 0]  # Red for counting

        elif task_type == "verification":
            # For verification tasks, use a different color
            viz_config["box_color"] = [0, 0, 255]  # Blue for verification

        elif task_type in ["tracking", "activity"]:
            # For tracking/activity tasks, use a more visible color
            viz_config["box_color"] = [255, 165, 0]  # Orange for tracking/activity
            viz_config["line_thickness"] = 3  # Thicker lines

        # If specific attributes were requested, adjust the visualization
        if query_params.get("attributes"):
            # If looking for specific attributes, highlight them more
            viz_config["line_thickness"] += 1

        return viz_config

    def _create_visualization(
            self,
            media_path: str,
            all_detections: Dict[str, Any],
            highlighted_objects: List[Dict[str, Any]],
            query_params: Dict[str, Any],
            is_video: bool,
    ) -> str:
        """
        Create visualization adapted for new compressed video results structure.
        """
        # Generate output path
        output_path = self.media_processor.get_output_path(media_path)

        # Get visualization config
        visualization_config = self._get_visualization_config(query_params)

        if is_video:
            # For videos, create visualization with both detections and summary
            self._create_video_detection_visualization(
                media_path,
                output_path,
                all_detections,
                visualization_config
            )
        else:
            # For images, we still have frame-level data
            original_box_color = visualization_config["box_color"]
            highlight_color = [0, 0, 255]  # Red color (BGR) for highlighted objects
            image_objects = all_detections.get("objects", [])

            self.media_processor.visualize_image_with_highlights(
                media_path,
                output_path,
                image_objects,
                [obj["detection"] for obj in highlighted_objects],  # Extract detection objects
                original_box_color=original_box_color,
                highlight_color=highlight_color,
                text_color=visualization_config["text_color"],
                line_thickness=visualization_config["line_thickness"],
                show_attributes=visualization_config.get("show_attributes", True),
                show_confidence=visualization_config.get("show_confidence", True),
            )

        return output_path

    def _create_video_detection_visualization(
            self,
            video_path: str,
            output_path: str,
            compressed_results: Dict[str, Any],
            viz_config: Dict[str, Any]
    ) -> None:
        """
        Create video visualization with both object detection and summary overlay.
        Re-runs detection on video frames to show bounding boxes while using compressed results for overlay.
        """
        import cv2

        try:
            # Open input video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                import shutil
                shutil.copy2(video_path, output_path)
                return

            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # Extract summary info for overlay
            summary = compressed_results.get("summary", {})
            video_info = summary.get("video_info", {})
            advanced_stats = summary.get("advanced_stats", {})

            # Create overlay text
            overlay_lines = []
            if "primary_objects" in video_info:
                overlay_lines.append(f"Objects: {', '.join(video_info['primary_objects'])}")

            if "counting" in advanced_stats:
                counting = advanced_stats["counting"]
                if "in_count" in counting and "out_count" in counting:
                    overlay_lines.append(f"In: {counting['in_count']}, Out: {counting['out_count']}")

            if "speed" in advanced_stats:
                speed = advanced_stats["speed"]
                if "avg_speed" in speed:
                    overlay_lines.append(f"Avg Speed: {speed['avg_speed']:.1f} km/h")

            # Initialize vision model for detection (reuse from vision processor)
            vision_model = self.vision_processor.model
            confidence = self.vision_processor.config.get("confidence", 0.5)

            frame_idx = 0
            sample_rate = 5  # Process every 5th frame for performance

            # Process frames with detection and overlay
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Run detection on sampled frames
                if frame_idx % sample_rate == 0 and vision_model:
                    try:
                        # Run YOLO detection on current frame
                        results = vision_model(frame, conf=confidence)

                        # Draw bounding boxes
                        for result in results:
                            boxes = result.boxes
                            for box in boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                conf = float(box.conf[0])
                                cls_id = int(box.cls[0])
                                label = result.names[cls_id]

                                # Draw bounding box
                                color = viz_config.get("box_color", [0, 255, 0])
                                thickness = viz_config.get("line_thickness", 2)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                                # Draw label
                                label_text = f"{label} {conf:.2f}"
                                text_color = viz_config.get("text_color", [255, 255, 255])
                                cv2.putText(
                                    frame, label_text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1
                                )
                    except Exception as e:
                        self.logger.warning(f"Detection failed on frame {frame_idx}: {e}")

                # Add summary overlay (top-left corner)
                y_offset = 30
                for line in overlay_lines:
                    # Add background for better text visibility
                    text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (5, y_offset - 25), (15 + text_size[0], y_offset + 5), (0, 0, 0), -1)

                    cv2.putText(
                        frame, line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        viz_config.get("text_color", [255, 255, 255]), 2
                    )
                    y_offset += 35

                writer.write(frame)
                frame_idx += 1

            cap.release()
            writer.release()
            self.logger.info(f"Created video detection visualization: {output_path}")

        except Exception as e:
            self.logger.error(f"Error creating video visualization: {e}")
            # Fallback: copy original video
            import shutil
            shutil.copy2(video_path, output_path)