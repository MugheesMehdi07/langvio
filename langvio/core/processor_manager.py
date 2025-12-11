"""
Manages LLM and vision processor lifecycle and coordination
"""

import logging
import sys
from typing import Any, Dict

from langvio.utils.file_utils import is_video_file


class ProcessorManager:
    """Manages LLM and vision processor lifecycle and coordination"""

    def __init__(self, config):
        self.config = config
        self.llm_processor = None
        self.vision_processor = None
        self.logger = logging.getLogger(__name__)

    def set_llm_processor(self, processor_name: str) -> None:
        """Set the LLM processor"""
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
        """Set the vision processor"""
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

    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse a natural language query into structured parameters.

        Args:
            query: Natural language query string

        Returns:
            Dictionary containing parsed query parameters
            (target_objects, task_type, etc.)

        Raises:
            ValueError: If LLM processor is not set
        """
        if not self.llm_processor:
            error_msg = "LLM processor not set. Call set_llm_processor() first."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.debug(f"Parsing query: {query[:100]}...")
        parsed_params = self.llm_processor.parse_query(query)
        self.logger.debug(
            f"Parsed query parameters: {parsed_params.get('task_type', 'unknown')} task"
        )
        return parsed_params

    def process_media(
        self, media_path: str, query_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process media file with vision processor.

        Args:
            media_path: Path to the media file (image or video)
            query_params: Parsed query parameters from LLM

        Returns:
            Dictionary containing detection results

        Raises:
            ValueError: If vision processor is not set
        """
        if not self.vision_processor:
            error_msg = "Vision processor not set. Call set_vision_processor() first."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Check media type
        is_video = is_video_file(media_path)
        self.logger.info(f"Processing {'video' if is_video else 'image'}: {media_path}")

        if is_video:
            # For video processing, check if we need to adjust sample rate based on task
            sample_rate = 5  # Default sample rate (process every 5th frame)
            task_type = query_params.get("task_type", "")
            if task_type in ["tracking", "activity"]:
                # Use a more frequent sampling for tracking and activity detection
                # to capture more temporal information
                sample_rate = 2
                self.logger.debug(
                    f"Using higher sample rate ({sample_rate}) for {task_type} task"
                )

            # Get all detections with YOLO-World + ByteTracker integration
            self.logger.info(f"Processing video with sample_rate={sample_rate}")
            detections = self.vision_processor.process_video(
                media_path, query_params, sample_rate
            )
            frame_count = len(detections.get("frame_detections", {}))
            self.logger.info(
                f"Video processing complete. "
                f"Found {frame_count} frames with detections"
            )
            return detections
        else:
            # Get all detections with YOLO-World for image
            self.logger.info("Processing image")
            detections = self.vision_processor.process_image(media_path, query_params)
            num_objects = len(detections.get("objects", []))
            self.logger.info(f"Image processing complete. Found {num_objects} objects")
            return detections

    def generate_explanation(self, query: str, detections: Dict[str, Any]) -> str:
        """
        Generate explanation using LLM processor.

        Args:
            query: Original user query
            detections: Detection results from vision processor

        Returns:
            Natural language explanation of the detection results

        Raises:
            ValueError: If LLM processor is not set
        """
        if not self.llm_processor:
            error_msg = "LLM processor not set. Call set_llm_processor() first."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Determine if this is a video based on detection structure
        is_video = "frame_detections" in detections
        self.logger.info(
            f"Generating explanation for {'video' if is_video else 'image'}"
        )

        explanation = self.llm_processor.generate_explanation(
            query, detections, is_video
        )
        self.logger.debug(
            f"Generated explanation length: {len(explanation)} characters"
        )
        return explanation

    def get_highlighted_objects(self):
        """Get highlighted objects from LLM processor"""
        if not self.llm_processor:
            return []

        return self.llm_processor.get_highlighted_objects()

    def has_processors(self) -> bool:
        """Check if both processors are set"""
        return self.llm_processor is not None and self.vision_processor is not None
