"""
Enhanced core pipeline for connecting LLMs with vision models - refactored
"""

import logging
import os
from typing import Any, Dict, Optional

from langvio.config import Config
from langvio.core.processor_manager import ProcessorManager
from langvio.core.visualization_manager import VisualizationManager
from langvio.utils.file_utils import is_video_file
from langvio.utils.logging import setup_logging


class Pipeline:
    """Main pipeline for processing queries with LLMs and vision models"""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize pipeline"""
        # Initialize configuration
        self.config = Config(config_path)

        # Set up logging
        setup_logging(self.config.get_logging_config())
        self.logger = logging.getLogger(__name__)

        # Initialize managers
        self.processor_manager = ProcessorManager(self.config)
        self.visualization_manager = VisualizationManager(self.config)

        self.logger.info("Enhanced Pipeline initialized")

    def load_config(self, config_path: str) -> None:
        """Load configuration from file"""
        self.config.load_config(config_path)
        self.logger.info(f"Loaded configuration from {config_path}")

        # Update managers with new config
        self.processor_manager.config = self.config
        self.visualization_manager.config = self.config

    def set_llm_processor(self, processor_name: str) -> None:
        """Set the LLM processor"""
        self.processor_manager.set_llm_processor(processor_name)

    def set_vision_processor(self, processor_name: str) -> None:
        """Set the vision processor"""
        self.processor_manager.set_vision_processor(processor_name)

    def process(self, query: str, media_path: str) -> Dict[str, Any]:
        """
        Process a query on media with enhanced capabilities.

        This is the main entry point for processing queries. It orchestrates:
        1. Query parsing using LLM
        2. Object detection using vision models
        3. Explanation generation using LLM
        4. Visualization creation

        Args:
            query: Natural language query (e.g., "Count all red cars")
            media_path: Path to image or video file

        Returns:
            Dictionary containing:
                - query: Original query
                - media_path: Input media path
                - media_type: "image" or "video"
                - output_path: Path to processed/annotated media
                - explanation: Natural language explanation
                - detections: Raw detection results
                - query_params: Parsed query parameters
                - highlighted_objects: Objects highlighted in visualization

        Raises:
            ValueError: If processors are not set
            FileNotFoundError: If media file doesn't exist
        """
        self.logger.info(f"Processing query: {query} on media: {media_path}")

        # Check if processors are set
        if not self.processor_manager.has_processors():
            if not self.processor_manager.llm_processor:
                error_msg = "ERROR: LLM processor not set"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            if not self.processor_manager.vision_processor:
                error_msg = "ERROR: Vision processor not set"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

        # Check if media file exists
        if not os.path.exists(media_path):
            error_msg = f"ERROR: Media file not found: {media_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Check media type
        is_video = is_video_file(media_path)
        media_type = "video" if is_video else "image"
        self.logger.info(f"Detected media type: {media_type}")

        try:
            # Step 1: Parse query with LLM to get structured parameters
            # This extracts target objects, task type, attributes, etc.
            self.logger.debug("Step 1: Parsing query with LLM")
            query_params = self.processor_manager.parse_query(query)
            self.logger.info(
                f"Parsed query params: task_type={query_params.get('task_type')}, "
                f"target_objects={query_params.get('target_objects', [])}"
            )

            # Step 2: Run detection with vision processor
            # This performs object detection, tracking (for videos), and attribute extraction
            self.logger.debug("Step 2: Running vision detection")
            all_detections = self.processor_manager.process_media(
                media_path, query_params
            )

            # Log detection results to JSON for debugging/analysis
            self._log_detections_to_json(all_detections, query, media_path)

            # Step 3: Generate explanation using all detected objects and metrics
            # The LLM synthesizes the detection results into natural language
            self.logger.debug("Step 3: Generating explanation with LLM")
            explanation = self.processor_manager.generate_explanation(
                query, all_detections
            )

            # Step 4: Get highlighted objects from the LLM processor
            # These are objects that match the query criteria and should be emphasized
            highlighted_objects = self.processor_manager.get_highlighted_objects()
            self.logger.debug(f"Found {len(highlighted_objects)} highlighted objects")

            # Step 5: Create visualization with highlighted objects
            # This draws bounding boxes, labels, and highlights on the media
            self.logger.debug("Step 5: Creating visualization")
            output_path = self.visualization_manager.create_visualization(
                media_path, all_detections, highlighted_objects, query_params
            )

            # Prepare result dictionary
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

            self.logger.info(f"Processing complete. Output saved to: {output_path}")
            return result

        except Exception as e:
            self.logger.error(f"Error processing query: {e}", exc_info=True)
            raise

    def _log_detections_to_json(
        self, all_detections: Dict[str, Any], query: str, media_path: str
    ):
        """Log detection results to JSON file"""
        import json
        import os
        from datetime import datetime

        # Create logs directory if it doesn't exist
        logs_dir = "logs"
        os.makedirs(logs_dir, exist_ok=True)

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        media_name = os.path.splitext(os.path.basename(media_path))[0]
        filename = f"detections_{media_name}_{timestamp}.json"
        filepath = os.path.join(logs_dir, filename)

        # Prepare detection log data
        detection_log = {
            "query": query,
            "media_path": media_path,
            "timestamp": datetime.now().isoformat(),
            "detections": all_detections,
        }

        # Write to JSON file
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(detection_log, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Detection results logged to: {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to log detection results to file: {e}")
