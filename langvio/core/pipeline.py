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
        """Process a query on media with enhanced capabilities"""
        self.logger.info(f"Processing query: {query}")

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

        try:
            # 1. Parse query with LLM to get structured parameters
            query_params = self.processor_manager.parse_query(query)
            self.logger.info(f"Parsed query params: {query_params}")

            # 2. Run detection with vision processor
            all_detections = self.processor_manager.process_media(
                media_path, query_params
            )

            # Log detection results to JSON
            self._log_detections_to_json(all_detections, query, media_path)

            # 3. Generate explanation using all detected objects and metrics
            explanation = self.processor_manager.generate_explanation(
                query, all_detections
            )

            # 4. Get highlighted objects from the LLM processor
            highlighted_objects = self.processor_manager.get_highlighted_objects()

            # 5. Create visualization with highlighted objects
            output_path = self.visualization_manager.create_visualization(
                media_path, all_detections, highlighted_objects, query_params
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

            self.logger.info("Processed query successfully")
            return result

        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            raise

    def _log_detections_to_json(self, all_detections: Dict[str, Any], query: str, media_path: str):
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
            "detections": all_detections
        }
        
        # Write to JSON file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(detection_log, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Detection results logged to: {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to log detection results to file: {e}")
