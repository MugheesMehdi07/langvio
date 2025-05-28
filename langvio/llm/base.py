"""
Enhanced base classes for LLM processors with expanded capabilities
"""

import importlib.util
import json
import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional

from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langvio.core.base import Processor
from langvio.prompts.constants import TASK_TYPES
from langvio.prompts.templates import (
    EXPLANATION_TEMPLATE,
    QUERY_PARSING_TEMPLATE,
    SYSTEM_PROMPT,
)
from langvio.utils.llm_utils import (
    format_detection_summary,
    index_detections,
    parse_explanation_response,
)


class BaseLLMProcessor(Processor):
    """Enhanced base class for all LLM processors"""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize LLM processor."""
        super().__init__(name, config)
        self.logger = logging.getLogger(__name__)
        self.llm = None
        self.query_chat_prompt = None
        self.explanation_chat_prompt = None

    def initialize(self) -> bool:
        """Initialize the processor with its configuration."""
        try:
            # Initialize the specific LLM implementation
            self._initialize_llm()

            # Set up prompts
            self._setup_prompts()

            return True
        except Exception as e:
            self.logger.error(f"Error initializing LLM processor: {e}")
            return False

    @abstractmethod
    def _initialize_llm(self) -> None:
        """Initialize the specific LLM implementation."""

    def _setup_prompts(self) -> None:
        """Set up the prompt templates with system message."""
        system_message = SystemMessage(content=SYSTEM_PROMPT)

        # Query parsing prompt
        self.query_chat_prompt = ChatPromptTemplate.from_messages(
            [
                system_message,
                MessagesPlaceholder(variable_name="history"),
                ("user", QUERY_PARSING_TEMPLATE),
            ]
        )

        # Explanation prompt
        self.explanation_chat_prompt = ChatPromptTemplate.from_messages(
            [
                system_message,
                MessagesPlaceholder(variable_name="history"),
                ("user", EXPLANATION_TEMPLATE),
            ]
        )

        # Create chains
        json_parser = SimpleJsonOutputParser()
        self.query_chain = self.query_chat_prompt | self.llm | json_parser
        self.explanation_chain = self.explanation_chat_prompt | self.llm

    def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse a natural language query into structured parameters."""
        self.logger.info(f"Parsing query: {query}")

        try:
            # Invoke the chain with proper output parsing
            parsed = self.query_chain.invoke({"query": query, "history": []})

            # Ensure all required fields exist with defaults
            parsed = self._ensure_parsed_fields(parsed)

            # Log the parsed query
            self.logger.debug(f"Parsed query: {json.dumps(parsed, indent=2)}")

            return parsed

        except Exception as e:
            self.logger.error(f"Error parsing query: {e}")

            return {"error": e}

    def _ensure_parsed_fields(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all required fields exist in the parsed query."""
        defaults = {
            "target_objects": [],
            "count_objects": False,
            "task_type": "identification",
            "attributes": [],
            "spatial_relations": [],
            "activities": [],
            "custom_instructions": "",
        }

        # Add any missing fields with defaults
        for key, default_value in defaults.items():
            if key not in parsed or parsed[key] is None:
                parsed[key] = default_value

        # Ensure task_type is valid
        if parsed["task_type"] not in TASK_TYPES:
            self.logger.warning(
                f"Invalid task type: {parsed['task_type']}. Using 'identification' instead."
            )
            parsed["task_type"] = "identification"

        return parsed

    def generate_explanation(
        self, query: str, detections: Dict[str, List[Dict[str, Any]]] , is_video: bool=False
    ) -> str:
        """Generate an explanation based on detection results."""
        self.logger.info("Generating explanation for detection results")

        # Get the original parsed query
        parsed_query = self.parse_query(query)

        if is_video:
            detection_summary = detections
        else:
            # Add unique IDs to each detection for reference
            indexed_detections, detection_map = index_detections(detections)

            # Format detection summary with object IDs
            detection_summary = format_detection_summary(indexed_detections, parsed_query)

        # Create a summary of detections
        print(detection_summary)
        try:
            # Invoke the explanation chain
            response = self.explanation_chain.invoke(
                {
                    "query": query,
                    "detection_summary": detection_summary,
                    "parsed_query": json.dumps(parsed_query, indent=2),
                    "history": [],
                }
            )

            # Parse the response
            if response and hasattr(response, "content"):
                explanation_text, highlight_objects = parse_explanation_response(
                    response.content, detection_map
                )

                # Store the highlighted objects for visualization
                self._highlighted_objects = highlight_objects

                return explanation_text
            else:
                return "Error generating explanation: No valid response from LLM"

        except Exception as e:
            self.logger.error(f"Error generating explanation: {e}")
            return f"Error analyzing the image: {e}"

    def get_highlighted_objects(self) -> List[Dict[str, Any]]:
        """
        Get the objects that were highlighted in the last explanation.

        Returns:
            List of highlighted objects with frame references
        """
        return getattr(self, "_highlighted_objects", [])

    def is_package_installed(self, package_name: str) -> bool:
        """Check if a Python package is installed."""
        return importlib.util.find_spec(package_name) is not None

    def parse_solution_results(solution_results):
        """
        Convert YOLO11 SolutionResults objects to well-structured dictionaries.

        Args:
            solution_results: Either object counting or speed estimation results

        Returns:
            Dictionary with structured information
        """
        # Convert to string first
        result_str = str(solution_results)

        # Create a base dictionary
        parsed_data = {}

        # Check if it's object counting results
        if "in_count" in result_str and "out_count" in result_str:
            # Extract the basic counts
            parsed_data["type"] = "object_counting"
            parsed_data["in_count"] = solution_results.in_count
            parsed_data["out_count"] = solution_results.out_count
            parsed_data["total_tracks"] = solution_results.total_tracks

            # Extract class-wise counts in a more accessible format
            class_counts = {}
            for class_name, directions in solution_results.classwise_count.items():
                # Only include classes that have non-zero counts
                if directions["IN"] > 0 or directions["OUT"] > 0:
                    class_counts[class_name] = {
                        "in": directions["IN"],
                        "out": directions["OUT"],
                        "total": directions["IN"] + directions["OUT"]
                    }

            parsed_data["class_counts"] = class_counts

            # Add a summary for quick access
            active_classes = [cls for cls, counts in class_counts.items()]
            parsed_data["summary"] = {
                "total_objects": parsed_data["in_count"] + parsed_data["out_count"],
                "active_classes": active_classes,
                "most_common_class": max(class_counts.items(), key=lambda x: x[1]["total"])[0] if class_counts else None
            }

        # Check if it's speed estimation results
        elif "total_tracks" in result_str:
            parsed_data["type"] = "speed_estimation"
            parsed_data["total_tracks"] = solution_results.total_tracks

            # If there are additional attributes in the speed results, extract them
            # This will depend on what attributes the SolutionResults object has
            if hasattr(solution_results, "track_speeds"):
                parsed_data["track_speeds"] = solution_results.track_speeds

            if hasattr(solution_results, "avg_speed"):
                parsed_data["avg_speed"] = solution_results.avg_speed

            if hasattr(solution_results, "class_speeds"):
                parsed_data["class_speeds"] = solution_results.class_speeds

        return parsed_data
