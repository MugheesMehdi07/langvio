"""
Enhanced base classes for LLM processors with expanded capabilities
"""

import importlib.util
import json
import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers.json import SimpleJsonOutputParser
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
    format_video_summary,
    parse_explanation_response,
    process_image_detections_and_format_summary,
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

        # Ensure processor is initialized
        if not hasattr(self, "query_chain") or self.query_chain is None:
            if not self.initialize():
                error_msg = "LLM processor initialization failed. Cannot parse query."
                self.logger.error(error_msg)
                return {"error": error_msg}

        try:
            # Invoke the chain with proper output parsing
            parsed = self.query_chain.invoke({"query": query, "history": []})

            # Ensure all required fields exist with defaults
            parsed = self._ensure_parsed_fields(parsed)

            # Log the parsed query
            self.logger.debug(f"Parsed query: {json.dumps(parsed, indent=2)}")

            return parsed

        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error parsing query: {error_msg}")

            return {"error": error_msg}

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
                f"Invalid task type: {parsed['task_type']}. "
                f"Using 'identification' instead."
            )
            parsed["task_type"] = "identification"

        return parsed

    def generate_explanation(
        self,
        query: str,
        detections: Dict[str, List[Dict[str, Any]]],
        is_video: bool = False,
    ) -> str:
        """Generate an explanation based on detection results."""
        self.logger.info("Generating explanation for detection results")

        # Ensure processor is initialized
        if not hasattr(self, "explanation_chain") or self.explanation_chain is None:
            if not self.initialize():
                error_msg = "LLM processor initialization failed. Cannot generate explanation."
                self.logger.error(error_msg)
                return f"Error: {error_msg}"

        # Get the original parsed query
        parsed_query = self.parse_query(query)

        # Check if query parsing failed
        if "error" in parsed_query:
            error_msg = parsed_query["error"]
            self.logger.warning(
                f"Query parsing failed, using fallback parsing. Error: {error_msg}"
            )
            # Use fallback parsed query with basic defaults
            parsed_query = {
                "target_objects": [],
                "count_objects": False,
                "task_type": "identification",
                "attributes": [],
                "spatial_relations": [],
                "activities": [],
                "custom_instructions": "",
            }
            # Try to extract target objects from query text as fallback
            query_lower = query.lower()
            if "count" in query_lower:
                parsed_query["count_objects"] = True

        if is_video:
            # For videos, we have compressed results - format them directly
            detection_summary = format_video_summary(detections, parsed_query)

            # No object highlighting for videos since we don't have per-frame data
            self._highlighted_objects = []

        else:
            # For images, use the existing indexing and formatting approach
            detection_summary, detection_map = (
                process_image_detections_and_format_summary(detections, parsed_query)
            )

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
                if is_video:
                    # For videos, just return the explanation without highlighting
                    explanation_text = response.content
                    if "EXPLANATION:" in explanation_text:
                        explanation_text = explanation_text.split("EXPLANATION:", 1)[
                            1
                        ].strip()

                    # Clean up any highlight sections that might appear
                    if "HIGHLIGHT_OBJECTS:" in explanation_text:
                        explanation_text = explanation_text.split("HIGHLIGHT_OBJECTS:")[
                            0
                        ].strip()

                    self._highlighted_objects = []
                    return explanation_text
                else:
                    # For images, use the existing parsing with highlighting
                    explanation_text, highlight_objects = parse_explanation_response(
                        response.content, detection_map
                    )
                    self._highlighted_objects = highlight_objects
                    return explanation_text
            else:
                return "Error generating explanation: No valid response from LLM"

        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error generating explanation: {error_msg}")
            return f"Error analyzing the media: {error_msg}"

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

