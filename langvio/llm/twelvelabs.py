"""
TwelveLabs Pegasus LLM processor implementation.

Pegasus is a video-understanding foundation model. Unlike the chat LLM
processors (OpenAI, Gemini) that reason over detections produced by a local
vision model, Pegasus analyses the *video itself* server-side. This lets
langvio answer natural-language questions about a video without running any
local computer-vision pipeline -- the heavy lifting happens on TwelveLabs'
servers and only text comes back.

This processor is fully opt-in: it is only registered when the optional
``twelvelabs`` package is installed and is never used as a default.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from langvio.llm.base import BaseLLMProcessor
from langvio.prompts.constants import TASK_TYPES
from langvio.utils.file_utils import is_video_file

# Pegasus requires max_tokens between 512 and 98304.
_MIN_PEGASUS_MAX_TOKENS = 512


class PegasusProcessor(BaseLLMProcessor):
    """LLM processor backed by TwelveLabs Pegasus video understanding."""

    def __init__(
        self,
        name: str = "pegasus",
        model_name: str = "pegasus1.5",
        model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize the Pegasus processor.

        Args:
            name: Processor name
            model_name: Pegasus model id (e.g. "pegasus1.5" or "pegasus1.2")
            model_kwargs: Extra analyze parameters (temperature, max_tokens)
            **kwargs: Additional processor parameters
        """
        config = {
            "model_name": model_name,
            "model_kwargs": model_kwargs or {},
            **kwargs,
        }
        super().__init__(name, config)
        self.logger = logging.getLogger(__name__)
        self.client: Optional[Any] = None

    def _initialize_llm(self) -> None:
        """
        Initialize the TwelveLabs client.

        Pegasus is not a LangChain chat model, so unlike the other processors
        this sets up a TwelveLabs API client rather than ``self.llm``. Query
        parsing and explanation are handled directly against that client.
        """
        try:
            if not self.is_package_installed("twelvelabs"):
                raise ImportError(
                    "The 'twelvelabs' package is required to use Pegasus models. "
                    "Please install it with 'pip install langvio[twelvelabs]'"
                )

            from twelvelabs import TwelveLabs

            api_key = os.getenv("TWELVELABS_API_KEY")
            if not api_key:
                self.logger.warning(
                    "TWELVELABS_API_KEY environment variable not found. "
                    "Please set it using 'export TWELVELABS_API_KEY=your_key' "
                    "or add it to your .env file"
                )
                raise ValueError(
                    "TWELVELABS_API_KEY environment variable is required. "
                    "Please set it using 'export TWELVELABS_API_KEY=your_key' "
                    "or add it to your .env file"
                )

            self.client = TwelveLabs(api_key=api_key)
            self.logger.info(
                f"Initialized TwelveLabs Pegasus model: {self.config['model_name']}"
            )
        except Exception as e:
            self.logger.error(f"Error initializing TwelveLabs Pegasus model: {e}")
            raise

    def _setup_prompts(self) -> None:
        """
        Pegasus does not use LangChain prompt chains.

        Query parsing and explanation talk to the Pegasus API directly, so the
        chain attributes the base class relies on are left unset.
        """
        self.query_chain = None  # type: ignore[assignment]
        self.explanation_chain = None  # type: ignore[assignment]

    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Extract lightweight structured parameters from the query.

        Pegasus reasons over the raw query against the video, so it does not
        need the rich JSON parse the chat LLMs produce. We do a cheap
        keyword-based extraction so the rest of the pipeline (vision sampling,
        visualization) still gets sensible parameters.
        """
        if self.client is None and not self.initialize():
            return {"error": "Pegasus processor initialization failed."}

        query_lower = query.lower()
        task_type = "identification"
        if any(w in query_lower for w in ("count", "how many", "number of")):
            task_type = "counting"
        elif any(w in query_lower for w in ("track", "follow", "trajectory")):
            task_type = "tracking"
        elif any(
            w in query_lower for w in ("doing", "activity", "happening", "action")
        ):
            task_type = "activity"
        elif any(w in query_lower for w in ("where", "located", "position", "on the")):
            task_type = "verification"

        if task_type not in TASK_TYPES:
            task_type = "identification"

        return self._ensure_parsed_fields(
            {
                "task_type": task_type,
                "count_objects": task_type == "counting",
                "custom_instructions": query,
            }
        )

    def analyze_video(self, video: str, prompt: str) -> str:
        """
        Run Pegasus video understanding on a video and return its answer.

        Args:
            video: A public video URL, a TwelveLabs asset id, or a local path.
                Local paths are uploaded as a TwelveLabs asset first.
            prompt: Natural-language instruction / question for the video.

        Returns:
            Pegasus' text response.
        """
        if self.client is None and not self.initialize():
            raise RuntimeError("Pegasus processor is not initialized.")

        from twelvelabs.types.video_context import (
            VideoContext_AssetId,
            VideoContext_Url,
        )

        if video.startswith(("http://", "https://")):
            video_context: Any = VideoContext_Url(url=video)
        elif os.path.exists(video):
            asset = self.client.assets.create(method="direct", file=video)
            video_context = VideoContext_AssetId(asset_id=asset.id)
        else:
            # Treat as an existing TwelveLabs asset id.
            video_context = VideoContext_AssetId(asset_id=video)

        model_kwargs = dict(self.config.get("model_kwargs", {}))
        max_tokens = max(
            int(model_kwargs.pop("max_tokens", _MIN_PEGASUS_MAX_TOKENS)),
            _MIN_PEGASUS_MAX_TOKENS,
        )

        response = self.client.analyze(
            model_name=self.config["model_name"],
            video=video_context,
            prompt=prompt,
            max_tokens=max_tokens,
            **model_kwargs,
        )
        return response.data or ""

    def generate_explanation(
        self,
        query: str,
        detections: Dict[str, List[Dict[str, Any]]],
        is_video: bool = False,
        media_path: Optional[str] = None,
    ) -> str:
        """
        Generate an explanation using Pegasus video understanding.

        When a video ``media_path`` is supplied, the video is sent straight to
        Pegasus, which answers the query from the footage itself. Pegasus is a
        video-native model, so for images (or when no path is available) it
        falls back to a plain detection summary.
        """
        self._highlighted_objects: List[Dict[str, Any]] = []

        if media_path and is_video_file(media_path):
            try:
                self.logger.info("Generating explanation with Pegasus video analysis")
                return self.analyze_video(media_path, query)
            except Exception as e:
                self.logger.error(f"Pegasus video analysis failed: {e}")
                return f"Error analyzing the video with Pegasus: {e}"

        # Pegasus targets video; summarise detections for the image case.
        from langvio.utils.llm_utils import process_image_detections_and_format_summary

        parsed_query = self.parse_query(query)
        summary, _ = process_image_detections_and_format_summary(
            detections, parsed_query
        )
        return (
            "Pegasus is a video-understanding model and works best on videos. "
            "Detection summary for this image:\n" + summary
        )
