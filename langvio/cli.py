"""
Command-line interface for langvio
"""

import argparse
import logging
import os
import sys

from langvio import create_pipeline
from langvio.llm.qwen_api import DEFAULT_API_KEY_ENV
from langvio.utils.file_utils import is_image_file, is_video_file
from langvio.utils.logging import setup_logging


def main():
    """Main entry point for the langvio CLI"""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="langvio: Connect LLMs with video models"
    )

    # Required arguments
    parser.add_argument("--query", "-q", required=True, help="Natural language query")
    parser.add_argument(
        "--media", "-m", required=True, help="Path to image or video file"
    )

    # Optional arguments
    parser.add_argument("--config", "-c", help="Path to configuration file")
    parser.add_argument("--llm", "-l", help="LLM processor to use")
    parser.add_argument("--vision", "-v", help="Vision processor to use")
    parser.add_argument("--output", "-o", help="Output directory for processed media")
    parser.add_argument(
        "--api-key",
        help="Qwen API key (defaults to OPENROUTER_API_KEY / QWEN_API_KEY)",
    )
    parser.add_argument(
        "--api-model",
        help="Hosted Qwen-VL model id, e.g. qwen/qwen3-vl-8b-instruct",
    )
    parser.add_argument(
        "--api-base-url",
        help="OpenAI-compatible endpoint (defaults to OpenRouter)",
    )
    parser.add_argument(
        "--backend",
        choices=["api", "local", "auto"],
        help="Run Qwen-VL over the API (default) or with local weights",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument("--log-file", help="Path to log file")
    parser.add_argument(
        "--list-models", action="store_true", help="List available models and exit"
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging({"level": args.log_level, "file": args.log_file})

    logger = logging.getLogger(__name__)

    # CLI flags map onto the env vars the Qwen backend already reads, so they
    # take precedence over .env (which was loaded when langvio was imported).
    for value, env_var in (
        (args.api_key, DEFAULT_API_KEY_ENV),
        (args.api_key, "QWEN_API_KEY"),
        (args.api_model, "QWEN_API_MODEL"),
        (args.api_base_url, "QWEN_API_BASE_URL"),
        (args.backend, "LANGVIO_BACKEND"),
    ):
        if value:
            os.environ[env_var] = value

    # List models if requested
    if args.list_models:
        list_available_models()
        return 0

    # Check if media file exists
    if not os.path.exists(args.media):
        logger.error("Media file not found: %s", args.media)
        return 1

    # Check if media file is supported
    if not is_image_file(args.media) and not is_video_file(args.media):
        logger.error("Unsupported media file format: %s", args.media)
        return 1

    try:
        # Create pipeline
        pipeline = create_pipeline(
            config_path=args.config, llm_name=args.llm, vision_name=args.vision
        )

        # Update output directory if specified
        if args.output:
            pipeline.config.config["media"]["output_dir"] = args.output
            os.makedirs(args.output, exist_ok=True)

        # Process query
        result = pipeline.process(args.query, args.media)

        # Print results
        print("\n===== langvio Results =====")
        print("Query: {}".format(result["query"]))
        print("Media: {} ({})".format(result["media_path"], result["media_type"]))
        print("Output: {}".format(result["output_path"]))
        print("\nExplanation: {}".format(result["explanation"]))
        print("\nDetection summary:")

        # Count detections by label. The detections payload can take a few
        # shapes (per-frame mapping, or a single {"objects": [...], "summary":
        # {...}} dict), so walk it defensively and count anything with a label.
        counts = {}

        def _iter_objects(container):
            if isinstance(container, dict):
                if isinstance(container.get("objects"), list):
                    yield from container["objects"]
                else:
                    for value in container.values():
                        yield from _iter_objects(value)
            elif isinstance(container, list):
                for item in container:
                    if isinstance(item, dict) and "label" in item:
                        yield item
                    else:
                        yield from _iter_objects(item)

        for det in _iter_objects(result.get("detections", {})):
            label = det.get("label")
            if label is not None:
                counts[label] = counts.get(label, 0) + 1

        for label, count in counts.items():
            print("- {}: {}".format(label, count))

        print("\nProcessing complete!")
        return 0

    except Exception as e:
        logger.error("Error processing query: %s", e, exc_info=True)
        return 1


def list_available_models():
    """List all available models in the registry"""
    from langvio import registry

    print("\n===== Available LLM Processors =====")
    for name, cls in registry.list_llm_processors().items():
        print("- {}".format(name))

    print("\n===== Available Vision Processors =====")
    for name, cls in registry.list_vision_processors().items():
        print("- {}".format(name))


if __name__ == "__main__":
    sys.exit(main())
