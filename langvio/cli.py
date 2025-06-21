"""
Command-line interface for langvio
"""

import argparse
import logging
import os
import sys

from langvio import create_pipeline
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

        # Count detections by label
        counts = {}
        for frame_dets in result["detections"].values():
            for det in frame_dets:
                label = det["label"]
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
