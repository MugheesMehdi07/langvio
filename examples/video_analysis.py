"""
Example script for counting objects with Langvio
"""

import logging
import os

import cv2

from langvio import create_pipeline

# Set up logging
logging.basicConfig(level=logging.INFO)


def main():
    """Run an object counting example"""
    # Create default pipeline
    pipeline = create_pipeline()

    # Create output directory
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    # Example for video counting
    video_path = "input2_processed.mp4"  # Replace with your video path

    if os.path.exists(video_path):
        print(f"\n--- Processing video: {video_path} ---")

        # Counting query
        query = "Count the number of vehicles in this video"
        print(f"Query: {query}")

        # Process the query
        result = pipeline.process(query, video_path)

        # Display results
        print(f"Output saved to: {result['output_path']}")
        print(f"Explanation: {result['explanation']}")


if __name__ == "__main__":
    main()
