"""
Advanced color detection utility class
"""

from typing import Dict, List, Tuple, Union

import cv2
import numpy as np


class ColorDetector:
    """
    Advanced color detection utility class for image processing.
    Provides precise color detection with support for a wide range of color shades.

    This class can be integrated with any object detection system to add color attribute detection.
    """

    # Define comprehensive color ranges in HSV
    # Format: (lower_bound, upper_bound, color_name)
    # H: 0-180, S: 0-255, V: 0-255 for OpenCV
    COLOR_RANGES = [
        # Grayscale spectrum
        ((0, 0, 0), (180, 30, 50), "black"),
        ((0, 0, 51), (180, 30, 140), "dark_gray"),
        ((0, 0, 141), (180, 30, 200), "gray"),
        ((0, 0, 201), (180, 30, 255), "white"),
        # Red spectrum (wraps around the hue spectrum)
        ((0, 70, 50), (10, 255, 255), "red"),
        ((160, 70, 50), (180, 255, 255), "red"),
        ((0, 70, 50), (10, 150, 200), "dark_red"),
        ((160, 70, 50), (180, 150, 200), "dark_red"),
        ((0, 50, 200), (10, 150, 255), "light_red"),
        ((160, 50, 200), (180, 150, 255), "light_red"),
        # Pink spectrum
        ((145, 30, 190), (165, 120, 255), "pink"),
        ((145, 120, 190), (165, 255, 255), "hot_pink"),
        ((0, 30, 200), (10, 70, 255), "salmon"),
        ((160, 30, 200), (180, 70, 255), "salmon"),
        # Orange spectrum
        ((5, 120, 150), (25, 255, 255), "orange"),
        ((5, 150, 150), (18, 255, 255), "pure_orange"),
        ((18, 150, 150), (27, 255, 255), "amber"),
        ((10, 70, 150), (25, 150, 200), "bronze"),
        ((10, 100, 100), (25, 150, 150), "brown"),
        # Yellow spectrum
        ((25, 100, 150), (40, 255, 255), "yellow"),
        ((25, 150, 200), (40, 255, 255), "bright_yellow"),
        ((25, 100, 100), (40, 150, 150), "olive"),
        ((25, 50, 150), (40, 100, 200), "gold"),
        ((25, 30, 200), (40, 70, 255), "cream"),
        # Green spectrum
        ((40, 70, 50), (85, 255, 255), "green"),
        ((40, 150, 100), (70, 255, 200), "pure_green"),
        ((70, 150, 100), (85, 255, 200), "lime_green"),
        ((40, 100, 50), (70, 200, 100), "dark_green"),
        ((40, 50, 150), (70, 100, 200), "light_green"),
        ((35, 30, 70), (50, 80, 120), "olive_green"),
        # Cyan spectrum
        ((85, 70, 100), (105, 255, 255), "cyan"),
        ((85, 150, 150), (105, 255, 255), "bright_cyan"),
        ((85, 70, 100), (105, 150, 150), "teal"),
        ((85, 50, 150), (105, 100, 200), "turquoise"),
        # Blue spectrum
        ((105, 70, 50), (135, 255, 255), "blue"),
        ((105, 150, 100), (125, 255, 200), "pure_blue"),
        ((105, 100, 50), (125, 200, 100), "dark_blue"),
        ((105, 50, 150), (125, 100, 200), "light_blue"),
        ((125, 150, 100), (135, 255, 200), "royal_blue"),
        # Purple spectrum
        ((135, 70, 50), (160, 255, 255), "purple"),
        ((135, 150, 100), (150, 255, 200), "purple"),
        ((135, 100, 50), (150, 200, 100), "dark_purple"),
        ((135, 50, 150), (150, 100, 200), "light_purple"),
        ((140, 150, 100), (160, 255, 200), "violet"),
        ((135, 30, 100), (150, 70, 150), "lavender"),
        # Brown spectrum (additional)
        ((0, 50, 50), (20, 150, 120), "brown"),
        ((5, 50, 50), (25, 100, 100), "dark_brown"),
        ((5, 30, 100), (25, 70, 150), "light_brown"),
        ((10, 30, 120), (30, 70, 170), "tan"),
        ((10, 20, 140), (30, 50, 200), "beige"),
        # Metallic colors
        ((0, 0, 150), (180, 20, 200), "silver"),
        ((20, 30, 100), (40, 70, 150), "gold"),
    ]

    @classmethod
    def detect_color(
            cls, image_region: np.ndarray, return_all: bool = False,
            threshold: float = 0.15
    ) -> Union[str, Dict[str, float]]:
        """
        Detect the dominant color(s) in an image region using HSV color space.

        Args:
            image_region: Image region as numpy array (BGR format)
            return_all: If True, returns all detected colors with percentages
            threshold: Minimum percentage for a color to be considered (when return_all=False)

        Returns:
            Dominant color name if return_all=False, or dictionary of {color_name: percentage} if return_all=True
        """
        # Check if region is valid
        if (
                image_region is None
                or image_region.size == 0
                or image_region.shape[0] == 0
                or image_region.shape[1] == 0
        ):
            return "unknown" if not return_all else {}

        # Convert to HSV for better color analysis
        try:
            hsv_region = cv2.cvtColor(image_region, cv2.COLOR_BGR2HSV)
        except cv2.error:
            return "unknown" if not return_all else {}

        # Create a mask for each color range and count pixels
        color_counts = {}
        nonzero_pixels = np.count_nonzero(
            hsv_region[:, :, 0] >= 0
        )  # Total valid pixels

        if nonzero_pixels == 0:
            return "unknown" if not return_all else {}

        for lower, upper, color_name in cls.COLOR_RANGES:
            lower_bound = np.array(lower)
            upper_bound = np.array(upper)

            # Create mask for this color range
            mask = cv2.inRange(hsv_region, lower_bound, upper_bound)
            pixel_count = np.count_nonzero(mask)

            # Calculate percentage of pixels in this color range
            percentage = pixel_count / nonzero_pixels

            # Add to color counts if percentage is significant
            if (
                    percentage > 0.05
            ):  # Only count colors covering at least 5% of the region
                if color_name in color_counts:
                    color_counts[color_name] += percentage
                else:
                    color_counts[color_name] = percentage

        # If return_all is True, return the full color dictionary
        if return_all:
            # Sort by percentage (highest first)
            return dict(sorted(color_counts.items(), key=lambda x: x[1], reverse=True))

        # Get the color with the highest percentage
        if not color_counts:
            return "unknown"

        dominant_color = max(color_counts.items(), key=lambda x: x[1])

        # Only return a color if it covers a significant portion of the object
        if dominant_color[1] >= threshold:  # At least threshold % of pixels
            return dominant_color[0]
        else:
            return "multicolored"  # No single dominant color

    @classmethod
    def detect_colors_layered(
            cls, image_region: np.ndarray, max_colors: int = 3
    ) -> List[str]:
        """
        Detect up to max_colors different colors in the image region in order of dominance.

        Args:
            image_region: Image region as numpy array (BGR format)
            max_colors: Maximum number of colors to return

        Returns:
            List of color names in order of dominance
        """
        # Get all colors with their percentages
        color_percentages = cls.detect_color(image_region, return_all=True)

        # Return the top colors
        return [color for color, _ in list(color_percentages.items())[:max_colors]]

    @classmethod
    def get_color_profile(cls, image_region: np.ndarray) -> Dict[str, any]:
        """
        Get a comprehensive color profile of the image region.

        Args:
            image_region: Image region as numpy array (BGR format)

        Returns:
            Dictionary with color information:
                - dominant_color: Main color name
                - color_percentages: Dictionary of all detected colors and their percentages
                - is_multicolored: Boolean indicating if the object has multiple significant colors
                - brightness: Average brightness value
                - saturation: Average saturation value
        """
        # Check if region is valid
        if image_region is None or image_region.size == 0:
            return {
                "dominant_color": "unknown",
                "color_percentages": {},
                "is_multicolored": False,
                "brightness": 0,
                "saturation": 0,
            }

        # Get all color percentages
        color_percentages = cls.detect_color(image_region, return_all=True)

        # Get dominant color
        dominant_color = "unknown"
        if color_percentages:
            dominant_color = max(color_percentages.items(), key=lambda x: x[1])[0]

        # Determine if multicolored (more than one color with significant percentage)
        significant_colors = [c for c, p in color_percentages.items() if p >= 0.2]
        is_multicolored = len(significant_colors) > 1

        # Compute average brightness and saturation
        try:
            hsv_region = cv2.cvtColor(image_region, cv2.COLOR_BGR2HSV)
            avg_saturation = np.mean(hsv_region[:, :, 1])
            avg_brightness = np.mean(hsv_region[:, :, 2])
        except:
            avg_saturation = 0
            avg_brightness = 0

        return {
            "dominant_color": dominant_color,
            "color_percentages": color_percentages,
            "is_multicolored": is_multicolored,
            "brightness": float(avg_brightness),
            "saturation": float(avg_saturation),
        }

    @classmethod
    def get_color_name(cls, bgr_color: Tuple[int, int, int]) -> str:
        """
        Get the name of a color given its BGR values.

        Args:
            bgr_color: Tuple of (Blue, Green, Red) values (0-255)

        Returns:
            Name of the closest matching color
        """
        # Convert single BGR color to a 1x1 image
        pixel = np.array([[[bgr_color[0], bgr_color[1], bgr_color[2]]]], dtype=np.uint8)
        return cls.detect_color(pixel)

    @classmethod
    def find_objects_by_color(cls, image: np.ndarray, target_color: str) -> np.ndarray:
        """
        Create a mask highlighting areas of the specified color in the image.

        Args:
            image: Input image as numpy array (BGR format)
            target_color: Color name to find

        Returns:
            Binary mask where areas of the target color are white (255)
        """
        # Check if image is valid
        if image is None or image.size == 0:
            return np.array([], dtype=np.uint8)

        # Convert to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create combined mask for the target color
        combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Find all ranges matching the target color
        for lower, upper, color_name in cls.COLOR_RANGES:
            if color_name == target_color:
                mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
                combined_mask = cv2.bitwise_or(combined_mask, mask)

        return combined_mask

    @classmethod
    def visualize_colors(cls, image_region: np.ndarray) -> np.ndarray:
        """
        Create a visualization of detected colors in the image region.

        Args:
            image_region: Image region as numpy array (BGR format)

        Returns:
            Visualization image with color information
        """
        if image_region is None or image_region.size == 0:
            return np.zeros((100, 200, 3), dtype=np.uint8)

        # Get color profile
        profile = cls.get_color_profile(image_region)

        # Create visualization image
        height, width = 30 * len(profile["color_percentages"]) + 60, 300
        vis_image = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Add title
        cv2.putText(
            vis_image,
            "Color Analysis",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )

        # Add color bars
        y_offset = 60
        for i, (color, percentage) in enumerate(profile["color_percentages"].items()):
            # Draw color name and percentage
            text = f"{color}: {percentage * 100:.1f}%"
            cv2.putText(
                vis_image,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

            # Draw color bar
            bar_width = int(percentage * 150)

            # Try to use actual color for visualization
            try:
                # Find a matching color range
                for lower, upper, c_name in cls.COLOR_RANGES:
                    if c_name == color:
                        # Use the middle value of the range for visualization
                        h = (lower[0] + upper[0]) // 2
                        s = (lower[1] + upper[1]) // 2
                        v = (lower[2] + upper[2]) // 2

                        # Convert HSV to BGR for display
                        hsv_color = np.uint8([[[h, s, v]]])
                        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
                        break
                else:
                    # Default color if no match found
                    bgr_color = (0, 0, 0)
            except:
                # Fallback color
                bgr_color = (0, 0, 0)

            cv2.rectangle(
                vis_image,
                (150, y_offset - 15),
                (150 + bar_width, y_offset),
                bgr_color.tolist(),
                -1,
            )

            y_offset += 30

        return vis_image
