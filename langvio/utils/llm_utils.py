"""
Enhanced utility functions for LLM processing with better YOLO11 metrics handling
"""

import json
import re
from typing import Any, Dict, List, Tuple


def index_detections(
    detections: Dict[str, List[Dict[str, Any]]],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
    """
    Create a copy of detections with unique object_id assigned to each detection.

    Args:
        detections: Original detection results

    Returns:
        Tuple of (indexed_detections, detection_map)
    """
    indexed_detections = {}
    detection_map = {}
    object_id_counter = 0

    for frame_key, frame_detections in detections.items():
        # Skip non-frame keys like "metrics" or "summary"
        if not frame_key.isdigit():
            indexed_detections[frame_key] = frame_detections
            continue

        indexed_detections[frame_key] = []

        for det in frame_detections:
            # Skip if not a dictionary
            if not isinstance(det, dict):
                continue

            # Create a copy with object_id
            object_id = f"obj_{object_id_counter}"
            det_copy = det.copy()
            det_copy["object_id"] = object_id

            # Add to indexed detections
            indexed_detections[frame_key].append(det_copy)

            # Add to object map with frame reference
            detection_map[object_id] = {
                "frame_key": frame_key,
                "detection": det,  # Store original detection
            }

            object_id_counter += 1

    return indexed_detections, detection_map


def format_detection_summary(
    detections: Dict[str, List[Dict[str, Any]]], query_params: Dict[str, Any]
) -> str:
    """
    Format the detection summary in a structured and readable way,
    including YOLO11 metrics if available.

    Args:
        detections: Dictionary with detection results and metrics
        query_params: Parsed query parameters

    Returns:
        Formatted detection summary string
    """
    summary_parts = []

    # Check if we have metrics or summary data from YOLO11
    has_metrics = "metrics" in detections
    has_summary = "summary" in detections

    # Add YOLO11 metrics summary if available
    if has_summary:
        summary = detections["summary"]

        # Add summary header
        summary_parts.append("# Summary Analysis")

        # Add object counts
        if "counts" in summary:
            summary_parts.append("\n## Object Counts")
            for label, count in sorted(summary["counts"].items(), key=lambda x: x[1], reverse=True):
                summary_parts.append(f"- {label}: {count} instances")

        # Add unique tracked objects for videos
        if "unique_tracked_objects" in summary:
            summary_parts.append(f"\n## Tracking Information")
            summary_parts.append(f"- Total unique objects tracked: {summary['unique_tracked_objects']}")

            if "unique_by_type" in summary:
                summary_parts.append("- Unique objects by type:")
                for label, count in sorted(summary["unique_by_type"].items(), key=lambda x: x[1], reverse=True):
                    summary_parts.append(f"  - {label}: {count} unique instances")

        # Add speed data if available
        if "speed" in summary:
            summary_parts.append("\n## Speed Information")
            # Format depends on the exact structure returned by YOLO11
            if isinstance(summary["speed"], dict):
                for obj_type, speed_data in summary["speed"].items():
                    if isinstance(speed_data, (int, float)):
                        summary_parts.append(f"- Average speed of {obj_type}: {speed_data:.1f} units/s")
                    elif isinstance(speed_data, dict) and "avg_speed" in speed_data:
                        summary_parts.append(f"- Average speed of {obj_type}: {speed_data['avg_speed']:.1f} units/s")

        # Add counting data if available
        if "counting" in summary:
            summary_parts.append("\n## Object Counting")
            counting_data = summary["counting"]
            if isinstance(counting_data, dict):
                for count_type, count_value in counting_data.items():
                    summary_parts.append(f"- {count_type}: {count_value}")

        # Add frame info for videos
        if "total_frames_analyzed" in summary:
            summary_parts.append(f"\n## Video Analysis")
            summary_parts.append(f"- Total frames analyzed: {summary['total_frames_analyzed']}")

        # Add a separator
        summary_parts.append("\n" + "-" * 40 + "\n")

    # Count objects by type from basic detections
    object_counts = {}
    object_details = []

    # Process the detection frames (skip metadata keys)
    frame_keys = [k for k in detections.keys() if k.isdigit()]

    if frame_keys:
        # For images (single frame) or videos (multiple frames)
        is_video = len(frame_keys) > 1

        if is_video:
            summary_parts.append(f"Analyzed {len(frame_keys)} frames of video")

        # Collect statistics and object details
        for frame_key in frame_keys:
            frame_prefix = f"Frame {frame_key}: " if is_video else ""
            frame_detections = detections[frame_key]

            for det in frame_detections:
                if not isinstance(det, dict):
                    continue

                label = det.get("label", "unknown")
                object_id = det.get("object_id", "unknown")

                # Update counts
                if label not in object_counts:
                    object_counts[label] = 0
                object_counts[label] += 1

                # Create detailed object entry
                obj_details = f"{frame_prefix}[{object_id}] {label}"

                # Add confidence
                if "confidence" in det:
                    obj_details += f" (confidence: {det['confidence']:.2f})"

                # Add attributes
                if "attributes" in det and det["attributes"]:
                    attrs = [f"{k}:{v}" for k, v in det["attributes"].items()]
                    obj_details += f" - {', '.join(attrs)}"

                # Add position if available
                if "position_area" in det:
                    obj_details += f" - position: {det['position_area']}"

                # Add activities for videos
                if "activities" in det and det["activities"]:
                    obj_details += f" - activities: {', '.join(det['activities'])}"

                # Add key relationships (limit to first 2 for readability)
                if "relationships" in det and det["relationships"]:
                    for rel in det["relationships"][:2]:
                        if "object" in rel and "relations" in rel and rel["relations"]:
                            obj_details += f" - {rel['relations'][0]} {rel['object']}"

                object_details.append(obj_details)

        # Add object counts to summary if not already added from summary data
        if not has_summary and object_counts:
            summary_parts.append("Basic Object Counts:")
            for label, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True):
                summary_parts.append(f"- {label}: {count} instances")

        # Add detailed object list (limited to avoid overwhelming the LLM)
        if object_details:
            summary_parts.append("\nDetailed Object List:")
            # Limit to 50 objects to keep context manageable
            for detail in object_details[:50]:
                summary_parts.append(f"- {detail}")

            if len(object_details) > 50:
                summary_parts.append(f"... and {len(object_details) - 50} more objects")

        # Check for target objects specified in the query
        if query_params.get("target_objects"):
            target_objects = query_params["target_objects"]
            summary_parts.append(
                f"\nTarget objects specified in query: {', '.join(target_objects)}"
            )
    else:
        # No detections available
        summary_parts.append("No detections available in the provided media")

    return "\n".join(summary_parts)


def extract_object_ids(highlight_text: str) -> List[str]:
    """
    Extract object IDs from highlight text, handling various formats.

    Args:
        highlight_text: Text containing object IDs to highlight

    Returns:
        List of object IDs
    """
    object_ids = []

    # Clean text
    cleaned_text = highlight_text.strip()

    # Try to parse as JSON array first
    if cleaned_text.startswith("[") and cleaned_text.endswith("]"):
        try:
            parsed_ids = json.loads(cleaned_text)
            if isinstance(parsed_ids, list):
                for item in parsed_ids:
                    if isinstance(item, str):
                        object_ids.append(item)
                    elif isinstance(item, dict) and "object_id" in item:
                        object_ids.append(item["object_id"])
                return object_ids
        except json.JSONDecodeError:
            pass

    # Regular expression to find object IDs (obj_X format)
    obj_pattern = r"obj_\d+"
    found_ids = re.findall(obj_pattern, cleaned_text)
    if found_ids:
        return found_ids

    # Look for any bracketed IDs
    bracket_pattern = r"\[([^\]]+)\]"
    bracket_matches = re.findall(bracket_pattern, cleaned_text)
    for match in bracket_matches:
        if match.startswith("obj_"):
            object_ids.append(match)

    # If still no IDs found, split by lines and look for obj_ prefix
    if not object_ids:
        lines = [line.strip() for line in cleaned_text.split("\n")]
        for line in lines:
            if line.startswith("obj_") or "obj_" in line:
                # Extract the obj_X part
                parts = line.split()
                for part in parts:
                    if part.startswith("obj_"):
                        # Remove any punctuation
                        clean_part = re.sub(r"[^\w_]", "", part)
                        object_ids.append(clean_part)

    return object_ids


def get_objects_by_ids(
    object_ids: List[str], detection_map: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Get the actual detection objects by their IDs.

    Args:
        object_ids: List of object IDs to retrieve
        detection_map: Map of object_id to detection information

    Returns:
        List of detection objects with frame reference
    """
    result = []

    for obj_id in object_ids:
        if obj_id in detection_map:
            object_info = detection_map[obj_id]
            # Create a reference that includes both the detection and its frame
            result.append(
                {
                    "frame_key": object_info["frame_key"],
                    "detection": object_info["detection"],
                }
            )

    return result


def parse_explanation_response(
    response_content: str, detection_map: Dict[str, Dict[str, Any]]
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Parse the LLM response to extract explanation and highlighted objects.
    The explanation section will be cleaned to remove the highlighting instructions.

    Args:
        response_content: LLM response content
        detection_map: Map of object_id to detection information

    Returns:
        Tuple of (explanation_text, highlight_objects)
    """
    explanation_text = ""
    highlight_objects = []

    # Extract explanation and highlight sections
    parts = response_content.split("HIGHLIGHT_OBJECTS:")

    if len(parts) > 1:
        explanation_part = parts[0].strip()
        highlight_part = parts[1].strip()

        # Extract the explanation text (remove the EXPLANATION: prefix if present)
        if "EXPLANATION:" in explanation_part:
            explanation_text = explanation_part.split("EXPLANATION:", 1)[1].strip()
        else:
            explanation_text = explanation_part

        # Extract object IDs and get corresponding objects
        object_ids = extract_object_ids(highlight_part)
        highlight_objects = get_objects_by_ids(object_ids, detection_map)
    else:
        # If no highlight section found, use the whole response as explanation
        # but still try to clean it if it has the EXPLANATION: prefix
        if "EXPLANATION:" in response_content:
            explanation_text = response_content.split("EXPLANATION:", 1)[1].strip()
        else:
            explanation_text = response_content

    return explanation_text, highlight_objects