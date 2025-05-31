"""
Enhanced utility functions for LLM processing with better YOLO11 metrics handling
"""

import json
import re
from typing import Any, Dict, List, Tuple


def process_image_detections_and_format_summary(
        detections: Dict[str, Any], query_params: Dict[str, Any]
) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """
    Process image detections in the new format and create both summary and detection map.

    Args:
        detections: Detection results in format {"objects": [...], "summary": {...}}
        query_params: Parsed query parameters

    Returns:
        Tuple of (formatted_summary, detection_map)
    """
    detection_map = {}

    # Extract objects list
    objects = detections.get("objects", [])
    summary_info = detections.get("summary", {})

    if not objects:
        summary_text = "No objects detected in the image"
        return summary_text, {}

    # Build detection map from objects
    for obj in objects:
        obj_id = obj.get("id", f"obj_{len(detection_map)}")
        detection_map[obj_id] = {
            "frame_key": "0",  # Images are always frame 0
            "detection": obj  # Store the object data
        }

    # Create formatted summary
    summary_parts = []

    # Header
    summary_parts.append("# Image Analysis Summary")

    # Basic info from summary
    image_info = summary_info.get("image_info", {})
    if image_info:
        summary_parts.append(f"Image Resolution: {image_info.get('resolution', 'unknown')}")
        summary_parts.append(f"Total Objects: {image_info.get('total_objects', len(objects))}")
        summary_parts.append(f"Unique Object Types: {image_info.get('unique_types', 'unknown')}")

    # Object distribution
    object_dist = summary_info.get("object_distribution", {})
    by_type = object_dist.get("by_type", {})
    if by_type:
        summary_parts.append("\n## Object Counts by Type")
        for obj_type, count in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
            summary_parts.append(f"- {obj_type}: {count} instances")

    # Notable patterns
    patterns = summary_info.get("notable_patterns", [])
    if patterns:
        summary_parts.append("\n## Notable Patterns")
        for pattern in patterns:
            summary_parts.append(f"- {pattern}")

    # Detailed object list with IDs for highlighting
    summary_parts.append("\n## Detailed Object List")
    for obj in objects:
        obj_id = obj.get("id", "unknown")
        obj_type = obj.get("type", "unknown")
        confidence = obj.get("confidence", 0)

        # Create detailed object entry
        obj_details = f"[{obj_id}] {obj_type}"

        # Add confidence
        obj_details += f" (confidence: {confidence:.2f})"

        # Add attributes if available
        attributes = []
        for key, value in obj.items():
            if key in ["size", "color", "position"] and value:
                attributes.append(f"{key}:{value}")

        if attributes:
            obj_details += f" - {', '.join(attributes)}"

        summary_parts.append(f"- {obj_details}")

    # Add query context
    summary_parts.append(f"\n## Query Context")
    summary_parts.append(f"Task Type: {query_params.get('task_type', 'identification')}")

    if query_params.get("target_objects"):
        summary_parts.append(f"Target Objects: {', '.join(query_params['target_objects'])}")

    if query_params.get("attributes"):
        attr_list = []
        for attr in query_params['attributes']:
            if isinstance(attr, dict):
                attr_str = f"{attr.get('attribute', 'unknown')}:{attr.get('value', 'unknown')}"
                attr_list.append(attr_str)
        if attr_list:
            summary_parts.append(f"Requested Attributes: {', '.join(attr_list)}")

    return "\n".join(summary_parts), detection_map

def format_video_summary(
        compressed_results: Dict[str, Any], parsed_query: Dict[str, Any]
) -> str:
    """
    Format compressed video results for LLM explanation.

    Args:
        compressed_results: Compressed video analysis results
        parsed_query: Parsed query parameters

    Returns:
        Formatted summary string for LLM
    """
    summary_parts = []

    # Extract the summary section
    summary = compressed_results.get("summary", {})

    # Video basic info
    video_info = summary.get("video_info", {})
    if video_info:
        summary_parts.append("# Video Analysis Summary")
        summary_parts.append(f"Duration: {video_info.get('duration_seconds', 0)} seconds")
        summary_parts.append(f"Resolution: {video_info.get('resolution', 'unknown')}")
        summary_parts.append(f"Activity Level: {video_info.get('activity_level', 'unknown')}")

        if video_info.get('primary_objects'):
            summary_parts.append(f"Primary Objects: {', '.join(video_info['primary_objects'])}")

        summary_parts.append(f"Total Objects Tracked: {video_info.get('total_objects_tracked', 0)}")

    # Spatial distribution
    spatial_dist = summary.get("spatial_distribution", {})
    if spatial_dist:
        summary_parts.append("\n## Spatial Distribution")
        for obj_type, info in spatial_dist.items():
            summary_parts.append(f"- {obj_type}: {info.get('count', 0)} instances")

            # Add position info if available
            if info.get('common_positions'):
                positions = ', '.join([f"{pos}({count})" for pos, count in info['common_positions'].items()])
                summary_parts.append(f"  Common positions: {positions}")

            # Add size distribution if available
            if info.get('size_distribution'):
                sizes = info['size_distribution']
                size_info = ', '.join([f"{size}:{count}" for size, count in sizes.items() if count > 0])
                if size_info:
                    summary_parts.append(f"  Size distribution: {size_info}")

    # Temporal analysis
    temporal = summary.get("temporal_analysis", {})
    if temporal:
        summary_parts.append("\n## Temporal Analysis")
        if temporal.get('peak_activity_time') is not None:
            summary_parts.append(f"Peak Activity Time: {temporal['peak_activity_time']} seconds")
        if temporal.get('peak_activity_count'):
            summary_parts.append(f"Peak Activity Count: {temporal['peak_activity_count']} objects")
        if temporal.get('avg_objects_per_window'):
            summary_parts.append(f"Average Objects per Window: {temporal['avg_objects_per_window']:.1f}")

        # Add timeline if available (limit to key periods)
        if temporal.get('activity_timeline'):
            summary_parts.append("Activity Timeline:")
            for period in temporal['activity_timeline'][:3]:  # Limit to 3 periods
                time = period.get('time', 0)
                total = period.get('total_objects', 0)
                types = period.get('object_types', {})
                type_str = ', '.join([f"{obj}:{count}" for obj, count in types.items()])
                summary_parts.append(f"  {time}s: {total} objects ({type_str})")

    # Advanced statistics (counting and speed)
    advanced_stats = summary.get("advanced_stats", {})
    if advanced_stats:
        summary_parts.append("\n## Advanced Statistics")

        # Counting data
        if "counting" in advanced_stats:
            counting = advanced_stats["counting"]
            summary_parts.append("### Object Counting")
            if counting.get('in_count') is not None and counting.get('out_count') is not None:
                summary_parts.append(f"Objects Entered: {counting['in_count']}")
                summary_parts.append(f"Objects Exited: {counting['out_count']}")
                summary_parts.append(f"Net Flow: {counting['in_count'] - counting['out_count']}")

            # Class-wise counting
            if counting.get('class_counts'):
                summary_parts.append("By Object Type:")
                for obj_type, counts in counting['class_counts'].items():
                    summary_parts.append(f"  {obj_type}: in={counts.get('in', 0)}, out={counts.get('out', 0)}")

        # Speed data
        if "speed" in advanced_stats:
            speed = advanced_stats["speed"]
            summary_parts.append("### Speed Analysis")
            if speed.get('avg_speed'):
                summary_parts.append(f"Average Speed: {speed['avg_speed']:.1f} km/h")
            if speed.get('total_tracks'):
                summary_parts.append(f"Objects with Speed Data: {speed['total_tracks']}")

            # Class-wise speeds
            if speed.get('class_speeds'):
                summary_parts.append("Speed by Object Type:")
                for obj_type, speed_info in speed['class_speeds'].items():
                    avg_speed = speed_info.get('avg_speed', 0)
                    sample_count = speed_info.get('sample_count', 0)
                    summary_parts.append(f"  {obj_type}: {avg_speed:.1f} km/h (samples: {sample_count})")

    # Add query context
    if parsed_query.get("target_objects"):
        summary_parts.append(f"\nQuery Target Objects: {', '.join(parsed_query['target_objects'])}")

    summary_parts.append(f"Query Task Type: {parsed_query.get('task_type', 'unknown')}")

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


def format_enhanced_video_summary(
        video_results: Dict[str, Any], parsed_query: Dict[str, Any]
) -> str:
    """
    Format enhanced video results for LLM explanation with focus on YOLO11 metrics.

    Args:
        video_results: Enhanced video analysis results
        parsed_query: Parsed query parameters

    Returns:
        Formatted summary string optimized for LLM processing
    """
    summary_parts = []
    summary = video_results.get("summary", {})

    # Video Overview
    video_info = summary.get("video_info", {})
    if video_info:
        summary_parts.append("# Enhanced Video Analysis Summary")
        summary_parts.append(f"Duration: {video_info.get('duration_seconds', 0)} seconds")
        summary_parts.append(f"Resolution: {video_info.get('resolution', 'unknown')}")
        summary_parts.append(f"Activity Level: {video_info.get('activity_level', 'unknown')}")

        if video_info.get('primary_objects'):
            summary_parts.append(f"Primary Objects: {', '.join(video_info['primary_objects'])}")

    # YOLO11 Counting Analysis (PRIMARY SOURCE)
    counting = summary.get("counting_analysis", {})
    if counting:
        summary_parts.append("\n## YOLO11 Object Counting Results")
        summary_parts.append(f"Objects Entered Zone: {counting.get('objects_entered', 0)}")
        summary_parts.append(f"Objects Exited Zone: {counting.get('objects_exited', 0)}")
        summary_parts.append(
            f"Net Flow: {counting.get('net_flow', 0)} ({'inward' if counting.get('net_flow', 0) > 0 else 'outward'})")
        summary_parts.append(f"Total Boundary Crossings: {counting.get('total_crossings', 0)}")

        # Class-wise counting
        if counting.get('by_object_type'):
            summary_parts.append("\nCounting by Object Type:")
            for obj_type, counts in counting['by_object_type'].items():
                summary_parts.append(
                    f"  {obj_type}: {counts.get('entered', 0)} in, {counts.get('exited', 0)} out (net: {counts.get('net_flow', 0)})")

        if counting.get('most_active_type'):
            summary_parts.append(f"Most Active Object Type: {counting['most_active_type']}")

    # YOLO11 Speed Analysis
    speed = summary.get("speed_analysis", {})
    if speed and speed.get("speed_available"):
        summary_parts.append("\n## YOLO11 Speed Analysis Results")
        summary_parts.append(f"Objects with Speed Data: {speed.get('objects_with_speed', 0)}")

        if speed.get('average_speed_kmh'):
            summary_parts.append(
                f"Average Speed: {speed['average_speed_kmh']} km/h ({speed.get('speed_category', 'unknown')} pace)")

        # Class-wise speeds
        if speed.get('by_object_type'):
            summary_parts.append("\nSpeed by Object Type:")
            for obj_type, speed_info in speed['by_object_type'].items():
                avg_speed = speed_info.get('average_speed', 0)
                category = speed_info.get('speed_category', 'unknown')
                summary_parts.append(f"  {obj_type}: {avg_speed} km/h ({category})")

        if speed.get('fastest_type'):
            summary_parts.append(f"Fastest Object Type: {speed['fastest_type']}")

    # Temporal Analysis (Movement Patterns)
    temporal = summary.get("temporal_relationships", {})
    if temporal:
        summary_parts.append("\n## Temporal Movement Analysis")

        movement = temporal.get("movement_patterns", {})
        if movement:
            summary_parts.append(f"Stationary Objects: {movement.get('stationary_count', 0)}")
            summary_parts.append(f"Moving Objects: {movement.get('moving_count', 0)}")
            summary_parts.append(f"Fast Moving Objects: {movement.get('fast_moving_count', 0)}")

            directions = movement.get("primary_directions", {})
            if directions:
                summary_parts.append("Primary Movement Directions:")
                for direction, count in directions.items():
                    summary_parts.append(f"  {direction}: {len(count) if isinstance(count, list) else count} objects")

        # Object interactions
        if temporal.get('co_occurrence_events', 0) > 0:
            summary_parts.append(f"\nObject Co-occurrence Events: {temporal['co_occurrence_events']}")

            interactions = temporal.get('interaction_summary', [])
            if interactions:
                summary_parts.append("Key Interactions:")
                for interaction in interactions[:3]:  # Top 3
                    obj1 = interaction.get('object1', 'unknown')
                    obj2 = interaction.get('object2', 'unknown')
                    relationship = interaction.get('relationship', 'unknown')
                    summary_parts.append(f"  {obj1} and {obj2}: {relationship}")

    # Spatial Relationships
    spatial = summary.get("spatial_relationships", {})
    if spatial:
        summary_parts.append("\n## Spatial Relationship Analysis")

        common_relations = spatial.get("common_relations", {})
        if common_relations:
            summary_parts.append("Most Common Spatial Relations:")
            for relation, count in list(common_relations.items())[:3]:
                summary_parts.append(f"  {relation}: {count} occurrences")

        frequent_pairs = spatial.get("frequent_pairs", {})
        if frequent_pairs:
            summary_parts.append("Frequently Co-occurring Object Pairs:")
            for pair, count in list(frequent_pairs.items())[:3]:
                summary_parts.append(f"  {pair}: {count} times together")

    # Object Analysis with Attributes
    object_analysis = summary.get("object_analysis", {})
    if object_analysis:
        summary_parts.append("\n## Object Characteristics Analysis")

        characteristics = object_analysis.get("object_characteristics", {})
        for obj_type, chars in list(characteristics.items())[:5]:  # Top 5 object types
            summary_parts.append(f"\n{obj_type}:")
            summary_parts.append(f"  Total Instances: {chars.get('total_instances', 0)}")
            summary_parts.append(f"  Movement Behavior: {chars.get('movement_behavior', 'unknown')}")

            # Common attributes
            common_attrs = chars.get('common_attributes', {})
            if common_attrs:
                attr_list = [f"{attr}({count})" for attr, count in list(common_attrs.items())[:2]]
                summary_parts.append(f"  Common Attributes: {', '.join(attr_list)}")

    # Primary Insights (Key Takeaways)
    insights = summary.get("primary_insights", [])
    if insights:
        summary_parts.append("\n## Key Insights")
        for i, insight in enumerate(insights, 1):
            summary_parts.append(f"{i}. {insight}")

    # Query Context
    summary_parts.append(f"\n## Query Context")
    summary_parts.append(f"Analysis Type: {parsed_query.get('task_type', 'identification')}")

    if parsed_query.get("target_objects"):
        summary_parts.append(f"Target Objects: {', '.join(parsed_query['target_objects'])}")

    if parsed_query.get("count_objects"):
        summary_parts.append("Counting Analysis: Requested (results from YOLO11 above)")

    # Processing metadata
    processing_info = video_results.get("processing_info", {})
    if processing_info:
        frames_analyzed = processing_info.get("frames_analyzed", 0)
        total_frames = processing_info.get("total_frames", 0)
        if total_frames > 0:
            analysis_coverage = (frames_analyzed / total_frames) * 100
            summary_parts.append(
                f"Analysis Coverage: {frames_analyzed}/{total_frames} frames ({analysis_coverage:.1f}%)")

    return "\n".join(summary_parts)


def create_video_detection_map_for_highlighting(
        video_results: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """
    Create detection map for video highlighting (simplified for video).
    Since videos don't support per-object highlighting like images,
    this creates a basic structure for consistency.

    Args:
        video_results: Enhanced video results

    Returns:
        Simple detection map (mostly empty for videos)
    """
    # For videos, we don't do per-object highlighting like images
    # But we maintain the structure for consistency

    frame_detections = video_results.get("frame_detections", {})
    detection_map = {}

    # Create a basic map from the most recent frame for consistency
    if frame_detections:
        latest_frame_key = max(frame_detections.keys(), key=int)
        latest_detections = frame_detections[latest_frame_key]

        for i, det in enumerate(latest_detections[:5]):  # Limit to 5 for performance
            obj_id = det.get("object_id", f"video_obj_{i}")
            detection_map[obj_id] = {
                "frame_key": latest_frame_key,
                "detection": det
            }

    return detection_map