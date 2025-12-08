"""
Enhanced utility functions for LLM processing with YOLO-World + ByteTracker integration
"""

import json
import re
from typing import Any, Counter, Dict, List, Tuple


def process_image_detections_and_format_summary(
    detections: Dict[str, Any], query_params: Dict[str, Any]
) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """
    Process image detections in new format and create both summary and detection map.

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
            "detection": obj,  # Store the object data
        }

    # Create formatted summary
    summary_parts = []

    # Header
    summary_parts.append("# Image Analysis Summary")

    # Basic info from summary
    image_info = summary_info.get("image_info", {})
    if image_info:
        summary_parts.append(
            f"Image Resolution: {image_info.get('resolution', 'unknown')}"
        )
        summary_parts.append(
            f"Total Objects: {image_info.get('total_objects', len(objects))}"
        )
        summary_parts.append(
            f"Unique Object Types: {image_info.get('unique_types', 'unknown')}"
        )

    # Object distribution
    object_dist = summary_info.get("object_distribution", {})
    by_type = object_dist.get("by_type", {})
    if by_type:
        summary_parts.append("\n## Object Counts by Type")
        for obj_type, count in sorted(
            by_type.items(), key=lambda x: x[1], reverse=True
        ):
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
    summary_parts.append("\n## Query Context")
    summary_parts.append(
        f"Task Type: {query_params.get('task_type', 'identification')}"
    )

    if query_params.get("target_objects"):
        summary_parts.append(
            f"Target Objects: {', '.join(query_params['target_objects'])}"
        )

    if query_params.get("attributes"):
        attr_list = []
        for attr in query_params["attributes"]:
            if isinstance(attr, dict):
                attr_str = (
                    f"{attr.get('attribute', 'unknown')}:{attr.get('value', 'unknown')}"
                )
                attr_list.append(attr_str)
        if attr_list:
            summary_parts.append(f"Requested Attributes: {', '.join(attr_list)}")

    return "\n".join(summary_parts), detection_map


def format_video_summary(
    video_results: Dict[str, Any], parsed_query: Dict[str, Any]
) -> str:
    """
    Enhanced format for comprehensive video results with better frame data handling.

    Args:
        video_results: Complete video analysis results
        parsed_query: Parsed query parameters

    Returns:
        Formatted summary string optimized for LLM processing
    """
    summary_parts = []

    # Extract main components
    summary = video_results.get("summary", {})
    frame_detections = video_results.get("frame_detections", {})
    processing_info = video_results.get("processing_info", {})

    # === VIDEO OVERVIEW ===
    video_info = summary.get("video_info", {})
    if video_info:
        summary_parts.append("# COMPREHENSIVE VIDEO ANALYSIS REPORT")
        summary_parts.append(
            f"Duration: {video_info.get('duration_seconds', 0)}s | "
            f"Resolution: {video_info.get('resolution', 'unknown')} "
            f"| FPS: {video_info.get('fps', 0)}"
        )
        summary_parts.append(
            f"Activity Level: {video_info.get('activity_level', 'unknown').upper()}"
        )

        if video_info.get("primary_objects"):
            summary_parts.append(
                f"Primary Objects: {', '.join(video_info['primary_objects'][:5])}"
            )

    # === YOLO-WORLD + BYTETRACKER COUNTING RESULTS (PRIORITY SECTION) ===
    counting = summary.get("counting_analysis", {})
    if counting:
        summary_parts.append("\n## 🎯 YOLO-WORLD OBJECT COUNTING ANALYSIS")
        summary_parts.append(
            f"**BOUNDARY CROSSINGS:** {counting.get('total_crossings', 0)} total events"
        )
        summary_parts.append(
            f"**FLOW DIRECTION:** {counting.get('flow_direction', 'unknown').upper()}"
        )
        n = counting.get("net_flow", 0)
        summary_parts.append(
            f"**NET MOVEMENT:** {n} objects "
            f"({'in' if n > 0 else 'out' if n < 0 else 'bal'})"
        )

        # Detailed breakdown
        summary_parts.append(
            f"- Objects Entered Zone: {counting.get('objects_entered', 0)}"
        )
        summary_parts.append(
            f"- Objects Exited Zone: {counting.get('objects_exited', 0)}"
        )

        # Class-wise analysis
        if counting.get("by_object_type"):
            summary_parts.append("\n**COUNTING BY OBJECT TYPE:**")
            for obj_type, data in counting["by_object_type"].items():
                entered = data.get("entered", 0)
                exited = data.get("exited", 0)
                net = data.get("net_flow", 0)
                dominance = data.get("dominance", "unknown")
                summary_parts.append(
                    f"• {obj_type.upper()}: {entered}↑, {exited}↓ "
                    f"(net: {net:+}, trend: {dominance})"
                )

        if counting.get("most_active_type"):
            summary_parts.append(
                f"\n**MOST ACTIVE OBJECT:** {counting['most_active_type'].upper()}"
            )

    # === SPEED ANALYSIS (if available) ===
    speed = summary.get("speed_analysis", {})
    if speed and speed.get("speed_available"):
        summary_parts.append("\n## 🚀 YOLO-WORLD SPEED ANALYSIS")
        summary_parts.append(
            f"**Objects with Speed Data:** {speed.get('objects_with_speed', 0)}"
        )

        if speed.get("average_speed_kmh"):
            summary_parts.append(
                f"**Average Speed:** {speed['average_speed_kmh']} km/h "
                f"({speed.get('speed_category', 'unknown')} pace)"
            )

        # Class-wise speeds
        if speed.get("by_object_type"):
            summary_parts.append("\n**SPEED BY OBJECT TYPE:**")
            for obj_type, speed_data in speed["by_object_type"].items():
                avg_speed = speed_data.get("average_speed", 0)
                sample_count = speed_data.get("sample_count", 0)
                category = speed_data.get("speed_category", "unknown")
                summary_parts.append(
                    f"  • {obj_type}: {avg_speed} km/h "
                    f"({category}, {sample_count} samples)"
                )

    # === FRAME-BY-FRAME ACTIVITY ANALYSIS ===
    if frame_detections:
        frame_analysis = analyze_frame_activity(frame_detections)
        summary_parts.append("\n## 📊 FRAME-BY-FRAME ACTIVITY ANALYSIS")
        summary_parts.append(
            f"**Frames Analyzed:** {len(frame_detections)} of "
            f"{processing_info.get('total_frames', 0)}"
        )
        summary_parts.append(
            f"**Peak Activity:** Frame {frame_analysis['peak_frame']} with "
            f"{frame_analysis['peak_count']} objects"
        )
        summary_parts.append(
            f"**Average Objects per Frame:** {frame_analysis['avg_objects']:.1f}"
        )

        # Activity timeline (key moments)
        if frame_analysis.get("activity_timeline"):
            summary_parts.append("\n**KEY ACTIVITY MOMENTS:**")
            for moment in frame_analysis["activity_timeline"][:5]:  # Top 5 moments
                frame_num = moment["frame"]
                count = moment["count"]
                types = moment["types"]
                time_sec = (
                    frame_num * video_info.get("fps", 30) / 30
                )  # Approximate time
                summary_parts.append(
                    f"  • Frame {frame_num} (t={time_sec:.1f}s): {count} objects - "
                    f"{types}"
                )

    # === SPATIAL RELATIONSHIPS ===
    spatial = summary.get("spatial_relationships", {})
    if spatial:
        summary_parts.append("\n## 🗺️ SPATIAL RELATIONSHIP ANALYSIS")

        common_relations = spatial.get("common_relations", {})
        if common_relations:
            summary_parts.append("**Most Common Spatial Relations:**")
            for relation, count in list(common_relations.items())[:5]:
                summary_parts.append(
                    f"  • {relation.replace('_', ' ').title()}: {count} occurrences"
                )

        frequent_pairs = spatial.get("frequent_pairs", {})
        if frequent_pairs:
            summary_parts.append("\n**Frequently Co-occurring Object Pairs:**")
            for pair, count in list(frequent_pairs.items())[:5]:
                summary_parts.append(f"  • {pair}: {count} co-occurrences")

        spatial_patterns = spatial.get("spatial_patterns", {})
        if spatial_patterns:
            summary_parts.append("\n**Top Spatial Patterns:**")
            sorted_patterns = sorted(
                spatial_patterns.items(), key=lambda x: x[1], reverse=True
            )
            for pattern, count in sorted_patterns[:5]:
                formatted_pattern = pattern.replace("-", " → ").replace("_", " ")
                summary_parts.append(f"  • {formatted_pattern}: {count} times")

    # === OBJECT CHARACTERISTICS ANALYSIS ===
    object_analysis = summary.get("object_analysis", {})
    if object_analysis:
        summary_parts.append("\n## 🔍 DETAILED OBJECT ANALYSIS")

        characteristics = object_analysis.get("object_characteristics", {})
        most_common = object_analysis.get("most_common_types", [])

        if most_common:
            summary_parts.append(
                f"**Object Type Diversity:** {len(characteristics)} unique types "
                f"detected"
            )
            summary_parts.append(f"**Most Common Types:** {', '.join(most_common[:5])}")

        # Detailed analysis for top 3 object types
        summary_parts.append("\n**TOP OBJECT TYPES DETAILED ANALYSIS:**")
        for obj_type in most_common[:3]:
            if obj_type in characteristics:
                char = characteristics[obj_type]
                summary_parts.append(f"\n  📋 **{obj_type.upper()}:**")
                summary_parts.append(
                    f"    - Total Instances: {char.get('total_instances', 0)}"
                )
                summary_parts.append(
                    f"    - Movement Behavior: "
                    f"{char.get('movement_behavior', 'unknown')}"
                )

                common_attrs = char.get("common_attributes", {})
                if common_attrs:
                    top_attrs = sorted(
                        common_attrs.items(), key=lambda x: x[1], reverse=True
                    )[:3]
                    attrs_str = ", ".join(
                        [
                            f"{attr.split(':')[1] if ':' in attr else attr}({count})"
                            for attr, count in top_attrs
                        ]
                    )
                    summary_parts.append(f"    - Common Attributes: {attrs_str}")

    # === TEMPORAL MOVEMENT ANALYSIS ===
    temporal = summary.get("temporal_relationships", {})
    if temporal:
        summary_parts.append("\n## ⏱️ TEMPORAL MOVEMENT ANALYSIS")

        movement = temporal.get("movement_patterns", {})
        if movement:
            stationary = movement.get("stationary_count", 0)
            moving = movement.get("moving_count", 0)
            fast_moving = movement.get("fast_moving_count", 0)

            summary_parts.append("**Movement Distribution:**")
            summary_parts.append(f"  • Stationary Objects: {stationary}")
            summary_parts.append(f"  • Moving Objects: {moving}")
            summary_parts.append(f"  • Fast Moving Objects: {fast_moving}")

            # Movement directions
            directions = movement.get("primary_directions", {})
            if directions:
                summary_parts.append("\n**Primary Movement Directions:**")
                for direction, objects in directions.items():
                    count = len(objects) if isinstance(objects, list) else objects
                    summary_parts.append(
                        f"  • {direction.replace('_', ' ').title()}: {count} objects"
                    )

        # Co-occurrence events
        co_events = temporal.get("co_occurrence_events", 0)
        if co_events > 0:
            summary_parts.append(f"\n**Object Interaction Events:** {co_events}")

            interactions = temporal.get("interaction_summary", [])
            if interactions:
                summary_parts.append("**Key Interactions:**")
                for interaction in interactions[:3]:
                    obj1 = interaction.get("object1", "unknown")
                    obj2 = interaction.get("object2", "unknown")
                    relationship = interaction.get("relationship", "unknown")
                    summary_parts.append(f"  • {obj1} ↔ {obj2}: {relationship}")

    # === PRIMARY INSIGHTS ===
    insights = summary.get("primary_insights", [])
    if insights:
        summary_parts.append("\n## 💡 KEY INSIGHTS & CONCLUSIONS")
        for i, insight in enumerate(insights, 1):
            summary_parts.append(f"{i}. {insight}")

    # === QUERY CONTEXT ===
    summary_parts.append("\n## 🎯 QUERY ANALYSIS CONTEXT")
    summary_parts.append(
        f"**Query Type:** {parsed_query.get('task_type', 'identification').upper()}"
    )

    if parsed_query.get("target_objects"):
        summary_parts.append(
            f"**Target Objects:** {', '.join(parsed_query['target_objects'])}"
        )

    if parsed_query.get("count_objects"):
        summary_parts.append(
            "**Counting Analysis:** ✅ COMPLETED (YOLO-World results above)"
        )

    if parsed_query.get("attributes"):
        attrs = [
            f"{attr.get('attribute', 'unknown')}:{attr.get('value', 'unknown')}"
            for attr in parsed_query["attributes"]
        ]
        summary_parts.append(f"**Requested Attributes:** {', '.join(attrs)}")

    # === TECHNICAL METADATA ===
    summary_parts.append("\n## ⚙️ PROCESSING METADATA")
    summary_parts.append(
        f"**Analysis Coverage:** {processing_info.get('frames_analyzed', 0)}/"
        f"{processing_info.get('total_frames', 0)} frames"
    )
    summary_parts.append(
        f"**YOLO-World Enhanced:** "
        f"{'✅ YES' if processing_info.get('yolo_world_enabled') else '❌ NO'}"
    )
    summary_parts.append(
        f"**Analysis Type:** {processing_info.get('analysis_type', 'unknown').upper()}"
    )

    return "\n".join(summary_parts)


def analyze_frame_activity(
    frame_detections: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Analyze frame-by-frame activity to extract key insights.

    Args:
        frame_detections: Dictionary mapping frame numbers to detection lists

    Returns:
        Dictionary with frame activity analysis
    """
    if not frame_detections:
        return {
            "peak_frame": 0,
            "peak_count": 0,
            "avg_objects": 0.0,
            "activity_timeline": [],
        }

    frame_counts = {}
    frame_types = {}
    total_objects = 0

    # Analyze each frame
    for frame_key, detections in frame_detections.items():
        frame_num = int(frame_key)
        count = len(detections)
        frame_counts[frame_num] = count
        total_objects += count

        # Track object types in this frame
        types_in_frame = Counter(det.get("label", "unknown") for det in detections)
        frame_types[frame_num] = types_in_frame

    # Find peak activity
    peak_frame = (
        max(frame_counts.items(), key=lambda x: x[1]) if frame_counts else (0, 0)
    )
    avg_objects = total_objects / len(frame_counts) if frame_counts else 0.0

    # Create activity timeline for significant moments
    activity_timeline = []
    sorted_frames = sorted(frame_counts.items(), key=lambda x: x[1], reverse=True)

    for frame_num, count in sorted_frames[:10]:  # Top 10 most active frames
        types = frame_types.get(frame_num, {})
        types_str = ", ".join(
            [f"{obj_type}({cnt})" for obj_type, cnt in types.most_common(3)]
        )

        activity_timeline.append(
            {"frame": frame_num, "count": count, "types": types_str}
        )

    return {
        "peak_frame": peak_frame[0],
        "peak_count": peak_frame[1],
        "avg_objects": avg_objects,
        "activity_timeline": activity_timeline,
        "total_frames_with_activity": len([c for c in frame_counts.values() if c > 0]),
        "frame_counts": frame_counts,  # For potential additional analysis
    }


def create_frame_summary_for_llm(
    frame_detections: Dict[str, List[Dict[str, Any]]], max_frames: int = 20
) -> str:
    """
    Create a concise frame summary for LLM when full frame data is needed.

    Args:
        frame_detections: Dictionary mapping frame numbers to detection lists
        max_frames: Maximum number of frames to include in detail

    Returns:
        Formatted frame summary string
    """
    if not frame_detections:
        return "No frame data available."

    summary_parts = []
    summary_parts.append("## DETAILED FRAME ANALYSIS")

    # Sort frames by activity level (most active first)
    frame_activity = [
        (int(frame_key), len(detections))
        for frame_key, detections in frame_detections.items()
    ]
    frame_activity.sort(key=lambda x: x[1], reverse=True)

    # Include top active frames and some representative frames
    selected_frames = []

    # Top 10 most active frames
    selected_frames.extend([frame_num for frame_num, _ in frame_activity[:10]])

    # Add some evenly distributed frames for temporal coverage
    all_frame_nums = sorted([int(k) for k in frame_detections.keys()])
    if len(all_frame_nums) > 10:
        step = len(all_frame_nums) // min(10, max_frames - 10)
        representative_frames = all_frame_nums[::step]
        selected_frames.extend(representative_frames)

    # Remove duplicates and sort
    selected_frames = sorted(list(set(selected_frames)))[:max_frames]

    summary_parts.append(
        f"Showing {len(selected_frames)}"
        f"most relevant frames out of {len(frame_detections)} total:"
    )

    for frame_num in selected_frames:
        frame_key = str(frame_num)
        if frame_key in frame_detections:
            detections = frame_detections[frame_key]

            if not detections:
                continue

            # Count objects by type
            object_counts = Counter(det.get("label", "unknown") for det in detections)
            objects_summary = ", ".join(
                [
                    f"{obj_type}({count})"
                    for obj_type, count in object_counts.most_common()
                ]
            )

            # Note any special attributes
            special_attrs = []
            for det in detections:
                attrs = det.get("attributes", {})
                if "color" in attrs and attrs["color"] != "unknown":
                    special_attrs.append(
                        f"{attrs['color']} {det.get('label', 'object')}"
                    )

            attr_note = (
                f" | Notable: {', '.join(special_attrs[:3])}" if special_attrs else ""
            )

            summary_parts.append(
                f"Frame {frame_num}: {len(detections)}"
                f"objects ({objects_summary}){attr_note}"
            )

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
        summary_parts.append(
            f"Duration: {video_info.get('duration_seconds', 0)} seconds"
        )
        summary_parts.append(f"Resolution: {video_info.get('resolution', 'unknown')}")
        summary_parts.append(
            f"Activity Level: {video_info.get('activity_level', 'unknown')}"
        )

        if video_info.get("primary_objects"):
            summary_parts.append(
                f"Primary Objects: {', '.join(video_info['primary_objects'])}"
            )

    # YOLO-World Counting Analysis (PRIMARY SOURCE)
    counting = summary.get("counting_analysis", {})
    if counting:
        summary_parts.append("\n## YOLO-World Object Counting Results")
        summary_parts.append(
            f"Objects Entered Zone: {counting.get('objects_entered', 0)}"
        )
        summary_parts.append(
            f"Objects Exited Zone: {counting.get('objects_exited', 0)}"
        )
        summary_parts.append(
            f"Net Flow: {counting.get('net_flow', 0)} "
            f"({'inward' if counting.get('net_flow', 0) > 0 else 'outward'})"
        )
        summary_parts.append(
            f"Total Boundary Crossings: {counting.get('total_crossings', 0)}"
        )

        # Class-wise counting
        if counting.get("by_object_type"):
            summary_parts.append("\nCounting by Object Type:")
            for obj_type, counts in counting["by_object_type"].items():
                summary_parts.append(
                    f"  {obj_type}: {counts.get('entered', 0)} in, "
                    f"{counts.get('exited', 0)} out "
                    f"(net: {counts.get('net_flow', 0)})"
                )

        if counting.get("most_active_type"):
            summary_parts.append(
                f"Most Active Object Type: {counting['most_active_type']}"
            )

    # YOLO-World Speed Analysis
    speed = summary.get("speed_analysis", {})
    if speed and speed.get("speed_available"):
        summary_parts.append("\n## YOLO-World Speed Analysis Results")
        summary_parts.append(
            f"Objects with Speed Data: {speed.get('objects_with_speed', 0)}"
        )

        if speed.get("average_speed_kmh"):
            summary_parts.append(
                f"Average Speed: {speed['average_speed_kmh']} km/h "
                f"({speed.get('speed_category', 'unknown')} pace)"
            )
        # Class-wise speeds
        if speed.get("by_object_type"):
            summary_parts.append("\nSpeed by Object Type:")
            for obj_type, speed_info in speed["by_object_type"].items():
                avg_speed = speed_info.get("average_speed", 0)
                category = speed_info.get("speed_category", "unknown")
                summary_parts.append(f"  {obj_type}: {avg_speed} km/h ({category})")

        if speed.get("fastest_type"):
            summary_parts.append(f"Fastest Object Type: {speed['fastest_type']}")

    # Temporal Analysis (Movement Patterns)
    temporal = summary.get("temporal_relationships", {})
    if temporal:
        summary_parts.append("\n## Temporal Movement Analysis")

        movement = temporal.get("movement_patterns", {})
        if movement:
            summary_parts.append(
                f"Stationary Objects: {movement.get('stationary_count', 0)}"
            )
            summary_parts.append(f"Moving Objects: {movement.get('moving_count', 0)}")
            summary_parts.append(
                f"Fast Moving Objects: {movement.get('fast_moving_count', 0)}"
            )

            directions = movement.get("primary_directions", {})
            if directions:
                summary_parts.append("Primary Movement Directions:")
                for direction, count in directions.items():
                    summary_parts.append(
                        f"  {direction}: "
                        f"{len(count) if isinstance(count, list) else count} objects"
                    )
        # Object interactions
        if temporal.get("co_occurrence_events", 0) > 0:
            summary_parts.append(
                f"\nObject Co-occurrence Events: {temporal['co_occurrence_events']}"
            )

            interactions = temporal.get("interaction_summary", [])
            if interactions:
                summary_parts.append("Key Interactions:")
                for interaction in interactions[:3]:  # Top 3
                    obj1 = interaction.get("object1", "unknown")
                    obj2 = interaction.get("object2", "unknown")
                    relationship = interaction.get("relationship", "unknown")
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
            summary_parts.append(
                f"  Total Instances: {chars.get('total_instances', 0)}"
            )
            summary_parts.append(
                f"  Movement Behavior: {chars.get('movement_behavior', 'unknown')}"
            )

            # Common attributes
            common_attrs = chars.get("common_attributes", {})
            if common_attrs:
                attr_list = [
                    f"{attr}({count})" for attr, count in list(common_attrs.items())[:2]
                ]
                summary_parts.append(f"  Common Attributes: {', '.join(attr_list)}")

    # Primary Insights (Key Takeaways)
    insights = summary.get("primary_insights", [])
    if insights:
        summary_parts.append("\n## Key Insights")
        for i, insight in enumerate(insights, 1):
            summary_parts.append(f"{i}. {insight}")

    # Query Context
    summary_parts.append("\n## Query Context")
    summary_parts.append(
        f"Analysis Type: {parsed_query.get('task_type', 'identification')}"
    )

    if parsed_query.get("target_objects"):
        summary_parts.append(
            f"Target Objects: {', '.join(parsed_query['target_objects'])}"
        )

    if parsed_query.get("count_objects"):
        summary_parts.append("Counting Analysis: Requested (results from YOLO-World above)")

    # Processing metadata
    processing_info = video_results.get("processing_info", {})
    if processing_info:
        frames_analyzed = processing_info.get("frames_analyzed", 0)
        total_frames = processing_info.get("total_frames", 0)
        if total_frames > 0:
            analysis_coverage = (frames_analyzed / total_frames) * 100
            summary_parts.append(
                f"Analysis Coverage: {frames_analyzed}/{total_frames} frames "
                f"({analysis_coverage:.1f}%)"
            )

    return "\n".join(summary_parts)


def create_video_detection_map_for_highlighting(
    video_results: Dict[str, Any],
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
            detection_map[obj_id] = {"frame_key": latest_frame_key, "detection": det}

    return detection_map
