"""
YOLO result formatting and analysis module
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Tuple


class YOLOResultFormatter:
    """Handles formatting and analysis of YOLO results"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def create_enhanced_video_results(
        self,
        frame_detections: Dict[str, List[Dict[str, Any]]],
        temporal_tracker: Any,
        spatial_analyzer: Any,
        counter_results: Any,
        speed_results: Any,
        video_props: Tuple,
        query_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Create comprehensive video results with temporal, spatial,
        and YOLO11 analysis
        """
        width, height, fps, total_frames = video_props
        duration = total_frames / fps

        # Get temporal analysis
        movement_patterns = temporal_tracker.get_movement_patterns()
        temporal_relationships = temporal_tracker.get_temporal_relationships()

        # Get spatial analysis
        spatial_summary = spatial_analyzer.get_relationship_summary()

        # Parse YOLO11 metrics using enhanced parser
        yolo11_metrics = self._parse_enhanced_yolo11_results(
            counter_results, speed_results
        )

        # Create object summary with temporal and spatial context
        object_analysis = self._create_comprehensive_object_analysis(
            temporal_tracker, movement_patterns, spatial_summary
        )

        # Determine primary insights based on query type
        primary_insights = self._extract_primary_insights(
            query_params, yolo11_metrics, movement_patterns, spatial_summary
        )

        return {
            # For LLM processing - compressed and focused
            "summary": {
                "video_info": {
                    "duration_seconds": round(duration, 1),
                    "resolution": f"{width}x{height}",
                    "fps": round(fps, 1),
                    "activity_level": self._assess_activity_level(
                        movement_patterns, duration
                    ),
                    "primary_objects": object_analysis.get("most_common_types", [])[:3],
                },
                # YOLO11 metrics (counting and speed) - PRIMARY SOURCE OF TRUTH
                "counting_analysis": yolo11_metrics.get("counting", {}),
                "speed_analysis": yolo11_metrics.get("speed", {}),
                # Temporal analysis
                "temporal_relationships": {
                    "movement_patterns": {
                        "stationary_count": len(
                            movement_patterns.get("stationary_objects", [])
                        ),
                        "moving_count": len(
                            movement_patterns.get("moving_objects", [])
                        ),
                        "fast_moving_count": len(
                            movement_patterns.get("fast_moving_objects", [])
                        ),
                        "primary_directions": dict(
                            list(
                                movement_patterns.get(
                                    "directional_movements", {}
                                ).items()
                            )[:3]
                        ),
                    },
                    "co_occurrence_events": len(temporal_relationships),
                    "interaction_summary": temporal_relationships[:5],
                },
                # Spatial analysis
                "spatial_relationships": {
                    "common_relations": spatial_summary.get(
                        "most_common_relations", {}
                    ),
                    "frequent_pairs": spatial_summary.get("frequent_object_pairs", {}),
                    "spatial_patterns": spatial_summary.get("spatial_patterns", {}),
                },
                # Object analysis with attributes
                "object_analysis": object_analysis,
                # Query-specific insights
                "primary_insights": primary_insights,
            },
            # For visualization - detailed frame data
            "frame_detections": frame_detections,
            # Metadata
            "processing_info": {
                "frames_analyzed": len(frame_detections),
                "total_frames": total_frames,
                "analysis_type": query_params.get("task_type", "identification"),
                "yolo11_enabled": bool(counter_results or speed_results),
            },
        }

    def _parse_enhanced_yolo11_results(
        self, counter_results: Any, speed_results: Any
    ) -> Dict[str, Any]:
        """Enhanced parsing of YOLO11 results with better structure"""
        metrics = {}

        # Enhanced counting analysis
        if counter_results and hasattr(counter_results, "in_count"):
            counting_data = {
                "objects_entered": counter_results.in_count,
                "objects_exited": counter_results.out_count,
                "net_flow": counter_results.in_count - counter_results.out_count,
                "total_crossings": counter_results.in_count + counter_results.out_count,
                "flow_direction": (
                    "inward"
                    if counter_results.in_count > counter_results.out_count
                    else "outward"
                ),
            }

            # Class-wise analysis
            if hasattr(counter_results, "classwise_count"):
                class_analysis = {}
                for obj_type, directions in counter_results.classwise_count.items():
                    if directions["IN"] > 0 or directions["OUT"] > 0:
                        class_analysis[obj_type] = {
                            "entered": directions["IN"],
                            "exited": directions["OUT"],
                            "net_flow": directions["IN"] - directions["OUT"],
                            "dominance": (
                                "entering"
                                if directions["IN"] > directions["OUT"]
                                else "exiting"
                            ),
                        }

                counting_data["by_object_type"] = class_analysis
                counting_data["most_active_type"] = (
                    max(
                        class_analysis.items(),
                        key=lambda x: x[1]["entered"] + x[1]["exited"],
                    )[0]
                    if class_analysis
                    else None
                )

            metrics["counting"] = counting_data

        # Enhanced speed analysis
        if speed_results and hasattr(speed_results, "total_tracks"):
            speed_data = {
                "objects_with_speed": speed_results.total_tracks,
                "speed_available": speed_results.total_tracks > 0,
            }

            if hasattr(speed_results, "avg_speed") and speed_results.avg_speed:
                speed_data["average_speed_kmh"] = round(speed_results.avg_speed, 1)
                speed_data["speed_category"] = self._categorize_speed(
                    speed_results.avg_speed
                )

            # Class-wise speed analysis
            if hasattr(speed_results, "class_speeds"):
                class_speeds = {}
                for obj_type, speeds in speed_results.class_speeds.items():
                    if speeds:
                        avg_speed = sum(speeds) / len(speeds)
                        class_speeds[obj_type] = {
                            "average_speed": round(avg_speed, 1),
                            "sample_count": len(speeds),
                            "speed_range": (
                                {
                                    "min": round(min(speeds), 1),
                                    "max": round(max(speeds), 1),
                                }
                                if len(speeds) > 1
                                else None
                            ),
                            "speed_category": self._categorize_speed(avg_speed),
                        }

                speed_data["by_object_type"] = class_speeds

                # Find fastest object type
                if class_speeds:
                    speed_data["fastest_type"] = max(
                        class_speeds.items(), key=lambda x: x[1]["average_speed"]
                    )[0]

            metrics["speed"] = speed_data

        return metrics

    def _create_comprehensive_object_analysis(
        self,
        temporal_tracker: Any,
        movement_patterns: Dict[str, Any],
        spatial_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create comprehensive object analysis from temporal and spatial data"""
        object_characteristics = defaultdict(
            lambda: {
                "total_instances": 0,
                "movement_behavior": "unknown",
                "common_attributes": defaultdict(int),
                "spatial_preferences": defaultdict(int),
            }
        )

        # Analyze from temporal tracker
        for obj_key, history in temporal_tracker.object_histories.items():
            obj_type = obj_key.split("_")[0]  # Extract object type
            characteristics = object_characteristics[obj_type]
            characteristics["total_instances"] += 1

            # Analyze movement behavior
            if len(history["positions"]) >= 3:
                movement_distance = temporal_tracker._calculate_total_movement(
                    list(history["positions"])
                )
                if movement_distance < 50:
                    characteristics["movement_behavior"] = "stationary"
                elif movement_distance > 200:
                    characteristics["movement_behavior"] = "highly_mobile"
                else:
                    characteristics["movement_behavior"] = "moderately_mobile"

            # Collect attributes from latest frames
            for attrs in list(history["attributes"])[-5:]:  # Last 5 frames
                for attr_name, attr_value in attrs.items():
                    if attr_value and attr_value != "unknown":
                        characteristics["common_attributes"][
                            f"{attr_name}:{attr_value}"
                        ] += 1

        # Convert to regular dict and find most common attributes
        final_analysis = {}
        most_common_types = sorted(
            object_characteristics.items(),
            key=lambda x: x[1]["total_instances"],
            reverse=True,
        )

        for obj_type, chars in most_common_types:
            # Get most common attributes
            top_attributes = dict(
                sorted(
                    chars["common_attributes"].items(), key=lambda x: x[1], reverse=True
                )[:3]
            )

            final_analysis[obj_type] = {
                "total_instances": chars["total_instances"],
                "movement_behavior": chars["movement_behavior"],
                "common_attributes": top_attributes,
            }

        return {
            "object_characteristics": final_analysis,
            "most_common_types": [obj_type for obj_type, _ in most_common_types[:5]],
            "total_unique_objects": len(object_characteristics),
        }

    def _extract_primary_insights(
        self,
        query_params: Dict[str, Any],
        yolo11_metrics: Dict[str, Any],
        movement_patterns: Dict[str, Any],
        spatial_summary: Dict[str, Any],
    ) -> List[str]:
        """Extract key insights based on query type and analysis results"""
        insights = []
        task_type = query_params.get("task_type", "identification")

        # Counting-specific insights (PRIMARY)
        if task_type == "counting" and "counting" in yolo11_metrics:
            counting = yolo11_metrics["counting"]
            insights.append(
                f"YOLO11 counted {counting.get('total_crossings', 0)} "
                f"total object crossings"
            )
            if counting.get("net_flow", 0) != 0:
                flow_type = "net inward" if counting["net_flow"] > 0 else "net outward"
                insights.append(
                    f"Overall flow: {abs(counting['net_flow'])} objects {flow_type}"
                )

            if counting.get("most_active_type"):
                insights.append(
                    f"Most active object type: {counting['most_active_type']}"
                )

        # Speed-specific insights
        if "speed" in yolo11_metrics and yolo11_metrics["speed"].get("speed_available"):
            speed = yolo11_metrics["speed"]
            if speed.get("average_speed_kmh"):
                insights.append(
                    f"Average speed: {speed['average_speed_kmh']} km/h "
                    f"({speed.get('speed_category', 'unknown')} pace)"
                )
            if speed.get("fastest_type"):
                insights.append(f"Fastest object type: {speed['fastest_type']}")

        # Movement pattern insights
        if movement_patterns:
            stationary_count = len(movement_patterns.get("stationary_objects", []))
            moving_count = len(movement_patterns.get("moving_objects", []))

            if stationary_count > moving_count:
                insights.append(
                    f"Scene is mostly static with {stationary_count} stationary objects"
                )
            elif moving_count > 0:
                insights.append(f"Active scene with {moving_count} moving objects")

        # Spatial relationship insights
        if spatial_summary.get("most_common_relations"):
            top_relation = list(spatial_summary["most_common_relations"].keys())[0]
            insights.append(f"Most common spatial relationship: {top_relation}")

        return insights[:4]  # Limit to top 4 insights

    def _categorize_speed(self, speed_kmh: float) -> str:
        """Categorize speed into human-readable categories"""
        if speed_kmh < 5:
            return "very_slow"
        elif speed_kmh < 15:
            return "slow"
        elif speed_kmh < 40:
            return "moderate"
        elif speed_kmh < 80:
            return "fast"
        else:
            return "very_fast"

    def _assess_activity_level(
        self, movement_patterns: Dict[str, Any], duration: float
    ) -> str:
        """Assess overall activity level of the video"""
        total_moving = len(movement_patterns.get("moving_objects", [])) + len(
            movement_patterns.get("fast_moving_objects", [])
        )
        total_stationary = len(movement_patterns.get("stationary_objects", []))

        if total_moving == 0:
            return "static"
        elif total_moving < total_stationary:
            return "low_activity"
        elif total_moving > total_stationary * 2:
            return "high_activity"
        else:
            return "moderate_activity"
