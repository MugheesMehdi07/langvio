"""
Flask application for Langvio - process images and videos with natural language
Updated to handle the new Langvio result format
"""

import logging
import os
import time
import uuid
import subprocess
from pathlib import Path
from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename
# for video re-encoding import imageio_ffmpeg
import imageio_ffmpeg
os.environ["PATH"] += os.pathsep + os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
import imageio_ffmpeg
print(imageio_ffmpeg.get_ffmpeg_exe())

# Import langvio
from langvio import create_pipeline
from langvio.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configure app
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev_key_change_in_production")

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "mp4", "mov", "avi", "webm"}

# Create required directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(os.path.join(app.root_path, "static", "results"), exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULTS_FOLDER"] = RESULTS_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB max-limit


# Initialize Langvio pipeline once
def create_langvio_pipeline():
    try:
        # Create config with good defaults
        config = Config()

        # Higher confidence for better visualization
        # Automatically set confidence for any YOLO-based model
        for model_key in config.config["vision"]["models"]:
            if model_key.startswith(("yolo_world", "yoloe", "yolo")):
                config.config["vision"]["models"][model_key]["confidence"] = 0.5
                # Improve tracking stability to reduce flickering
                # Increase track_buffer to handle occlusions better
                if "track_buffer" in config.config["vision"]["models"][model_key]:
                    config.config["vision"]["models"][model_key]["track_buffer"] = max(
                        config.config["vision"]["models"][model_key].get("track_buffer", 70),
                        90  # Increase buffer to reduce track ID switches
                    )
                # Slightly increase match_thresh for more stable matching
                if "match_thresh" in config.config["vision"]["models"][model_key]:
                    config.config["vision"]["models"][model_key]["match_thresh"] = max(
                        config.config["vision"]["models"][model_key].get("match_thresh", 0.6),
                        0.7  # Higher threshold = more conservative matching = less flickering
                    )


        # Set visualization options
        config.config["media"]["output_dir"] = app.config["RESULTS_FOLDER"]
        config.config["media"]["visualization"] = {
            "box_color": [0, 120, 255],  # Orange boxes
            "text_color": [255, 255, 255],  # White text
            "line_thickness": 2,  # Line thickness
            "show_attributes": True,  # Show attributes
            "show_confidence": False,  # Hide confidence to reduce clutter
        }
        
        openai_models = ["gpt-4", "gpt-3.5", "gpt-4o-mini", "gpt-4.1-mini"]
        selected_llm = None
        
        for llm_name in openai_models:
            try:
                # Try to create pipeline with this LLM
                test_pipeline = create_pipeline(llm_name=llm_name)
                # If successful, use this model
                selected_llm = llm_name
                logger.info(f"Selected OpenAI model: {llm_name}")
                break
            except Exception as e:
                error_msg = str(e)
                # If rate limited, try next model
                if "429" in error_msg or "rate_limit" in error_msg.lower() or "rate limit" in error_msg.lower():
                    logger.warning(f"Model {llm_name} is rate-limited, trying next model...")
                    continue
                # For other errors, also try next model
                logger.warning(f"Model {llm_name} failed: {error_msg[:100]}, trying next model...")
                continue
        
        if not selected_llm:
            # If all models failed, use the first one anyway (will show error later)
            selected_llm = openai_models[0]
            logger.warning(f"All OpenAI models failed, defaulting to {selected_llm}")
        
        for model in ["yolo_world_v2_m", "yolo_world_v2_s", "yolo_world_v2_l", "yolo_world_v2_x","yoloe_large", "yoloe", "yolo"]:
            if model in config.config["vision"]["models"]:
                # Use OpenAI for LLM
                pipeline = create_pipeline(vision_name=model, llm_name=selected_llm)
                logger.info(f"Created pipeline with vision model: {model} and LLM: {selected_llm}")
                return pipeline

        # Fallback to default (with OpenAI LLM)
        pipeline = create_pipeline(llm_name=selected_llm)
        logger.info(f"Created pipeline with default vision model and LLM: {selected_llm}")
        return pipeline
    except Exception as e:
        logger.error(f"Error creating pipeline: {e}")
        return None


# Initialize pipeline
pipeline = create_langvio_pipeline()

def ensure_browser_safe_video(input_path):
    if imageio_ffmpeg is None:
        print("[Warning] imageio_ffmpeg not available, skipping video re-encoding")
        return input_path
    output_path = input_path.replace(".mp4", "_browser.mp4")
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg_path, "-i", input_path, "-vcodec", "libx264", "-acodec", "aac", output_path]
    try:
        subprocess.run(cmd, check=True)
        return output_path
    except Exception as e:
        print(f"[Warning] Video re-encoding failed: {e}")
        return input_path
    
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def is_video(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {
        "mp4",
        "mov",
        "avi",
        "webm",
    }


def extract_image_stats(detections):
    """Extract statistics from new image format"""
    if not detections:
        return {}

    # For images, detections has 'objects' and 'summary' keys
    objects = detections.get("objects", [])
    summary = detections.get("summary", {})

    stats = {}

    # Basic image info
    image_info = summary.get("image_info", {})
    if image_info:
        stats["resolution"] = image_info.get("resolution", "unknown")
        stats["total_objects"] = image_info.get("total_objects", len(objects))
        stats["unique_types"] = image_info.get("unique_types", 0)

    # Object distribution
    object_dist = summary.get("object_distribution", {})
    stats["by_type"] = object_dist.get("by_type", {})
    stats["by_position"] = object_dist.get("by_position", {})
    stats["by_size"] = object_dist.get("by_size", {})
    stats["by_color"] = object_dist.get("by_color", {})

    # Notable patterns
    stats["notable_patterns"] = summary.get("notable_patterns", [])

    return stats


def extract_video_stats(detections):
    """Extract statistics from new video format"""
    if not detections:
        return {}

    stats = {}

    # For videos, detections has 'summary' and 'frame_detections' keys
    summary = detections.get("summary", {})
    frame_detections = detections.get("frame_detections", {})

    # Video info
    video_info = summary.get("video_info", {})
    if video_info:
        stats["duration"] = video_info.get("duration_seconds", 0)
        stats["resolution"] = video_info.get("resolution", "unknown")
        stats["fps"] = video_info.get("fps", 0)
        stats["activity_level"] = video_info.get("activity_level", "unknown")
        stats["primary_objects"] = video_info.get("primary_objects", [])

    # YOLO-World counting analysis
    counting = summary.get("counting_analysis", {})
    if counting:
        stats["counting"] = {
            "total_crossings": counting.get("total_crossings", 0),
            "objects_entered": counting.get("objects_entered", 0),
            "objects_exited": counting.get("objects_exited", 0),
            "net_flow": counting.get("net_flow", 0),
            "flow_direction": counting.get("flow_direction", "unknown"),
            "by_object_type": counting.get("by_object_type", {}),
            "most_active_type": counting.get("most_active_type", None),
        }

    # Speed analysis
    speed = summary.get("speed_analysis", {})
    if speed and speed.get("speed_available"):
        stats["speed"] = {
            "objects_with_speed": speed.get("objects_with_speed", 0),
            "average_speed_kmh": speed.get("average_speed_kmh", 0),
            "speed_category": speed.get("speed_category", "unknown"),
            "by_object_type": speed.get("by_object_type", {}),
            "fastest_type": speed.get("fastest_type", None),
        }

    # Temporal relationships
    temporal = summary.get("temporal_relationships", {})
    if temporal:
        movement = temporal.get("movement_patterns", {})
        stats["movement"] = {
            "stationary_count": movement.get("stationary_count", 0),
            "moving_count": movement.get("moving_count", 0),
            "fast_moving_count": movement.get("fast_moving_count", 0),
            "primary_directions": movement.get("primary_directions", {}),
        }
        stats["co_occurrence_events"] = temporal.get("co_occurrence_events", 0)

    # Spatial relationships
    spatial = summary.get("spatial_relationships", {})
    if spatial:
        stats["spatial"] = {
            "common_relations": spatial.get("common_relations", {}),
            "frequent_pairs": spatial.get("frequent_pairs", {}),
            "spatial_patterns": spatial.get("spatial_patterns", {}),
        }

    # Object analysis
    object_analysis = summary.get("object_analysis", {})
    if object_analysis:
        stats["object_characteristics"] = object_analysis.get(
            "object_characteristics", {}
        )
        stats["most_common_types"] = object_analysis.get("most_common_types", [])
        stats["total_unique_objects"] = object_analysis.get("total_unique_objects", 0)

    # Primary insights
    stats["primary_insights"] = summary.get("primary_insights", [])

    # Processing info
    processing_info = detections.get("processing_info", {})
    if processing_info:
        stats["frames_analyzed"] = processing_info.get("frames_analyzed", 0)
        stats["total_frames"] = processing_info.get("total_frames", 0)
        stats["yolo_world_enabled"] = processing_info.get("yolo_world_enabled", False)

    return stats


def get_object_counts_from_detections(detections, is_video_file):
    """Extract object counts from the new detection format"""
    object_counts = {}

    if is_video_file:
        frame_detections = detections.get("frame_detections", {})
        from collections import defaultdict, Counter
        frame_count_distributions = defaultdict(list)
        
        for frame_key, frame_dets in frame_detections.items():
            # Count unique track_ids per label in this frame
            frame_counts = defaultdict(set)
            for det in frame_dets:
                if "label" not in det:
                    continue
                label = det["label"]
                track_id = det.get("track_id") or det.get("object_id") or f"{frame_key}_{det.get('id', 'unknown')}"
                frame_counts[label].add(track_id)
            
            # Store the count for this frame for each label
            for label, track_set in frame_counts.items():
                frame_count_distributions[label].append(len(track_set))
        
        # Use a robust counting method for each object type
        object_counts = {}
        for label, counts in frame_count_distributions.items():
            if not counts:
                continue
            
            count_freq = Counter(counts)
            max_count = max(counts)
            mode_count = count_freq.most_common(1)[0][0]
            
            
            if count_freq[max_count] >= 2:
                # Max count appears multiple times - use it (reliable)
                object_counts[label] = max_count
            elif max_count > mode_count * 1.5:
                # Max is significantly higher than mode - likely accurate despite appearing once
                # This handles cases where objects appear together but tracking is inconsistent
                object_counts[label] = max_count
            else:
                # Find the highest count that appears in at least 2 frames
                consistent_count = mode_count  # Default to mode
                for count in sorted(count_freq.keys(), reverse=True):
                    if count_freq[count] >= 2:
                        consistent_count = count
                        break
                object_counts[label] = consistent_count
    else:
        # For images, look in summary -> object_distribution -> by_type
        summary = detections.get("summary", {})
        object_dist = summary.get("object_distribution", {})
        object_counts = object_dist.get("by_type", {})

        # Fallback: count from objects list
        if not object_counts:
            objects = detections.get("objects", [])
            from collections import Counter

            counts = Counter()
            for obj in objects:
                if "label" in obj:
                    counts[obj["label"]] += 1
            object_counts = dict(counts.most_common())

    return object_counts


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process_media():
    if "file" not in request.files:
        flash("No file part")
        return redirect(request.url)

    file = request.files["file"]
    query = request.form.get("query", "Describe what is in this image/video")

    if file.filename == "":
        flash("No selected file")
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Generate a unique filename to avoid collisions
        original_filename = secure_filename(file.filename)
        filename = f"{int(time.time())}_{uuid.uuid4().hex}_{original_filename}"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        # Save the uploaded file
        file.save(file_path)

        try:
            # Check if pipeline is initialized
            if pipeline is None:
                flash(
                    "Error: Could not initialize the Langvio pipeline. Please check your installation."
                )
                return redirect(url_for("index"))

            # Process with Langvio
            logger.info(f"Processing file: {filename} with query: {query}")
            start_time = time.time()
            result = pipeline.process(query, file_path)
            processing_time = time.time() - start_time

            # Get the output path and copy to static directory for serving
            # Get the output path and explanation from pipeline
            output_path = result.get("output_path", "")
            explanation = result.get("explanation", "No explanation provided.")

            # Ensure the video is browser-friendly
            if is_video(filename) and output_path:
                safe_output_path = ensure_browser_safe_video(output_path)
                if os.path.exists(safe_output_path):
                    output_path = safe_output_path


            if not output_path or not os.path.exists(output_path):
                flash("Error processing the file. No output was generated.")
                return redirect(url_for("index"))

            # Create a unique name for the result file
            output_filename = os.path.basename(output_path)
            destination = os.path.join(
                app.root_path, "static", "results", output_filename
            )

            # Copy the result file to the static directory
            import shutil

            shutil.copy2(output_path, destination)

            # Determine if the file is a video or image
            is_video_file = is_video(filename)
            result_url = url_for("static", filename=f"results/{output_filename}")

            # Extract statistics using the new format
            detections = result.get("detections", {})

            if is_video_file:
                stats = extract_video_stats(detections)
                video_stats = stats  # For backwards compatibility with template
                image_stats = {}
            else:
                stats = extract_image_stats(detections)
                image_stats = stats  # For backwards compatibility with template
                video_stats = {}

            # Get object counts using the new format
            object_counts = get_object_counts_from_detections(detections, is_video_file)

            # Extract additional data for template
            highlighted_objects_count = len(result.get("highlighted_objects", []))
            query_params = result.get("query_params", {})

            # Create detailed object list for display
            detailed_objects = []
            if is_video_file:
                # For videos, get a sample from frame detections
                frame_detections = detections.get("frame_detections", {})
                if frame_detections:
                    # Get objects from the first few frames
                    sample_frames = list(frame_detections.keys())[:3]
                    seen_objects = set()

                    for frame_key in sample_frames:
                        for det in frame_detections[frame_key]:
                            obj_signature = f"{det.get('label', 'unknown')}_{det.get('object_id', 'unknown')}"
                            if obj_signature not in seen_objects:
                                detailed_objects.append(
                                    {
                                        "id": det.get("object_id", "unknown"),
                                        "label": det.get("label", "unknown"),
                                        "confidence": det.get("confidence", 0),
                                        "frame": frame_key,
                                        "attributes": det.get("attributes", {}),
                                        "track_id": det.get("track_id", None),
                                    }
                                )
                                seen_objects.add(obj_signature)

                                if len(detailed_objects) >= 10:  # Limit to 10 objects
                                    break
                        if len(detailed_objects) >= 10:
                            break
            else:
                # For images, get objects from the objects list
                objects = detections.get("objects", [])
                for obj in objects[:10]:  # Limit to 10 objects
                    detailed_objects.append(
                        {
                            "id": obj.get("id", "unknown"),
                            "label": obj.get("label", "unknown"),
                            "confidence": obj.get("confidence", 0),
                            "frame": "0",  # Images are frame 0
                            "attributes": {
                                "size": obj.get("size", "unknown"),
                                "color": obj.get("color", "unknown"),
                                "position": obj.get("position", "unknown"),
                            },
                            "track_id": None,
                        }
                    )

            # Return the results page with enhanced data
            return render_template(
                "result.html",
                result_url=result_url,
                explanation=explanation,
                is_video=is_video_file,
                query=query,
                processing_time=round(processing_time, 2),
                # Original stats format for backwards compatibility
                stats=image_stats if not is_video_file else video_stats,
                object_counts=object_counts,
                video_stats=video_stats,
                # Enhanced data from new format
                image_stats=image_stats,
                detailed_objects=detailed_objects,
                highlighted_objects_count=highlighted_objects_count,
                query_params=query_params,
                # Additional context
                task_type=query_params.get("task_type", "identification"),
                target_objects=query_params.get("target_objects", []),
                requested_attributes=query_params.get("attributes", []),
            )

        except Exception as e:
            import traceback

            logger.error(f"Error processing file: {str(e)}")
            logger.error(traceback.format_exc())
            flash(f"Error processing your file: {str(e)}")
            return redirect(url_for("index"))
    else:
        flash(
            "File type not allowed. Please upload an image (png, jpg, jpeg) or video (mp4, mov, avi, webm)."
        )
        return redirect(url_for("index"))


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/results/<filename>")
def result_file(filename):
    return send_from_directory(app.config["RESULTS_FOLDER"], filename)


if __name__ == "__main__":
    # Check if pipeline was created successfully
    if pipeline is None:
        logger.error(
            "Failed to initialize Langvio pipeline. Check if Langvio is installed correctly."
        )
    else:
        app.run(debug=True, host="0.0.0.0", port=5001)
