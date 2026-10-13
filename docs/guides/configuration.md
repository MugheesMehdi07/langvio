# Configuration Guide

Langvio can be configured through YAML files or programmatically to customize models, performance, and output settings.

## Basic Configuration

### Using Different Models

```python
import langvio

# Use specific models
pipeline = langvio.create_pipeline(
    llm_name="qwen-7b",         # Local Qwen2.5-VL 7B model
    vision_name="yoloe_large"   # YOLOe large model
)

# Use configuration file
pipeline = langvio.create_pipeline(config_path="my_config.yaml")
```

### Available Models

**Vision Models:**
- `yolo_world_v2_s` - YOLO-World v2 small (fastest, good accuracy)
- `yolo_world_v2_m` - YOLO-World v2 medium (balanced speed/accuracy, recommended default)
- `yolo_world_v2_l` - YOLO-World v2 large (better accuracy)
- `yolo_world_v2x` - YOLO-World v2 extra-large (best accuracy, slower)
- `yolo11n` - YOLO11 nano (alias to YOLO-World v2 small, fastest)
- `yolo` - YOLO11 alias (alias to YOLO-World v2 small)
- `yoloe` - YOLOe alias (alias to YOLO-World v2 small)
- `yoloe_medium` - YOLOe medium (alias to YOLO-World v2 medium)
- `yoloe_large` - YOLOe large (alias to YOLO-World v2 large)

**Note:** All vision models use YOLO-World v2 under the hood. The aliases (yolo11n, yoloe, etc.) map to YOLO-World v2 models for flexible object detection without predefined classes.

**Language Models (local Qwen2.5-VL Vision Language Model):**
- `qwen` or `qwen-3b` - Qwen/Qwen2.5-VL-3B-Instruct (default, smallest, fastest)
- `qwen-7b` - Qwen/Qwen2.5-VL-7B-Instruct (larger, best reasoning)
- `qwen2-vl-7b` - Qwen/Qwen2-VL-7B-Instruct (previous-generation 7B model)

**Note:** All language models run locally via HuggingFace transformers - no API keys required. The first run downloads several GB of weights; a GPU is recommended, though CPU works but is slower.

## Configuration File

Create a `config.yaml` file:

```yaml
# Language Model Settings (local Qwen2.5-VL)
llm:
  default: "qwen"  # Default model
  models:
    qwen:
      model_name: "Qwen/Qwen2.5-VL-3B-Instruct"
      model_kwargs:
        temperature: 0.1
        max_new_tokens: 1024

    qwen-7b:
      model_name: "Qwen/Qwen2.5-VL-7B-Instruct"
      model_kwargs:
        temperature: 0.1
        max_new_tokens: 2048

    qwen2-vl-7b:
      model_name: "Qwen/Qwen2-VL-7B-Instruct"
      model_kwargs:
        temperature: 0.1
        max_new_tokens: 2048

# Vision Model Settings
vision:
  default: "yolo11n"  # Fastest default, or use "yolo_world_v2_m" for balanced
  models:
    yolo_world_v2_m:
      type: "yolo_world"
      model_name: "yolov8m-worldv2"
      confidence: 0.45
      track_thresh: 0.3
      track_buffer: 70
      match_thresh: 0.6
    
    yolo_world_v2_s:
      type: "yolo_world"
      model_name: "yolov8s-worldv2"
      confidence: 0.5
      track_thresh: 0.5
      track_buffer: 30
      match_thresh: 0.8
    
    yolo11n:  # Alias for YOLO-World v2 small
      type: "yolo"
      model_path: "yolo11n.pt"
      confidence: 0.8
      model_type: "yolo"

# Output Settings
media:
  output_dir: "./output"
  temp_dir: "./temp"
  visualization:
    box_color: [0, 255, 0]      # Green boxes
    text_color: [255, 255, 255]  # White text
    line_thickness: 2
    show_attributes: true
    show_confidence: true

# Logging
logging:
  level: "INFO"
  file: "langvio.log"
```

## Performance Tuning

### For Speed (Real-time Applications)
```python
pipeline = langvio.create_pipeline(
    llm_name="qwen",              # Smallest, fastest VLM (3B)
    vision_name="yolo11n"          # Fastest vision model (or yolo_world_v2_s)
)
```

### For Accuracy (Research/Analysis)
```python
pipeline = langvio.create_pipeline(
    llm_name="qwen-7b",
    vision_name="yolo_world_v2_l"  # Large model for best accuracy
)
```

### For Resource Efficiency
```python
pipeline = langvio.create_pipeline(
    llm_name="qwen",              # Smaller weights, lower memory
    vision_name="yolo11n"          # Fastest model (or yolo_world_v2_m for balance)
)
```

## Video Processing Settings

### Adjust Frame Sampling
```python
# Process every frame (accurate but slow)
result = pipeline.process(query, video_path)

# Or modify in config.yaml
vision:
  models:
    yoloe:
      confidence: 0.3  # Lower = more detections
      sample_rate: 5   # Process every 5th frame
```

### Memory Optimization
```yaml
vision:
  models:
    yolo_efficient:
      model_path: "yolo11n.pt"
      confidence: 0.4
      max_detections: 50  # Limit detections per frame
```

## Output Customization

### Change Visualization Colors
```yaml
media:
  visualization:
    box_color: [255, 0, 0]      # Red boxes
    highlight_color: [0, 0, 255] # Blue for highlighted objects
    text_color: [255, 255, 255]  # White text
    line_thickness: 3
    show_attributes: true
    show_confidence: false       # Hide confidence scores
```

### Custom Output Directory
```python
pipeline = langvio.create_pipeline()
pipeline.config.config["media"]["output_dir"] = "/custom/output/path"
```

## Environment Variables

No API keys are required - Qwen2.5-VL runs locally. Set these in your `.env` file or environment if you want to customize behavior:

```env
# Optional: HuggingFace cache location and token (for gated models)
HF_HOME=/path/to/hf/cache
HF_TOKEN=your_hf_token

# Optional: Performance Settings
CUDA_VISIBLE_DEVICES=0          # Use specific GPU
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Optional: Model Settings (override defaults)
LANGVIO_DEFAULT_LLM=qwen
LANGVIO_DEFAULT_VISION=yolo_world_v2_m
```

## Command Line Usage

```bash
# Basic usage
langvio --query "Count cars" --media parking.jpg

# With custom config
langvio --query "Find red objects" --media scene.jpg --config my_config.yaml

# Specify models
langvio --query "Analyze video" --media traffic.mp4 --llm qwen-7b --vision yolo_world_v2_l

# Set output directory
langvio --query "Count people" --media crowd.jpg --output ./results

# List available models
langvio --list-models
```

## Advanced Configuration

### Custom Model Paths
```yaml
vision:
  models:
    custom_yolo:
      model_path: "/path/to/custom/model.pt"
      confidence: 0.3
      device: "cuda"  # or "cpu"
```

### Batch Processing Settings
```python
# For processing multiple files
import os
pipeline = langvio.create_pipeline()

# Process all images in a directory
image_dir = "images/"
for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        result = pipeline.process(
            "What's in this image?", 
            os.path.join(image_dir, filename)
        )
        print(f"{filename}: {result['explanation']}")
```

### Integration with Other Systems
```python
# Use in existing applications
class MyVisionAnalyzer:
    def __init__(self):
        self.pipeline = langvio.create_pipeline(
            config_path="production_config.yaml"
        )
    
    def analyze_security_footage(self, video_path):
        return self.pipeline.process(
            "Detect any unusual activities or security concerns",
            video_path
        )
```

## Configuration Tips

1. **Start with defaults** and adjust based on your needs
2. **Use YOLO-World v2 models** - they offer flexible object detection without predefined classes
3. **Model selection**: Use `yolo_world_v2_s` for speed, `yolo_world_v2_l` for accuracy
4. **Qwen runs locally** - use `qwen` (3B) for speed and lower memory, `qwen-7b` for best results
5. **Lower confidence values** detect more objects but may have false positives
6. **Increase line thickness** for better visibility in high-resolution images
7. **Disable attributes** for faster processing if not needed
8. **Use environment variables** for easy configuration without code changes
9. **Check logs** - Langvio provides detailed logging for debugging configuration issues