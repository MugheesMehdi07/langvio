# API Reference

## Core Functions

### `create_pipeline(config_path=None, llm_name=None, vision_name=None)`

Creates and configures a Langvio pipeline.

**Parameters:**
- `config_path` (str, optional): Path to YAML configuration file
- `llm_name` (str, optional): LLM processor name ("gpt-3.5", "gpt-4", "gemini")
- `vision_name` (str, optional): Vision processor name ("yolo", "yoloe", "yoloe_large")

**Returns:**
- `Pipeline`: Configured pipeline object

**Example:**
```python
# Default pipeline
pipeline = langvio.create_pipeline()

# With specific models
pipeline = langvio.create_pipeline(
    llm_name="gpt-4",
    vision_name="yoloe_large"
)

# With config file
pipeline = langvio.create_pipeline(config_path="config.yaml")
```

## Pipeline Class

### `Pipeline.process(query, media_path)`

Process a query on an image or video.

**Parameters:**
- `query` (str): Natural language question about the media
- `media_path` (str): Path to image or video file

**Returns:**
- `dict`: Analysis results containing:
  - `explanation` (str): Natural language answer
  - `output_path` (str): Path to annotated media file
  - `detections` (dict): Structured detection data
  - `query_params` (dict): Parsed query parameters
  - `highlighted_objects` (list): Objects highlighted in visualization

**Example:**
```python
result = pipeline.process(
    "Count all people wearing red",
    "crowd_scene.jpg"
)
print(result['explanation'])
print(result['output_path'])
```

### `Pipeline.set_llm_processor(processor_name)`

Change the language model processor.

**Parameters:**
- `processor_name` (str): Name of LLM processor

**Example:**
```python
pipeline.set_llm_processor("gpt-4")
```

### `Pipeline.set_vision_processor(processor_name)`

Change the vision model processor.

**Parameters:**
- `processor_name` (str): Name of vision processor

**Example:**
```python
pipeline.set_vision_processor("yoloe_large")
```

## Configuration Class

### `Config(config_path=None)`

Manages configuration settings.

**Parameters:**
- `config_path` (str, optional): Path to YAML configuration file

**Methods:**

#### `get_llm_config(model_name=None)`
Get LLM model configuration.

#### `get_vision_config(model_name=None)`
Get vision model configuration.

#### `get_media_config()`
Get media processing configuration.

**Example:**
```python
from langvio.config import Config

config = Config("my_config.yaml")
llm_settings = config.get_llm_config("gpt-4")
```

## Result Structure

### Image Analysis Result
```python
{
    'explanation': 'I found 3 people in the image...',
    'output_path': './output/image_processed.jpg',
    'detections': {
        'objects': [
            {
                'id': 'obj_0',
                'label': 'person',
                'confidence': 0.85,
                'bbox': [100, 150, 200, 400],
                'size': 'medium',
                'color': 'red',
                'position': 'center-left'
            }
        ],
        'summary': {
            'total_objects': 3,
            'by_type': {'person': 3}
        }
    },
    'query_params': {
        'task_type': 'counting',
        'target_objects': ['person'],
        'attributes': [{'attribute': 'color', 'value': 'red'}]
    },
    'highlighted_objects': [...]
}
```

### Video Analysis Result
```python
{
    'explanation': 'Throughout the video, I observed...',
    'output_path': './output/video_processed.mp4',
    'detections': {
        'summary': {
            'video_info': {
                'duration_seconds': 30.5,
                'resolution': '1920x1080',
                'activity_level': 'high_activity'
            },
            'counting_analysis': {
                'total_crossings': 15,
                'objects_entered': 8,
                'objects_exited': 7,
                'by_object_type': {
                    'person': {'entered': 5, 'exited': 4},
                    'car': {'entered': 3, 'exited': 3}
                }
            },
            'speed_analysis': {
                'average_speed_kmh': 25.3,
                'by_object_type': {
                    'car': {'average_speed': 35.2}
                }
            }
        },
        'frame_detections': {
            '0': [{'label': 'person', ...}],
            '5': [{'label': 'car', ...}]
        }
    }
}
```

## Command Line Interface

### Basic Usage
```bash
langvio --query "QUERY" --media "FILE_PATH"
```

### Options
- `--query, -q`: Natural language query (required)
- `--media, -m`: Path to media file (required) 
- `--config, -c`: Configuration file path
- `--llm, -l`: LLM processor name
- `--vision, -v`: Vision processor name
- `--output, -o`: Output directory
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `--list-models`: List available models

### Examples
```bash
# Basic analysis
langvio -q "Count cars" -m parking.jpg

# With specific models
langvio -q "Find red objects" -m scene.jpg -l gpt-4 -v yoloe_large

# With config file
langvio -q "Analyze traffic" -m traffic.mp4 -c config.yaml

# List available models
langvio --list-models
```

## Supported File Formats

### Images
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)
- WebP (.webp)

### Videos
- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)
- WebM (.webm)

## Query Types

### Object Detection
```python
"What objects are in this image?"
"Identify all items in the scene"
"What can you see in this picture?"
```

### Counting
```python
"How many people are there?"
"Count all vehicles"
"How many red objects do you see?"
```

### Attribute Search
```python
"Find all red objects"
"Show me large items"
"Identify objects on the left side"
```

### Spatial Relationships
```python
"What is on the table?"
"What objects are near the car?"
"Describe object positions"
```

### Video Analysis
```python
"Track movement patterns"
"How many people crossed the street?"
"What activities are happening?"
"Measure vehicle speeds"
```

### Verification
```python
"Is there a dog in this image?"
"Are people wearing masks?"
"Is the area crowded?"
```

## Error Handling

### Common Errors

**Missing API Key:**
```python
# Error: LLM processor initialization failed
# Solution: Set OPENAI_API_KEY or GOOGLE_API_KEY in .env file
```

**File Not Found:**
```python
# Error: Media file not found
# Solution: Check file path and permissions
```

**Model Download Failed:**
```python
# Error: Failed to download YOLO model
# Solution: Check internet connection, ensure sufficient disk space
```

**Out of Memory:**
```python
# Error: CUDA out of memory
# Solution: Use smaller model (vision_name="yolo") or enable CPU mode
```

### Error Recovery
```python
try:
    result = pipeline.process(query, media_path)
except Exception as e:
    print(f"Analysis failed: {e}")
    # Fallback to basic analysis or retry with different settings
```

## Performance Guidelines

### For Speed
- Use `vision_name="yolo"` (fastest)
- Use `llm_name="gpt-3.5"` or `llm_name="gemini"`
- Lower confidence thresholds
- Process fewer video frames

### For Accuracy  
- Use `vision_name="yoloe_large"` (most accurate)
- Use `llm_name="gpt-4"` (best reasoning)
- Higher confidence thresholds
- Process more video frames

### For Cost Optimization
- Use `llm_name="gemini"` (free tier available)
- Use batch processing for multiple files
- Cache results when possible

## Environment Variables

```env
# Required
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key

# Optional
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LANGVIO_DEFAULT_LLM=gemini
LANGVIO_DEFAULT_VISION=yoloe
```