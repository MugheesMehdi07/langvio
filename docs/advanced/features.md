# Advanced Features

Langvio includes sophisticated capabilities for complex visual analysis tasks beyond basic object detection.

## Video Analysis Features

### ByteTracker Multi-Object Tracking

Langvio uses **ByteTracker** - a state-of-the-art multi-object tracking algorithm - combined with YOLO-World v2 for advanced object tracking across video frames. ByteTracker maintains object identity through occlusions and complex scenes using Kalman filtering and IoU-based data association.

**ByteTracker Key Features:**
- **Persistent Object IDs**: Each tracked object maintains a unique ID throughout the video
- **Occlusion Handling**: Tracks survive temporary occlusions using track buffer management
- **Motion Prediction**: Kalman filters predict object locations for smoother tracking
- **Data Association**: IoU-based matching links detections to existing tracks
- **Configurable Parameters**: Adjust tracking thresholds, buffer size, and matching sensitivity

**Configuration Options:**
```yaml
vision:
  models:
    yolo_world_v2_m:
      track_thresh: 0.3      # Minimum confidence for tracking
      track_buffer: 70        # Frames to keep lost tracks
      match_thresh: 0.6       # IoU threshold for track matching
```

### Object Tracking and Counting

```python
# Track objects crossing boundaries
result = pipeline.process(
    "How many vehicles entered vs exited the intersection?",
    "traffic_intersection.mp4"
)

# The result includes detailed counting metrics:
counting_data = result['detections']['summary']['counting_analysis']
print(f"Total crossings: {counting_data.get('total_crossings', 0)}")
print(f"Net flow: {counting_data.get('net_flow', 0)}")
```

### Speed Estimation
Analyze movement speed of tracked objects:

```python
result = pipeline.process(
    "What is the average speed of vehicles?",
    "highway.mp4"
)

speed_data = result['detections']['summary']['speed_analysis']
if speed_data.get('speed_available'):
    print(f"Average speed: {speed_data.get('average_speed_kmh', 0)} km/h")
```

### Temporal Analysis
Understand object behavior over time:

```python
result = pipeline.process(
    "Describe movement patterns and activities",
    "activity_video.mp4"
)

temporal_data = result['detections']['summary']['temporal_relationships']
movement = temporal_data.get('movement_patterns', {})
print(f"Moving objects: {movement.get('moving_count', 0)}")
print(f"Stationary objects: {movement.get('stationary_count', 0)}")
```

## Spatial Relationship Analysis

### Position Detection
Identify object positions and relationships:

```python
# Find objects in specific locations
result = pipeline.process(
    "What objects are on the table?",
    "kitchen.jpg"
)

# Objects include spatial information:
for obj in result['detections']['objects']:
    print(f"{obj['label']}: {obj.get('position', 'unknown')}")
```

### Relative Positioning
Understand spatial relationships:

```python
result = pipeline.process(
    "What is next to the blue car?",
    "street_scene.jpg"
)
```

## Attribute Analysis

### Color Detection
Advanced color recognition with 50+ color categories:

```python
result = pipeline.process(
    "Find all red objects",
    "colorful_scene.jpg"
)

# Each detection includes color information:
for obj in result['detections']['objects']:
    attrs = obj.get('attributes', {})
    print(f"{obj['label']}: {attrs.get('color', 'unknown')}")
    print(f"  Colors detected: {attrs.get('colors', [])}")
```

### Size Classification
Automatic size categorization:

```python
# Objects are automatically classified as small/medium/large
result = pipeline.process(
    "Show me all large objects",
    "warehouse.jpg"
)
```

## Custom Query Processing

### Multi-Part Queries
Ask complex questions with multiple parts:

```python
result = pipeline.process(
    """Analyze this scene and provide:
    1. Count of all people and vehicles
    2. Dominant colors
    3. Spatial relationships
    4. Movement patterns""",
    "complex_scene.mp4"
)
```

### Verification Queries
Get yes/no answers:

```python
result = pipeline.process(
    "Is there a dog in this image?",
    "park_scene.jpg"
)
# Explanation will clearly state yes or no
```

## Performance Optimization

### Frame Sampling
Control video processing speed:

```python
# Default: processes every 5th frame
# For tracking tasks: automatically uses every 2nd frame
# For general analysis: uses every 5th frame

result = pipeline.process(
    "Track all movement",  # Automatically uses higher sample rate
    "video.mp4"
)
```

### Model Selection
Choose models based on your needs:

```python
# Speed-optimized
pipeline = langvio.create_pipeline(
    vision_name="yolo_world_v2_s",
    llm_name="gpt-3.5"
)

# Accuracy-optimized
pipeline = langvio.create_pipeline(
    vision_name="yolo_world_v2_l",
    llm_name="gpt-4"
)
```

## Logging and Debugging

Langvio provides comprehensive logging:

```python
import logging

# Set logging level
logging.basicConfig(level=logging.DEBUG)

# Process with detailed logs
result = pipeline.process(query, media_path)
# Logs include:
# - Query parsing steps
# - Detection counts
# - Processing times
# - Error details
```

## Batch Processing

Process multiple files efficiently:

```python
import os
pipeline = langvio.create_pipeline()

results = []
for filename in os.listdir("images/"):
    if filename.endswith(('.jpg', '.png')):
        result = pipeline.process(
            "Count all objects",
            os.path.join("images/", filename)
        )
        results.append({
            'file': filename,
            'explanation': result['explanation']
        })
```

## Integration Examples

### Flask Web Service
```python
from flask import Flask, request, jsonify
import langvio

app = Flask(__name__)
pipeline = langvio.create_pipeline()

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['media']
    query = request.form['query']
    
    filepath = f"uploads/{file.filename}"
    file.save(filepath)
    
    result = pipeline.process(query, filepath)
    return jsonify({
        'explanation': result['explanation'],
        'output_path': result['output_path']
    })
```

### Custom Analysis Class
```python
class SecurityAnalyzer:
    def __init__(self):
        self.pipeline = langvio.create_pipeline(
            config_path="security_config.yaml"
        )
    
    def check_perimeter(self, image_path):
        return self.pipeline.process(
            "Are there any people in the restricted area?",
            image_path
        )
    
    def monitor_traffic(self, video_path):
        return self.pipeline.process(
            "Count vehicles and detect any unusual activity",
            video_path
        )
```

## Tips for Best Results

1. **Be specific**: "Count red cars" is better than "Count vehicles"
2. **Use natural language**: Ask questions as you would to a person
3. **Combine queries**: Break complex analysis into multiple queries
4. **Check logs**: Use DEBUG level for troubleshooting
5. **Optimize models**: Choose models based on speed vs accuracy needs
6. **Use environment variables**: Set defaults via `.env` file
7. **Batch processing**: Process multiple files in a loop for efficiency