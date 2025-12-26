# Langvio Examples

This guide provides practical examples for different use cases with Langvio.

## Basic Examples

### 1. Simple Object Detection
```python
import langvio

pipeline = langvio.create_pipeline()

# Basic detection
result = pipeline.process(
    "What objects are in this image?",
    "scene.jpg"
)
print(result['explanation'])
```

### 2. Object Counting
```python
# Count specific objects
result = pipeline.process(
    "How many people are in this photo?",
    "group_photo.jpg"
)

# Count multiple types
result = pipeline.process(
    "Count all vehicles: cars, trucks, and motorcycles",
    "traffic.jpg"
)
```

### 3. Attribute-Based Search
```python
# Find objects by color
result = pipeline.process(
    "Find all red objects in this image",
    "colorful_scene.jpg"
)

# Find by size
result = pipeline.process(
    "Show me all large objects",
    "warehouse.jpg"
)

# Combine attributes
result = pipeline.process(
    "Find small red cars",
    "parking_lot.jpg"
)
```

## Spatial Relationship Examples

### 4. Location-Based Queries
```python
# Objects in specific locations
result = pipeline.process(
    "What objects are on the table?",
    "kitchen.jpg"
)

# Spatial relationships
result = pipeline.process(
    "What is next to the blue car?",
    "street_scene.jpg"
)

# Position descriptions
result = pipeline.process(
    "Describe the location of each person in the image",
    "crowd.jpg"
)
```

## Video Analysis Examples

### 5. Movement Tracking
```python
# Track people movement
result = pipeline.process(
    "Track all people walking through the scene",
    "pedestrian_area.mp4"
)

# Vehicle tracking
result = pipeline.process(
    "How many vehicles passed through the intersection?",
    "traffic_cam.mp4"
)
```

### 6. Activity Detection
```python
# General activity
result = pipeline.process(
    "What activities are happening in this video?",
    "playground.mp4"
)

# Specific activities
result = pipeline.process(
    "Are people running or walking?",
    "park_joggers.mp4"
)

# Security monitoring
result = pipeline.process(
    "Detect any unusual or suspicious activities",
    "security_footage.mp4"
)
```

### 7. Speed and Flow Analysis
```python
# Speed estimation
result = pipeline.process(
    "What is the average speed of vehicles?",
    "highway.mp4"
)

# Flow analysis
result = pipeline.process(
    "How many people entered vs exited the building?",
    "entrance_monitoring.mp4"
)
```

## Business Use Cases

### 8. Retail Analytics
```python
# Customer counting
result = pipeline.process(
    "Count customers in the store at any given time",
    "store_camera.mp4"
)

# Product analysis
result = pipeline.process(
    "How many red shirts are visible on the clothing rack?",
    "retail_display.jpg"
)

# Queue analysis
result = pipeline.process(
    "How many people are waiting in line?",
    "checkout_line.jpg"
)
```

### 9. Security and Surveillance
```python
# Perimeter monitoring
result = pipeline.process(
    "Are there any people in the restricted area?",
    "security_zone.jpg"
)

# Vehicle monitoring
result = pipeline.process(
    "Count vehicles in the parking lot and identify any unusual ones",
    "parking_security.jpg"
)

# Incident detection
result = pipeline.process(
    "Detect any falls, fights, or emergency situations",
    "security_feed.mp4"
)
```

### 10. Traffic Analysis
```python
# Traffic density
result = pipeline.process(
    "How congested is the traffic? Count vehicles in each lane",
    "highway_cam.jpg"
)

# Intersection analysis
result = pipeline.process(
    "Monitor traffic light compliance and count violations",
    "intersection.mp4"
)

# Parking analysis
result = pipeline.process(
    "How many parking spaces are occupied vs available?",
    "parking_garage.jpg"
)
```

## Advanced Examples

### 11. Complex Multi-Part Analysis
```python
# Comprehensive scene analysis
result = pipeline.process(
    """Analyze this street scene and provide:
    1. Count of all people and vehicles
    2. Dominant colors of clothing
    3. Weather conditions based on what people are wearing
    4. Time of day indicators""",
    "street_scene.jpg"
)
```

### 12. Verification Tasks
```python
# Yes/No questions
result = pipeline.process(
    "Is there a dog in this image?",
    "park_scene.jpg"
)

# Safety verification
result = pipeline.process(
    "Are all workers wearing safety helmets?",
    "construction_site.jpg"
)

# Compliance checking
result = pipeline.process(
    "Are people wearing masks in this indoor space?",
    "office_meeting.jpg"
)
```

### 13. Batch Processing
```python
import os
import pandas as pd

pipeline = langvio.create_pipeline()

# Process multiple images
results = []
image_dir = "dataset/"

for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        result = pipeline.process(
            "Count people and vehicles",
            os.path.join(image_dir, filename)
        )
        
        results.append({
            'filename': filename,
            'explanation': result['explanation'],
            'people_count': extract_people_count(result['explanation']),
            'vehicle_count': extract_vehicle_count(result['explanation'])
        })

# Save results to CSV
df = pd.DataFrame(results)
df.to_csv('analysis_results.csv', index=False)
```

### 14. Custom Analysis Functions
```python
def analyze_store_traffic(video_path):
    """Analyze customer traffic patterns in retail store"""
    pipeline = langvio.create_pipeline()
    
    result = pipeline.process(
        "Count people entering and exiting, track their movement patterns, and identify peak activity times",
        video_path
    )
    
    return {
        'summary': result['explanation'],
        'traffic_data': result['detections'],
        'visualization': result['output_path']
    }

def safety_inspection(image_path):
    """Automated safety compliance checking"""
    pipeline = langvio.create_pipeline()
    
    queries = [
        "Are all workers wearing hard hats?",
        "Are safety barriers in place?",
        "Are there any safety violations visible?"
    ]
    
    results = {}
    for query in queries:
        result = pipeline.process(query, image_path)
        results[query] = result['explanation']
    
    return results
```

### 15. Integration Examples
```python
# Flask web service
from flask import Flask, request, jsonify
import langvio

app = Flask(__name__)
pipeline = langvio.create_pipeline()

@app.route('/analyze', methods=['POST'])
def analyze_media():
    file = request.files['media']
    query = request.form['query']
    
    # Save uploaded file
    filepath = f"uploads/{file.filename}"
    file.save(filepath)
    
    # Analyze
    result = pipeline.process(query, filepath)
    
    return jsonify({
        'explanation': result['explanation'],
        'output_url': f"/results/{os.path.basename(result['output_path'])}"
    })

# Command line tool
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', required=True)
    parser.add_argument('--media', required=True)
    parser.add_argument('--output', default='./output')
    
    args = parser.parse_args()
    
    pipeline = langvio.create_pipeline()
    result = pipeline.process(args.query, args.media)
    
    print(f"Analysis: {result['explanation']}")
    print(f"Output saved to: {result['output_path']}")

if __name__ == '__main__':
    main()
```

## Performance Optimization Examples

### 16. High-Performance Processing
```python
# For real-time applications
pipeline = langvio.create_pipeline(
    llm_name="gpt-4o-mini",  # Faster LLM
    vision_name="yolo11n"    # Fastest vision model
)

# Optimize for accuracy
pipeline = langvio.create_pipeline(
    llm_name="gpt-4",
    vision_name="yolo_world_v2_l"  # Large model for best accuracy
)
```

### 17. Memory-Efficient Batch Processing
```python
import gc
import torch

def process_large_dataset(file_list, query):
    pipeline = langvio.create_pipeline(vision_name="yolo11n")  # Lighter model
    
    results = []
    for i, filepath in enumerate(file_list):
        result = pipeline.process(query, filepath)
        results.append(result['explanation'])
        
        # Clear memory every 10 files
        if i % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return results
```

## Tips for Better Results

1. **Be specific in queries**: "Count red cars" vs "Count vehicles"
2. **Use natural language**: "How many people are wearing yellow?" 
3. **For videos, ask about patterns**: "What is the general movement pattern?"
4. **Combine multiple queries** for complex analysis
5. **Use verification queries** to double-check results: "Are there any missed objects?"