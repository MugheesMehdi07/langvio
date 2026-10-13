# Getting Started with Langvio

Langvio connects language models to computer vision for natural language visual analysis. Ask questions about images and videos in plain English and get intelligent analysis.

## Quick Installation

### 1. Basic Installation
```bash
pip install langvio
```

### 2. Install the Vision Language Model
Langvio uses a local Qwen2.5-VL Vision Language Model (VLM):

```bash
# Install the local VLM support
pip install langvio[vlm]

# Equivalent alias
pip install langvio[all-llm]
```

> **Note:** The first run downloads several GB of model weights from HuggingFace. A GPU is recommended; CPU works but is significantly slower.

### 3. Environment Setup (Optional)
No API keys are required - Qwen2.5-VL runs entirely on your machine. You can optionally create a `.env` file to customize behavior:

```env
# Optional: HuggingFace cache location and token (for gated models)
HF_HOME=/path/to/hf/cache
HF_TOKEN=your_hf_token

# Optional: choose the default local model
LANGVIO_DEFAULT_LLM=qwen
```

## Your First Analysis

### Basic Image Analysis
```python
import langvio

# Create a pipeline (automatically detects available LLM)
pipeline = langvio.create_pipeline()

# Analyze an image
result = pipeline.process(
    query="What objects are in this image?",
    media_path="path/to/your/image.jpg"
)

print(result['explanation'])
print(f"Annotated image saved to: {result['output_path']}")
```

### Count Objects
```python
result = pipeline.process(
    query="Count all the cars in this parking lot",
    media_path="parking_lot.jpg"
)
```

### Find Objects by Attributes
```python
result = pipeline.process(
    query="Find all red objects in this scene",
    media_path="colorful_scene.jpg"
)
```

### Video Analysis
```python
result = pipeline.process(
    query="How many people crossed the street?",
    media_path="traffic_video.mp4"
)
```

## Web Interface

For a graphical interface, use the included web app:

```bash
cd webapp
python app.py
```

Visit `http://localhost:5000` in your browser to upload and analyze media files.

## What You Get

Each analysis returns:
- **Explanation**: Natural language answer to your question
- **Annotated Media**: Visual output with detected objects highlighted
- **Detection Data**: Structured information about found objects
- **Query Parameters**: How your question was interpreted

## Supported Queries

- **Object Detection**: "What's in this image?"
- **Counting**: "How many cars are there?"
- **Attributes**: "Find red cars" or "Show large objects"
- **Spatial Relations**: "What's on the table?"
- **Video Analysis**: "Track movement patterns"
- **Verification**: "Is there a dog in this image?"

## Next Steps

- Check the [Configuration Guide](../guides/configuration.md) to customize models and settings
- See [Examples](../guides/examples.md) for more advanced use cases
- Read the [API Reference](../reference/api.md) for detailed function documentation
- Learn about [Advanced Features](../advanced/features.md) like video tracking and spatial analysis

## Troubleshooting

### Common Issues

**No VLM installed error**: 
```bash
# Install the local Vision Language Model support
pip install langvio[vlm]  # or
pip install langvio[all-llm]
```

**Slow first run / large download**: 
- The first run downloads several GB of Qwen2.5-VL weights from HuggingFace
- Optionally set `HF_HOME` to control the cache location, and `HF_TOKEN` for gated models
- A GPU is recommended; CPU works but is significantly slower

**CUDA/GPU issues**: 
- Langvio automatically falls back to CPU if GPU isn't available
- To force CPU: `CUDA_VISIBLE_DEVICES=""`

**Model download**: 
- YOLO-World and Qwen2.5-VL models download automatically on first use (the VLM weights are several GB and may take a while)
- Ensure you have internet connection and sufficient disk space

**Memory issues**: 
- Use smaller models: `vision_name="yolo_world_v2_s"`
- Reduce video frame sampling rate
- Process images instead of videos for large files

**Import errors**: 
- Ensure all dependencies are installed: `pip install langvio[all-llm]`
- Check Python version (3.10+ required)

## Testing

Run the test suite to verify your installation:

```bash
pip install langvio[dev]
pytest
```

See [tests/README.md](../../tests/README.md) for more information.