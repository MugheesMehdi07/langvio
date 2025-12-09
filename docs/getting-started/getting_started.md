# Getting Started with Langvio

Langvio connects language models to computer vision for natural language visual analysis. Ask questions about images and videos in plain English and get intelligent analysis.

## Quick Installation

### 1. Basic Installation
```bash
pip install langvio
```

### 2. Install with LLM Provider
Choose your preferred language model:

```bash
# For OpenAI models (GPT-3.5, GPT-4)
pip install langvio[openai]

# For Google Gemini models
pip install langvio[google]

# For all supported providers
pip install langvio[all-llm]
```

### 3. Set up API Keys
Create a `.env` file in your project directory:

```env
# For OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# For Google Gemini
GOOGLE_API_KEY=your_google_api_key_here
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

- Check the [Configuration Guide](configuration.md) to customize models and settings
- See [Examples](examples.md) for more advanced use cases
- Read the [API Reference](api_reference.md) for detailed function documentation
- Learn about [Advanced Features](advanced_features.md) like video tracking and spatial analysis

## Troubleshooting

### Common Issues

**No LLM provider error**: 
```bash
# Install an LLM provider
pip install langvio[openai]  # or
pip install langvio[google]  # or
pip install langvio[all-llm]
```

**Missing API Key**: 
- Set `OPENAI_API_KEY` or `GOOGLE_API_KEY` in your `.env` file
- Or export as environment variable: `export OPENAI_API_KEY=your_key`

**CUDA/GPU issues**: 
- Langvio automatically falls back to CPU if GPU isn't available
- To force CPU: `CUDA_VISIBLE_DEVICES=""`

**Model download**: 
- YOLO-World models download automatically on first use (may take a few minutes)
- Ensure you have internet connection and sufficient disk space

**Memory issues**: 
- Use smaller models: `vision_name="yolo_world_v2_s"`
- Reduce video frame sampling rate
- Process images instead of videos for large files

**Import errors**: 
- Ensure all dependencies are installed: `pip install langvio[all-llm]`
- Check Python version (3.8+ required)

## Testing

Run the test suite to verify your installation:

```bash
pip install langvio[dev]
pytest
```

See [tests/README.md](../../tests/README.md) for more information.