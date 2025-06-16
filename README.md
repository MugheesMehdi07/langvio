# 🧠 Langvio: Natural Language Computer Vision

<dlangvioiv align="center">

![Langvio Logo](https://img.shields.io/badge/Langvio-Vision%20%2B%20Language-blue?style=for-the-badge&logo=python)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/langvio.svg)](https://badge.fury.io/py/langvio)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://langvio.readthedocs.io/)

**Connect language models to vision models for natural language visual analysis**

[🚀 Quick Start](#-quick-start) • [📖 Documentation](https://langvio.readthedocs.io/) • [🎯 Examples](#-examples) • [🔧 Installation](#-installation) • [🤝 Contributing](#-contributing)

</dlangvioiv>

---

## ✨ What is Langvio?

Langvio bridges the gap between **human language** and **computer vision**. Ask questions about images and videos in plain English, and get intelligent analysis powered by state-of-the-art vision models and language models.

### 🎯 Key Features

- **🗣️ Natural Language Interface**: Ask questions like "Count all red cars" or "Find people wearing yellow"
- **🎥 Multi-Modal Support**: Works with both images and videos
- **🚀 Powered by YOLO**: Uses YOLOv11 and YOLOe for fast, accurate object detection
- **🤖 LLM Integration**: Supports OpenAI GPT and Google Gemini for intelligent explanations
- **📊 Advanced Analytics**: Object counting, speed estimation, spatial relationships
- **🎨 Visual Output**: Generates annotated images/videos with detection highlights
- **🌐 Web Interface**: Includes a Flask web app for easy interaction
- **🔧 Extensible**: Easy to add new models and capabilities

## 🎬 See It In Action

```python
import langvio

# Create a pipeline
pipeline = langvio.create_pipeline()

# Analyze an image
result = pipeline.process(
    query="Count how many people are wearing red shirts",
    media_path="street_scene.jpg"
)

print(result['explanation'])
# Output: "I found 3 people wearing red shirts in the image. 
#          Two are located in the center-left area, and one is on the right side."

# View the annotated result
print(f"Annotated image saved to: {result['output_path']}")
```

## 🔧 Installation

### Basic Installation

```bash
pip install langvio
```

### With LLM Provider Support

Choose your preferred language model provider:

```bash
# For OpenAI models (GPT-3.5, GPT-4)
pip install langvio[openai]

# For Google Gemini models
pip install langvio[google]

# For all supported providers
pip install langvio[all-llm]

# For development
pip install langvio[dev]
```

### Environment Setup

1. **Create a `.env` file** for your API keys:

```bash
# Copy the template
cp .env.template .env
```

2. **Add your API keys** to `.env`:

```env
# For OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# For Google Gemini  
GOOGLE_API_KEY=your_google_api_key_here
```

3. **Langvio automatically loads** these environment variables!

## 🚀 Quick Start

### Basic Usage

```python
import langvio

# Create a pipeline (automatically detects available LLM providers)
pipeline = langvio.create_pipeline()

# Process an image
result = pipeline.process(
    query="What objects are in this image?",
    media_path="path/to/your/image.jpg"
)

print(result['explanation'])
print(f"Output: {result['output_path']}")
```

### Video Analysis

```python
# Analyze videos with temporal understanding
result = pipeline.process(
    query="Count vehicles crossing the intersection",
    media_path="traffic_video.mp4"
)

# Get detailed analysis including speed and movement patterns
print(result['explanation'])
```

### Web Interface

```bash
# Launch the web interface
cd webapp
python app.py

# Visit http://localhost:5000 in your browser
```

## 🎯 Examples

### Object Detection & Counting

```python
# Count specific objects
pipeline.process("How many cars are in this parking lot?", "parking.jpg")

# Find objects by attributes  
pipeline.process("Find all red objects in this image", "scene.jpg")

# Spatial relationships
pipeline.process("What objects are on the table?", "kitchen.jpg")
```

### Video Analysis

```python
# Track movement patterns
pipeline.process("Track people walking through the scene", "crowd.mp4")

# Speed analysis
pipeline.process("What's the average speed of vehicles?", "highway.mp4")

# Activity detection
pipeline.process("Detect any unusual activities", "security_footage.mp4")
```

### Advanced Queries

```python
# Complex multi-part analysis
pipeline.process(
    "Count people and vehicles, identify their locations, and note distinctive colors",
    "street_scene.jpg"
)

# Verification tasks
pipeline.process("Is there a dog in this image?", "park_scene.jpg")

# Temporal analysis
pipeline.process("How many people entered vs exited the building?", "entrance.mp4")
```

## 🏗️ Architecture

```mermaid
graph TD
    A[User Query] --> B[LLM Processor]
    B --> C[Query Parser]
    C --> D[Vision Processor]
    D --> E[YOLO Detection]
    E --> F[Attribute Analysis]
    F --> G[Spatial Relationships]
    G --> H[Temporal Tracking]
    H --> I[LLM Explanation]
    I --> J[Visualization]
    J --> K[Output]
```

### Core Components

- **🧠 LLM Processor**: Parses queries and generates explanations (OpenAI, Google Gemini)
- **👁️ Vision Processor**: Detects objects and attributes (YOLO, YOLOe)
- **🎨 Media Processor**: Creates visualizations and handles I/O
- **⚙️ Pipeline**: Orchestrates the entire workflow

## 📊 Supported Models

### Vision Models
- **YOLOv11** (nano, small, medium, large, extra-large)
- **YOLOe** (enhanced YOLO variants)
- Automatic model selection based on performance needs

### Language Models
- **OpenAI**: GPT-3.5 Turbo, GPT-4 Turbo
- **Google**: Gemini Pro, Gemini Flash
- Extensible architecture for adding more providers

## 🛠️ Configuration

### Custom Configuration

```yaml
# config.yaml
llm:
  default: "gemini"
  models:
    gemini:
      model_name: "gemini-2.0-flash"
      model_kwargs:
        temperature: 0.2

vision:
  default: "yoloe_large"
  models:
    yoloe_large:
      model_path: "yoloe-11l-seg-pf.pt"
      confidence: 0.5

media:
  output_dir: "./results"
  visualization:
    box_color: [0, 255, 0]
    line_thickness: 2
```

```python
# Use custom configuration
pipeline = langvio.create_pipeline(config_path="config.yaml")
```

### Command Line Interface

```bash
# Basic usage
langvio --query "Count the cars" --media image.jpg

# With custom configuration
langvio --query "Find red objects" --media scene.jpg --config custom.yaml

# List available models
langvio --list-models
```

## 🌟 Advanced Features

### YOLO11 Solutions Integration

- **Object Counting**: Automatic boundary-crossing detection
- **Speed Estimation**: Real-time speed analysis for video
- **Advanced Tracking**: Multi-object tracking across frames

### Spatial Relationship Analysis

- **Positional Understanding**: "objects on the table", "cars in the parking lot"
- **Relative Positioning**: left/right, above/below, near/far relationships
- **Containment Detection**: objects inside other objects

### Temporal Analysis (Video)

- **Movement Patterns**: Track object trajectories and behaviors
- **Activity Recognition**: Detect activities and interactions
- **Temporal Relationships**: Understand object co-occurrence

### Color & Attribute Detection

- **Advanced Color Recognition**: 50+ color categories with confidence scoring
- **Size Classification**: Automatic small/medium/large categorization
- **Multi-attribute Analysis**: Combined color, size, and position analysis

## 🚀 Performance & Optimization

### Model Selection Strategy

```python
# Automatic model selection based on use case
pipeline = langvio.create_pipeline()  # Uses best available model

# Manual model selection for specific needs
pipeline = langvio.create_pipeline(
    vision_name="yoloe_large",  # High accuracy
    llm_name="gpt-4"           # Advanced reasoning
)
```

### Optimization Tips

- **YOLOe models**: Better accuracy for complex scenes
- **YOLO11 models**: Faster processing for real-time applications
- **Confidence thresholds**: Adjust based on precision/recall needs
- **Frame sampling**: Control video processing speed vs accuracy

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/langvio.git
cd langvio

# Install in development mode
pip install -e .[dev]

# Run tests
pytest

# Format code
black langvio/
isort langvio/
```

### Contributing Guidelines

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

## 📚 Documentation

- **📖 [Full Documentation](https://langvio.readthedocs.io/)**: Comprehensive guides and API reference
- **🎯 [Examples](./examples/)**: Ready-to-run example scripts
- **🌐 [Web App](./webapp/)**: Flask web interface for easy testing
- **⚙️ [Configuration](./examples/config_examples/)**: Sample configuration files

## 🔗 Links & Resources

- **🐙 [GitHub Repository](https://github.com/yourusername/langvio)**
- **📦 [PyPI Package](https://pypi.org/project/langvio/)**
- **📖 [Documentation](https://langvio.readthedocs.io/)**
- **🐛 [Issue Tracker](https://github.com/yourusername/langvio/issues)**
- **💬 [Discussions](https://github.com/yourusername/langvio/discussions)**

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics** for the amazing YOLO models
- **LangChain** for LLM integration framework
- **OpenAI** and **Google** for language model APIs
- **OpenCV** for computer vision utilities

---

<div align="center">

**⭐ Star us on GitHub if Langvio helps you!**

[⭐ Star](https://github.com/yourusername/langvio) • [🔗 Share](https://twitter.com/intent/tweet?text=Check%20out%20Langvio%20-%20Natural%20Language%20Computer%20Vision!&url=https://github.com/yourusername/langvio) • [🐛 Report Bug](https://github.com/yourusername/langvio/issues)

</div>