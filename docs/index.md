# Welcome to Langvio

<div align="center" markdown>

![Langvio Logo](assets/logo.png){ width="200" }

**Natural Language Computer Vision**

*Connect language models to vision models for intelligent visual analysis*

[Get Started](getting-started/installation.md){ .md-button .md-button--primary }
[View Examples](examples/basic-examples.md){ .md-button }
[GitHub Repository](https://github.com/yourusername/langvio){ .md-button }

</div>

---

## What is Langvio?

Langvio bridges the gap between human language and computer vision. Ask questions about images and videos in plain English, and get intelligent analysis powered by state-of-the-art vision models and language models.

```python
import langvio

# Create a pipeline
pipeline = langvio.create_pipeline()

# Ask a question about your image
result = pipeline.process(
    query="How many people are wearing red shirts?",
    media_path="street_scene.jpg"
)

print(result['explanation'])
# Output: "I found 3 people wearing red shirts in the image..."
```

## Key Features

<div class="grid cards" markdown>

-   :material-chat-processing:{ .lg .middle } __Natural Language Interface__

    ---

    Ask questions in plain English. No need to learn complex APIs or computer vision terminology.

    [:octicons-arrow-right-24: Learn more](guides/query-types.md)

-   :material-camera-burst:{ .lg .middle } __Multi-Modal Support__

    ---

    Works with both images and videos. Analyze static scenes or track objects over time.

    [:octicons-arrow-right-24: Image Guide](guides/image-analysis.md)

-   :material-rocket-launch:{ .lg .middle } __Powered by YOLO__

    ---

    Uses YOLOv11 and YOLOe for fast, accurate object detection with advanced tracking capabilities.

    [:octicons-arrow-right-24: Architecture](development/architecture.md)

-   :material-brain:{ .lg .middle } __LLM Integration__

    ---

    Supports OpenAI GPT and Google Gemini for intelligent explanations and complex reasoning.

    [:octicons-arrow-right-24: Configuration](guides/configuration.md)

-   :material-chart-line:{ .lg .middle } __Advanced Analytics__

    ---

    Object counting, speed estimation, spatial relationships, and temporal analysis.

    [:octicons-arrow-right-24: Video Analysis](guides/video-analysis.md)

-   :material-web:{ .lg .middle } __Web Interface__

    ---

    Includes a Flask web application for easy interaction and demonstration.

    [:octicons-arrow-right-24: Web Interface](getting-started/web-interface.md)

</div>

## Quick Example

=== "Image Analysis"

    ```python
    import langvio
    
    pipeline = langvio.create_pipeline()
    
    # Basic object detection
    result = pipeline.process(
        "What objects are in this image?",
        "scene.jpg"
    )
    ```

=== "Object Counting"

    ```python
    import langvio
    
    pipeline = langvio.create_pipeline()
    
    # Count specific objects
    result = pipeline.process(
        "Count all the cars in the parking lot",
        "parking_lot.jpg"
    )
    ```

=== "Video Analysis"

    ```python
    import langvio
    
    pipeline = langvio.create_pipeline()
    
    # Analyze video content
    result = pipeline.process(
        "How many people crossed the street?",
        "traffic_video.mp4"
    )
    ```

=== "Attribute Detection"

    ```python
    import langvio
    
    pipeline = langvio.create_pipeline()
    
    # Find objects by attributes
    result = pipeline.process(
        "Find all red objects in this scene",
        "colorful_scene.jpg"
    )
    ```

## Use Cases

**🏢 Business & Retail**
- Customer analytics and behavior tracking
- Inventory management and product recognition
- Queue length monitoring and optimization

**🛡️ Security & Surveillance** 
- Automated threat detection and monitoring
- Perimeter security and access control
- Incident analysis and reporting

**🚗 Transportation & Traffic**
- Traffic flow analysis and optimization
- Vehicle counting and classification
- Speed monitoring and safety analysis

**🔬 Research & Academia**
- Wildlife monitoring and behavior studies
- Medical image analysis and diagnostics
- Scientific data collection and analysis

## Getting Started

Ready to start using Langvio? Follow our quick installation guide:

1. **[Install Langvio](getting-started/installation.md)** - Set up the library with your preferred LLM provider
2. **[Quick Start](getting-started/quick-start.md)** - Run your first analysis in 5 minutes
3. **[Examples](examples/basic-examples.md)** - Explore real-world use cases
4. **[Configuration](guides/configuration.md)** - Customize for your needs

## Community & Support

- **📖 Documentation**: Complete guides and API reference
- **🐛 Bug Reports**: [GitHub Issues](https://github.com/yourusername/langvio/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/yourusername/langvio/discussions)
- **📧 Email**: support@langvio.dev

## License

Langvio is released under the [MIT License](license.md). Feel free to use it in your projects!

---

*Built with ❤️ by the Langvio team*