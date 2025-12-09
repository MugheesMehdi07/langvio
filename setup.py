from setuptools import find_packages, setup

setup(
    name="langvio",
    version="0.3.0",
    description="Connect language models to vision models for natural language visual analysis",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.20.0",
        "opencv-python>=4.5.0",
        "pyyaml>=6.0",
        "langchain-core>=0.1.0",  # LangChain core components
        "langchain-community>=0.0.10",  # LangChain community integrations
        "ultralytics>=8.0.0",
        "pillow>=8.0.0",
        "python-dotenv>=0.19.0",  # For loading .env files
        "tqdm>=4.64.0",
    ],
    extras_require={
        # Individual providers
        "openai": [
            "openai>=1.0.0",
            "langchain-openai>=0.0.1",
        ],
        "google": [
            "google-generativeai>=0.3.0",
            "langchain-google-genai>=0.0.1",
        ],
        # Grouped providers
        "all-llm": [
            "openai>=1.0.0",
            "langchain-openai>=0.0.1",
            "google-generativeai>=0.3.0",
            "langchain-google-genai>=0.0.1",
        ],
        # Development tools
        "dev": [
            "pytest>=6.0.0",
            "black>=21.5b2",
            "isort>=5.9.1",
            "flake8>=3.9.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "langvio=langvio.cli:main",
        ],
        "langvio.llm_processors": [
            "openai = langvio.llm.openai:OpenAIProcessor [openai]",
            "google = langvio.llm.google:GeminiProcessor [google]",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
