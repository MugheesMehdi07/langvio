from setuptools import find_packages, setup

setup(
    name="langvio",
    version="0.0.5",
    description="Connect language models to vision models for natural language visual analysis",
    author="Mughees Mehdi",
    author_email="mugheesmehdi@gmail.com",
    packages=find_packages(exclude=["tests*", "examples*", "webapp*", "docs*"]),
    install_requires=[
        "torch>=2.9.1",
        "ultralytics>=8.3.240",
        "opencv-python>=4.12.0.88",
        "numpy>=2.2.6",
        "pillow>=12.0.0",
        "langchain-core>=1.2.2",
        "langchain-community>=0.4.1",
        "pyyaml>=6.0.3",
        "python-dotenv>=1.2.1",
        "tqdm>=4.67.1",
    ],
    extras_require={
        # Individual providers
        "openai": [
            "openai>=1.0.0",
            "langchain-openai>=0.0.1",
        ],
        "google": [
            "google-generativeai>=0.3.0,<1.0.0",
            "langchain-google-genai>=2.0.0",
        ],
        # Grouped providers
        "all-llm": [
            "langvio[openai,google]",
        ],
        # Development tools
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        # Web app
        "webapp": [
            "flask>=2.0.0",
            "werkzeug>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "langvio=langvio.cli:main",
        ],
    },
    python_requires=">=3.8,<3.13",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
        "Topic :: Text Processing :: Linguistic",
    ],
    package_data={
        "langvio": ["*.yaml", "*.yml", "*.json"],
    },
)
