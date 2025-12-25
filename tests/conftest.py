"""
Pytest configuration and fixtures
"""

import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    # Cleanup
    import shutil
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)


@pytest.fixture
def temp_file():
    """Create a temporary file for tests"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        temp_path = f.name
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_config():
    """Sample configuration dictionary"""
    return {
        "llm": {
            "default": "gemini",
            "models": {
                "gemini": {
                    "model_name": "gemini-pro",
                    "model_kwargs": {"temperature": 0.2},
                }
            },
        },
        "vision": {
            "default": "yolo_world_v2_m",
            "models": {
                "yolo_world_v2_m": {
                    "type": "yolo_world",
                    "model_name": "yolov8m-worldv2",
                    "confidence": 0.45,
                }
            },
        },
        "media": {
            "output_dir": "./output",
            "temp_dir": "./temp",
        },
        "logging": {"level": "INFO", "file": None},
    }

