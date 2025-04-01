"""
Configuration management for langvio
"""

import os
import yaml
from typing import Dict, Any, Optional


class Config:
    """Configuration manager for langvio"""

    DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "default_config.yaml")

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to a YAML configuration file
        """
        # Default configuration
        self.config = {
            "llm": {
                "default": "langchain_openai",
                "models": {
                    "langchain_openai": {
                        "type": "langchain",
                        "model_name": "gpt-3.5-turbo",
                        "temperature": 0.0
                    }
                }
            },
            "vision": {
                "default": "yolo",
                "models": {
                    "yolo": {
                        "type": "yolo",
                        "model_path": "yolov8n.pt",
                        "confidence": 0.25
                    }
                }
            },
            "media": {
                "output_dir": "./output",
                "temp_dir": "./temp",
                "visualization": {
                    "box_color": [0, 255, 0],
                    "text_color": [255, 255, 255],
                    "line_thickness": 2
                }
            },
            "logging": {
                "level": "INFO",
                "file": None
            }
        }

        # Load user config if provided
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)

    def load_config(self, config_path: str) -> None:
        """
        Load configuration from a YAML file.

        Args:
            config_path: Path to a YAML configuration file
        """
        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)

            # Update configuration
            self._update_config(self.config, user_config)
        except Exception as e:
            raise ValueError(f"Error loading configuration from {config_path}: {e}")

    def _update_config(self, base_config: Dict[str, Any], new_config: Dict[str, Any]) -> None:
        """Recursively update base config with new config."""
        for key, value in new_config.items():
            if isinstance(value, dict) and key in base_config and isinstance(base_config[key], dict):
                self._update_config(base_config[key], value)
            else:
                base_config[key] = value

    def get_llm_config(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get configuration for an LLM model.

        Args:
            model_name: Name of the model to get config for

        Returns:
            Model configuration dictionary
        """
        if not model_name:
            model_name = self.config["llm"]["default"]

        if model_name not in self.config["llm"]["models"]:
            raise ValueError(f"LLM model '{model_name}' not found in configuration")

        return self.config["llm"]["models"][model_name]

    def get_vision_config(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get configuration for a vision model.

        Args:
            model_name: Name of the model to get config for

        Returns:
            Model configuration dictionary
        """
        if not model_name:
            model_name = self.config["vision"]["default"]

        if model_name not in self.config["vision"]["models"]:
            raise ValueError(f"Vision model '{model_name}' not found in configuration")

        return self.config["vision"]["models"][model_name]

    def get_media_config(self) -> Dict[str, Any]:
        """
        Get media processing configuration.

        Returns:
            Media configuration dictionary
        """
        return self.config["media"]

    def get_logging_config(self) -> Dict[str, Any]:
        """
        Get logging configuration.

        Returns:
            Logging configuration dictionary
        """
        return self.config["logging"]

    def save_config(self, config_path: str) -> None:
        """
        Save current configuration to a YAML file.

        Args:
            config_path: Path to save the configuration
        """
        try:
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        except Exception as e:
            raise ValueError(f"Error saving configuration to {config_path}: {e}")