"""
Configuration management for langvio
"""

import logging
import os
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv
import os

# Load environment variables from .env file if present
# Try to find .env file in multiple locations:
# 1. Current working directory
# 2. langvio package directory
# 3. Parent directory (project root)
env_paths = [
    os.path.join(os.getcwd(), ".env"),  # Current working directory
    os.path.join(os.path.dirname(__file__), "..", ".env"),  # langvio/.env
    os.path.join(os.path.dirname(__file__), "..", "..", ".env"),  # project root/.env
]

for env_path in env_paths:
    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)
        break
else:
    # If no .env file found, try default load_dotenv() behavior
    load_dotenv(override=True)

# Initialize logger for configuration module
logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for langvio"""

    DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "default_config.yaml")

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to a YAML configuration file
        """
        # Initialize empty config
        self.config = {}

        # First load default config
        self._load_default_config()

        # Then load user config if provided
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)

        # Finally, override with environment variables if set
        self._apply_environment_overrides()

    def _load_default_config(self) -> None:
        """Load the default configuration from default_config.yaml"""
        try:
            if os.path.exists(self.DEFAULT_CONFIG_PATH):
                with open(self.DEFAULT_CONFIG_PATH, "r") as f:
                    self.config = yaml.safe_load(f)
                    if self.config is None:  # Handle empty file case
                        self.config = {}
            else:
                # Fallback default configuration if file doesn't exist
                self.config = {
                    "llm": {
                        "default": "gemini",
                        "models": {
                            "gemini": {
                                "model_name": "gemini-pro",
                                "model_kwargs": {"temperature": 0.2},
                            },
                            "gpt": {
                                "model_name": "gpt-3.5-turbo",
                                "model_kwargs": {"temperature": 0.0},
                            },
                        },
                    },
                    "vision": {
                        "default": "yolo_world_v2_m",  # YOLO-World medium as default
                        "models": {
                            "yolo_world_v2_s": {  # small model – fastest
                                "type": "yolo_world",
                                "model_name": "yolov8s-worldv2",
                                "confidence": 0.5,
                                "track_thresh": 0.5,
                                "track_buffer": 30,
                                "match_thresh": 0.8,
                            },
                            "yolo_world_v2_m": {  # medium model – balanced
                                "type": "yolo_world",
                                "model_name": "yolov8m-worldv2",
                                "confidence": 0.45,
                                "track_thresh": 0.3,
                                "track_buffer": 70,
                                "match_thresh": 0.6,
                            },
                            "yolo_world_v2_l": {  # large model – most accurate
                                "type": "yolo_world",
                                "model_name": "yolov8l-worldv2",
                                "confidence": 0.5,
                                "track_thresh": 0.5,
                                "track_buffer": 30,
                                "match_thresh": 0.8,
                            },
                            "yolo_world_v2_x": {  # extra-large model – highest accuracy
                                "type": "yolo_world",
                                "model_name": "yolov8x-worldv2",
                                "confidence": 0.5,
                                "track_thresh": 0.5,
                                "track_buffer": 30,
                                "match_thresh": 0.8,
                            },
                        },
                    },
                    "media": {
                        "output_dir": "./output",
                        "temp_dir": "./temp",
                        "visualization": {
                            "box_color": [0, 255, 0],
                            "text_color": [255, 255, 255],
                            "line_thickness": 2,
                        },
                    },
                    "logging": {"level": "INFO", "file": None},
                }
        except Exception as e:
            raise ValueError(f"Error loading default configuration: {e}")

    def load_config(self, config_path: str) -> None:
        """
        Load configuration from a YAML file.

        Args:
            config_path: Path to a YAML configuration file
        """
        try:
            with open(config_path, "r") as f:
                user_config = yaml.safe_load(f)
                if user_config is None:  # Handle empty file case
                    return

            # Update configuration
            self._update_config(self.config, user_config)
        except Exception as e:
            raise ValueError(f"Error loading configuration from {config_path}: {e}")

    def _update_config(
        self, base_config: Dict[str, Any], new_config: Dict[str, Any]
    ) -> None:
        """Recursively update base config with new config."""
        for key, value in new_config.items():
            if (
                isinstance(value, dict)
                and key in base_config
                and isinstance(base_config[key], dict)
            ):
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
            raise ValueError(f'LLM model "{model_name}" not found in configuration')

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
            # If model is not in config but might be registered, return empty config
            # The registry will use its default kwargs
            logger.warning(
                f'Vision model "{model_name}" not found in configuration. '
                f"Using default registry settings if available."
            )
            return {}

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

    def get_langsmith_config(self) -> Dict[str, Any]:
        """
        Get LangSmith configuration if available.

        Returns:
            LangSmith configuration dictionary
        """
        return self.config.get("langsmith", {})

    def _apply_environment_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        # Override default LLM from environment variable if set
        env_default_llm = os.environ.get("LANGVIO_DEFAULT_LLM")
        if env_default_llm and "llm" in self.config:
            # Map common aliases to actual model names (matching factory registrations)
            llm_alias_map = {
                "openai": "gpt-3.5",  # Default OpenAI model
                "gpt": "gpt-3.5",
                "gpt-3": "gpt-3.5",  # Map gpt-3 config key to gpt-3.5 registry name
                "gpt-3.5": "gpt-3.5",
                "gpt-4": "gpt-4.1-mini",
                "gemini": "gemini",
            }

            # Use mapped name if available, otherwise use the env value directly
            mapped_name = llm_alias_map.get(env_default_llm.lower(), env_default_llm)

            # Check if mapped name exists in config models or is a valid registry name
            if "models" in self.config["llm"]:
                # First check if mapped_name exists in config
                if mapped_name in self.config["llm"]["models"]:
                    self.config["llm"]["default"] = mapped_name
                    logger.info(
                        f"Overriding default LLM from environment: "
                        f"{env_default_llm} -> {mapped_name}"
                    )
                # Also check if the original env value exists (for direct model names)
                elif env_default_llm in self.config["llm"]["models"]:
                    self.config["llm"]["default"] = env_default_llm
                    logger.info(
                        f"Overriding default LLM from environment: "
                        f"{env_default_llm} -> {env_default_llm}"
                    )
                # If it's a valid alias, set it anyway (registry will handle it)
                elif mapped_name in llm_alias_map.values():
                    self.config["llm"]["default"] = mapped_name
                    logger.info(
                        f"Overriding default LLM from environment: "
                        f"{env_default_llm} -> {mapped_name} (registry will handle)"
                    )
                else:
                    logger.warning(
                        f"Environment variable LANGVIO_DEFAULT_LLM='{env_default_llm}' "
                        f"does not match any configured LLM model. "
                        f"Available models: {list(self.config.get('llm', {}).get('models', {}).keys())}"
                    )

        # Override default vision model from environment variable
        env_vision = os.getenv("LANGVIO_DEFAULT_VISION")
        if env_vision:
            # Normalize the environment variable value
            env_vision_lower = env_vision.lower().replace("_", "-").replace(" ", "-")

            # Check if the value exists in config
            if (
                "vision" in self.config
                and "models" in self.config["vision"]
                and env_vision_lower in self.config["vision"]["models"]
            ):
                self.config["vision"]["default"] = env_vision_lower
                logger.info(
                    f"Overriding default vision model from environment: "
                    f"{env_vision} -> {env_vision_lower}"
                )
            else:
                logger.warning(
                    f"Environment variable LANGVIO_DEFAULT_VISION='{env_vision}' "
                    f"does not match any configured vision model. "
                    f"Available models: {list(self.config.get('vision', {}).get('models', {}).keys())}"
                )

    def save_config(self, config_path: str) -> None:
        """
        Save current configuration to a YAML file.

        Args:
            config_path: Path to save the configuration
        """
        try:
            with open(config_path, "w") as f:
                yaml.dump(self.config, f, default_flow_style=False)
        except Exception as e:
            raise ValueError(f"Error saving configuration to {config_path}: {e}")
