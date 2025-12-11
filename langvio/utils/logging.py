"""
Logging utilities
"""

import logging
import os
from typing import Any, Dict, Optional


def setup_logging(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Set up logging with configuration.

    Args:
        config: Logging configuration
    """
    if config is None:
        config = {"level": "INFO", "file": None}

    # Get log level
    level_name = config.get("level", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    # Basic configuration
    handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)

    # File handler if specified
    log_file = config.get("file")
    if log_file:
        # Create directory if needed
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)  # type: ignore[arg-type]

    # Configure root logger
    logging.basicConfig(level=level, handlers=handlers, force=True)
