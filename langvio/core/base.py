"""
Base classes for langvio components
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class Processor(ABC):
    """Base class for all processors in langvio"""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize processor.

        Args:
            name: Processor name
            config: Configuration parameters
        """
        self.name = name
        self.config = config or {}

    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the processor with its configuration.

        Returns:
            True if initialization was successful
        """

