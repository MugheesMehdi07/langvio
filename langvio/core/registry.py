"""
Registry for langvio models and processors
"""

from typing import Any, Dict, Type


class ModelRegistry:
    """Registry for all available models and processors"""

    def __init__(self):
        """Initialize empty registries"""
        self._llm_processors = {}
        self._vision_processors = {}

    def register_llm_processor(
            self, name: str, processor_class: Type, **kwargs
    ) -> None:
        """
        Register an LLM processor.

        Args:
            name: Name to register the processor under
            processor_class: Processor class
            **kwargs: Additional parameters to pass to the constructor
        """
        self._llm_processors[name] = (processor_class, kwargs)

    def register_vision_processor(
            self, name: str, processor_class: Type, **kwargs
    ) -> None:
        """
        Register a vision processor.

        Args:
            name: Name to register the processor under
            processor_class: Processor class
            **kwargs: Additional parameters to pass to the constructor
        """
        self._vision_processors[name] = (processor_class, kwargs)

    def get_llm_processor(self, name: str, **kwargs) -> Any:
        """
        Get an instance of an LLM processor.

        Args:
            name: Name of the registered processor
            **kwargs: Override parameters for the constructor

        Returns:
            Processor instance

        Raises:
            ValueError: If processor is not registered
        """
        if name not in self._llm_processors:
            raise ValueError(f"LLM processor '{name}' not registered")

        processor_class, default_kwargs = self._llm_processors[name]

        # Combine default kwargs with provided kwargs (provided take precedence)
        combined_kwargs = {**default_kwargs, **kwargs}

        return processor_class(**combined_kwargs)

    def get_vision_processor(self, name: str, **kwargs) -> Any:
        """
        Get an instance of a vision processor.

        Args:
            name: Name of the registered processor
            **kwargs: Override parameters for the constructor

        Returns:
            Processor instance

        Raises:
            ValueError: If processor is not registered
        """
        if name not in self._vision_processors:
            raise ValueError(f"Vision processor '{name}' not registered")

        processor_class, default_kwargs = self._vision_processors[name]

        # Combine default kwargs with provided kwargs (provided take precedence)
        combined_kwargs = {**default_kwargs, **kwargs}

        return processor_class(name, **combined_kwargs)

    def list_llm_processors(self) -> Dict[str, Type]:
        """
        List all registered LLM processors.

        Returns:
            Dictionary of processor names to processor classes
        """
        return {name: cls for name, (cls, _) in self._llm_processors.items()}

    def list_vision_processors(self) -> Dict[str, Type]:
        """
        List all registered vision processors.

        Returns:
            Dictionary of processor names to processor classes
        """
        return {name: cls for name, (cls, _) in self._vision_processors.items()}

    def register_from_entrypoints(self) -> None:
        """Load and register processors from entry points"""
        try:
            import importlib.metadata as metadata
        except ImportError:
            # Python < 3.8
            import importlib_metadata as metadata

        for ep in metadata.entry_points(group="langvio.llm_processors"):
            try:
                processor_class = ep.load()
                self.register_llm_processor(ep.name, processor_class)
            except Exception as e:
                # Log error and continue
                print(f"Error loading LLM processor {ep.name}: {e}")

        for ep in metadata.entry_points(group="langvio.vision_processors"):
            try:
                processor_class = ep.load()
                self.register_vision_processor(ep.name, processor_class)
            except Exception as e:
                # Log error and continue
                print(f"Error loading vision processor {ep.name}: {e}")
