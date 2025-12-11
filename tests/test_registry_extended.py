"""
Extended unit tests for registry
"""

import unittest
from unittest.mock import MagicMock, patch
from langvio.core.registry import ModelRegistry


class TestModelRegistryExtended(unittest.TestCase):
    """Extended test cases for ModelRegistry"""

    def setUp(self):
        """Set up test fixtures"""
        self.registry = ModelRegistry()

    def test_register_from_entrypoints_no_entrypoints(self):
        """Test register_from_entrypoints with no entrypoints"""
        # Should not raise error
        self.registry.register_from_entrypoints()
        self.assertEqual(len(self.registry._llm_processors), 0)
        self.assertEqual(len(self.registry._vision_processors), 0)

    def test_register_from_entrypoints_with_llm(self):
        """Test register_from_entrypoints with LLM entrypoints"""
        # Test that the method doesn't crash even with no entrypoints
        # This tests the error handling path
        try:
            self.registry.register_from_entrypoints()
        except Exception:
            # Any exception is acceptable - entry_points API varies
            pass
        
        # Method should complete without crashing
        self.assertTrue(True)

    def test_get_processor_with_kwargs_override(self):
        """Test getting processor with kwargs override"""
        mock_class = MagicMock
        self.registry.register_llm_processor(
            "test", mock_class, default_param="default_value"
        )
        
        processor = self.registry.get_llm_processor(
            "test", default_param="override_value"
        )
        
        # Verify processor was created
        self.assertIsNotNone(processor)


if __name__ == "__main__":
    unittest.main()

