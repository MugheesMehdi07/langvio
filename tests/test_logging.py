"""
Unit tests for logging utilities
"""

import logging
import os
import tempfile
import unittest

from langvio.utils.logging import setup_logging


class TestLogging(unittest.TestCase):
    """Test cases for logging utilities"""

    def setUp(self):
        """Set up test fixtures"""
        # Reset logging configuration
        logging.root.handlers = []
        logging.root.setLevel(logging.WARNING)

    def test_setup_logging_default(self):
        """Test setting up logging with default config"""
        setup_logging()
        
        # Check that console handler is set up
        handlers = logging.root.handlers
        self.assertGreater(len(handlers), 0)
        self.assertIsInstance(handlers[0], logging.StreamHandler)

    def test_setup_logging_with_level(self):
        """Test setting up logging with custom level"""
        setup_logging({"level": "DEBUG"})
        self.assertEqual(logging.root.level, logging.DEBUG)

    def test_setup_logging_with_file(self):
        """Test setting up logging with file handler"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            log_file = f.name

        try:
            setup_logging({"level": "INFO", "file": log_file})
            
            # Check that file handler is set up
            handlers = logging.root.handlers
            file_handlers = [h for h in handlers if isinstance(h, logging.FileHandler)]
            self.assertGreater(len(file_handlers), 0)
            
            # Test logging to file
            logger = logging.getLogger(__name__)
            logger.info("Test message")
            
            # Close all handlers to release file locks (Windows issue)
            for handler in handlers:
                handler.close()
            
            # Verify file was created and contains message
            self.assertTrue(os.path.exists(log_file))
            with open(log_file, 'r') as f:
                content = f.read()
            self.assertIn("Test message", content)
        finally:
            # Close handlers again and remove file
            for handler in logging.root.handlers[:]:
                handler.close()
                logging.root.removeHandler(handler)
            # Small delay for Windows file system
            import time
            time.sleep(0.1)
            if os.path.exists(log_file):
                try:
                    os.unlink(log_file)
                except PermissionError:
                    # File might still be locked, skip deletion
                    pass

    def test_setup_logging_invalid_level(self):
        """Test setting up logging with invalid level defaults to INFO"""
        setup_logging({"level": "INVALID"})
        # Should default to INFO
        self.assertEqual(logging.root.level, logging.INFO)


if __name__ == "__main__":
    unittest.main()

