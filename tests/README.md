# Langvio Test Suite

This directory contains unit tests for the langvio package.

## Running Tests

### Using pytest (recommended)

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=langvio --cov-report=html

# Run specific test file
pytest tests/test_config.py

# Run specific test
pytest tests/test_config.py::TestConfig::test_default_config_loaded

# Run with verbose output
pytest -v
```

### Using unittest

```bash
# Run all tests
python -m unittest discover tests

# Run specific test file
python -m unittest tests.test_config

# Run specific test
python -m unittest tests.test_config.TestConfig.test_default_config_loaded
```

## Test Structure

- `test_config.py` - Tests for configuration management
- `test_file_utils.py` - Tests for file utility functions
- `test_registry.py` - Tests for model registry
- `test_processor_manager.py` - Tests for processor manager
- `test_pipeline.py` - Tests for main pipeline
- `test_logging.py` - Tests for logging utilities
- `test_llm_base.py` - Tests for LLM base processor
- `test_environment_variables.py` - Tests for environment variable handling

## Test Coverage

The test suite covers:
- Configuration loading and environment variable overrides
- File utility functions
- Model registry operations
- Processor manager lifecycle
- Pipeline processing flow
- Logging setup
- LLM processor initialization
- Environment variable validation

## Writing New Tests

When adding new tests:

1. Follow the naming convention: `test_*.py` for test files, `test_*` for test functions
2. Use descriptive test names that explain what is being tested
3. Include docstrings for test classes and methods
4. Use fixtures from `conftest.py` when appropriate
5. Mock external dependencies (API calls, file I/O, etc.)
6. Clean up temporary files and directories in `tearDown` methods

## Continuous Integration

Tests should pass before merging any pull request. The CI pipeline runs:
- All unit tests
- Code coverage checks
- Linting checks

