"""
Test Suite for Search Bot

This package contains all unit tests and integration tests for the search bot application.
Tests are organized by component and use pytest as the testing framework.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path for testing
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Test configuration
TEST_DATA_DIR = project_root / "tests" / "test_data"
TEST_DOCUMENTS_DIR = TEST_DATA_DIR / "documents"

# Create test directories if they don't exist
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_DOCUMENTS_DIR.mkdir(exist_ok=True)

# Test utilities
def setup_test_environment():
    """
    Set up the test environment with necessary configurations and test data.
    """
    # Ensure test directories exist
    TEST_DATA_DIR.mkdir(exist_ok=True)
    TEST_DOCUMENTS_DIR.mkdir(exist_ok=True)
    
    # Set test environment variables
    os.environ['SEARCH_BOT_TEST_MODE'] = 'true'
    os.environ['SEARCH_BOT_DATA_DIR'] = str(TEST_DATA_DIR)
    

def cleanup_test_environment():
    """
    Clean up test environment after tests complete.
    """
    # Remove test environment variables
    os.environ.pop('SEARCH_BOT_TEST_MODE', None)
    os.environ.pop('SEARCH_BOT_DATA_DIR', None)


def create_test_document(filename, content, directory=None):
    """
    Create a test document for testing purposes.
    
    Args:
        filename (str): Name of the test file
        content (str): Content to write to the file
        directory (Path, optional): Directory to create the file in
    
    Returns:
        Path: Path to the created test file
    """
    if directory is None:
        directory = TEST_DOCUMENTS_DIR
    
    file_path = directory / filename
    file_path.write_text(content, encoding='utf-8')
    return file_path


# Import test modules for easier access
from .test_indexer import *
from .test_search_engine import *

__all__ = [
    "setup_test_environment",
    "cleanup_test_environment", 
    "create_test_document",
    "TEST_DATA_DIR",
    "TEST_DOCUMENTS_DIR"
]