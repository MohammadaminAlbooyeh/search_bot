"""
Search Bot Package

This module contains the core search bot functionality including
the search engine and response handler components.
"""

from .search_engine import SearchEngine
from .response_handler import ResponseHandler

__all__ = [
    "SearchEngine",
    "ResponseHandler"
]

__version__ = "1.0.0"
__description__ = "Core search bot components for document search and response generation"