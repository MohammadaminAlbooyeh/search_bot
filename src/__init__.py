"""
Search Bot - A Python-based search engine package

This package provides functionality for indexing documents and performing
intelligent search operations with response handling.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "A search bot for document indexing and retrieval"

# Import main components for easier access
from .bot.search_engine import SearchEngine
from .bot.response_handler import ResponseHandler
from .data.indexer import DocumentIndexer
from .utils.config import Config

__all__ = [
    "SearchEngine",
    "ResponseHandler", 
    "DocumentIndexer",
    "Config"
]