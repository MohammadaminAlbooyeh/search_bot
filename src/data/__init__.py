"""
Data Processing Package

This module contains data processing and indexing functionality for the search bot.
It handles document ingestion, text processing, and search index management.
"""

from .indexer import DocumentIndexer

__all__ = [
    "DocumentIndexer"
]

__version__ = "1.0.0"
__description__ = "Data processing and indexing components for search bot"