"""
Response Handler Module

This module handles the generation and formatting of responses for the search bot.
It processes search results and creates user-friendly, well-formatted responses
with various output formats and customization options.
"""

import logging
import re
import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import json
import markdown
from pathlib import Path

# Configuration
from ..utils.config import get_config
from .search_engine import SearchResult


@dataclass
class ResponseTemplate:
    """Template for response formatting."""
    no_results: str
    single_result: str
    multiple_results: str
    error: str
    result_item: str
    summary: str


class ResponseFormatter:
    """Handles different response formats (markdown, plain text, HTML, JSON)."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
    
    def format_markdown(self, content: str, results: List[SearchResult], 
                       query: str, **kwargs) -> str:
        """Format response as Markdown."""
        lines = []
        
        # Add header
        if len(results) == 0:
            lines.append(f"# No Results Found")
            lines.append(f"\nSorry, no documents were found matching **'{query}'**.")
        elif len(results) == 1:
            lines.append(f"# 1 Document Found")
            lines.append(f"\nFound 1 document matching **'{query}'**:")
        else:
            lines.append(f"# {len(results)} Documents Found")
            lines.append(f"\nFound {len(results)} documents matching **'{query}'**:")
        
        lines.append("")
        
        # Add results
        for i, result in enumerate(results, 1):
            lines.append(f"## {i}. {result.title}")
            lines.append(f"**File:** `{result.filename}`")
            lines.append(f"**Path:** `{result.path}`")
            lines.append(f"**Score:** {result.score:.3f}")
            
            if result.content_snippet:
                lines.append(f"\n**Preview:**")
                lines.append(f"> {result.content_snippet}")
            
            # Add metadata if available
            if self.config.response.include_source_info:
                lines.append(f"\n**Details:**")
                lines.append(f"- Size: {self._format_file_size(result.size)}")
                lines.append(f"- Words: {result.word_count:,}")
                lines.append(f"- Modified: {self._format_date(result.modified_time)}")
                
                if result.tags:
                    lines.append(f"- Tags: {', '.join(result.tags)}")
            
            lines.append("\n---\n")
        
        return "\n".join(lines)
    
    def format_plain_text(self, content: str, results: List[SearchResult], 
                         query: str, **kwargs) -> str:
        """Format response as plain text."""
        lines = []
        
        # Add header
        if len(results) == 0:
            lines.append("NO RESULTS FOUND")
            lines.append("=" * 50)
            lines.append(f"Sorry, no documents were found matching '{query}'.")
        elif len(results) == 1:
            lines.append("1 DOCUMENT FOUND")
            lines.append("=" * 50)
            lines.append(f"Found 1 document matching '{query}':")
        else:
            lines.append(f"{len(results)} DOCUMENTS FOUND")
            lines.append("=" * 50)
            lines.append(f"Found {len(results)} documents matching '{query}':")
        
        lines.append("")
        
        # Add results
        for i, result in enumerate(results, 1):
            lines.append(f"{i}. {result.title}")
            lines.append(f"   File: {result.filename}")
            lines.append(f"   Path: {result.path}")
            lines.append(f"   Score: {result.score:.3f}")
            
            if result.content_snippet:
                lines.append(f"   Preview: {result.content_snippet}")
            
            if self.config.response.include_source_info:
                lines.append(f"   Size: {self._format_file_size(result.size)}")
                lines.append(f"   Words: {result.word_count:,}")
                lines.append(f"   Modified: {self._format_date(result.modified_time)}")
                
                if result.tags:
                    lines.append(f"   Tags: {', '.join(result.tags)}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def format_html(self, content: str, results: List[SearchResult], 
                   query: str, **kwargs) -> str:
        """Format response as HTML."""
        html_parts = []
        
        # Add header
        if len(results) == 0:
            html_parts.append(f"<h1>No Results Found</h1>")
            html_parts.append(f"<p>Sorry, no documents were found matching <strong>'{query}'</strong>.</p>")
        elif len(results) == 1:
            html_parts.append(f"<h1>1 Document Found</h1>")
            html_parts.append(f"<p>Found 1 document matching <strong>'{query}'</strong>:</p>")
        else:
            html_parts.append(f"<h1>{len(results)} Documents Found</h1>")
            html_parts.append(f"<p>Found {len(results)} documents matching <strong>'{query}'</strong>:</p>")
        
        # Add results
        html_parts.append("<div class='search-results'>")
        
        for i, result in enumerate(results, 1):
            html_parts.append(f"<div class='result-item' data-rank='{i}'>")
            html_parts.append(f"<h2>{i}. {self._escape_html(result.title)}</h2>")
            html_parts.append(f"<p><strong>File:</strong> <code>{self._escape_html(result.filename)}</code></p>")
            html_parts.append(f"<p><strong>Path:</strong> <code>{self._escape_html(result.path)}</code></p>")
            html_parts.append(f"<p><strong>Score:</strong> {result.score:.3f}</p>")
            
            if result.content_snippet:
                html_parts.append(f"<div class='preview'><strong>Preview:</strong></div>")
                html_parts.append(f"<blockquote>{self._escape_html(result.content_snippet)}</blockquote>")
            
            if self.config.response.include_source_info:
                html_parts.append("<div class='details'>")
                html_parts.append(f"<p><strong>Size:</strong> {self._format_file_size(result.size)}</p>")
                html_parts.append(f"<p><strong>Words:</strong> {result.word_count:,}</p>")
                html_parts.append(f"<p><strong>Modified:</strong> {self._format_date(result.modified_time)}</p>")
                
                if result.tags:
                    tags_html = ", ".join(f"<span class='tag'>{self._escape_html(tag)}</span>" 
                                        for tag in result.tags)
                    html_parts.append(f"<p><strong>Tags:</strong> {tags_html}</p>")
                
                html_parts.append("</div>")
            
            html_parts.append("</div>")
        
        html_parts.append("</div>")
        
        return "\n".join(html_parts)
    
    def format_json(self, content: str, results: List[SearchResult], 
                   query: str, **kwargs) -> str:
        """Format response as JSON."""
        response_data = {
            "query": query,
            "total_results": len(results),
            "results": []
        }
        
        for result in results:
            result_data = {
                "title": result.title,
                "filename": result.filename,
                "path": result.path,
                "score": round(result.score, 3),
                "rank": result.rank,
                "content_snippet": result.content_snippet,
                "extension": result.extension,
                "size": result.size,
                "word_count": result.word_count,
                "modified_time": result.modified_time,
                "tags": result.tags,
                "highlights": result.highlights,
                "metadata": result.metadata
            }
            
            response_data["results"].append(result_data)
        
        return json.dumps(response_data, indent=2, ensure_ascii=False)
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        if size_bytes == 0:
            return "0 B"
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        
        return f"{size_bytes:.1f} TB"
    
    def _format_date(self, date_str: str) -> str:
        """Format date string for display."""
        if not date_str:
            return "Unknown"
        
        try:
            if isinstance(date_str, str):
                # Try to parse ISO format
                dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            else:
                dt = date_str
            
            return dt.strftime("%Y-%m-%d %H:%M")
        except:
            return str(date_str)
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        if not text:
            return ""
        
        return (text.replace('&', '&amp;')
                   .replace('<', '&lt;')
                   .replace('>', '&gt;')
                   .replace('"', '&quot;')
                   .replace("'", '&#x27;'))


class ResponseEnhancer:
    """Enhances responses with additional context and suggestions."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
    
    def add_suggestions(self, query: str, results: List[SearchResult]) -> List[str]:
        """Generate search suggestions based on results."""
        suggestions = []
        
        if not results:
            # Suggest related terms
            suggestions = [
                "Try different keywords",
                "Check your spelling",
                "Use broader terms",
                "Try using wildcards (*)"
            ]
        elif len(results) < 5:
            # Suggest refinements
            suggestions = [
                "Try more specific terms",
                "Add related keywords",
                "Use phrase search with quotes"
            ]
        
        return suggestions
    
    def add_statistics(self, results: List[SearchResult], search_time: float) -> Dict[str, Any]:
        """Generate response statistics."""
        if not results:
            return {}
        
        stats = {
            "total_results": len(results),
            "search_time_seconds": round(search_time, 3),
            "average_score": round(sum(r.score for r in results) / len(results), 3),
            "file_types": {}
        }
        
        # Count file types
        for result in results:
            ext = result.extension or "unknown"
            stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1
        
        return stats
    
    def add_related_files(self, results: List[SearchResult]) -> List[str]:
        """Find related files based on current results."""
        related = []
        
        # Extract directories from results
        directories = set()
        for result in results:
            path = Path(result.path)
            directories.add(str(path.parent))
        
        # Suggest exploring these directories
        for directory in list(directories)[:3]:
            related.append(f"Explore more files in: {directory}")
        
        return related


class ResponseHandler:
    """Main response handler class."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.formatter = ResponseFormatter()
        self.enhancer = ResponseEnhancer()
        
        # Load templates
        self.templates = self._load_templates()
    
    def _load_templates(self) -> ResponseTemplate:
        """Load response templates from configuration."""
        templates = self.config.response.templates
        
        return ResponseTemplate(
            no_results=templates.get("no_results", 
                "Sorry, I couldn't find any documents matching your query: '{query}'"),
            single_result=templates.get("single_result", 
                "I found 1 document related to your query:"),
            multiple_results=templates.get("multiple_results", 
                "I found {count} documents related to your query:"),
            error=templates.get("error", 
                "An error occurred while processing your request: {error}"),
            result_item="• {title} ({filename})",
            summary="Search completed in {time:.2f}s with {count} results."
        )
    
    def generate_response(self, query: str, results: List[SearchResult], 
                         **kwargs) -> str:
        """
        Generate a formatted response for search results.
        
        Args:
            query: Original search query
            results: List of search results
            **kwargs: Additional formatting options
            
        Returns:
            Formatted response string
        """
        if not query:
            return self.templates.error.format(error="Empty query provided")
        
        try:
            # Determine response format
            response_format = kwargs.get('format', self.config.response.format)
            search_time = kwargs.get('search_time', 0.0)
            
            # Generate base content
            content = self._generate_base_content(query, results, **kwargs)
            
            # Format according to specified format
            if response_format == 'markdown':
                formatted_response = self.formatter.format_markdown(content, results, query, **kwargs)
            elif response_format == 'html':
                formatted_response = self.formatter.format_html(content, results, query, **kwargs)
            elif response_format == 'json':
                formatted_response = self.formatter.format_json(content, results, query, **kwargs)
            else:  # plain text
                formatted_response = self.formatter.format_plain_text(content, results, query, **kwargs)
            
            # Enhance response with additional features
            if kwargs.get('include_suggestions', False):
                suggestions = self.enhancer.add_suggestions(query, results)
                if suggestions and response_format == 'markdown':
                    formatted_response += "\n\n## Suggestions\n"
                    for suggestion in suggestions:
                        formatted_response += f"- {suggestion}\n"
            
            # Add statistics if requested
            if kwargs.get('include_stats', False) and response_format != 'json':
                stats = self.enhancer.add_statistics(results, search_time)
                if stats and response_format == 'markdown':
                    formatted_response += f"\n\n## Statistics\n"
                    formatted_response += f"- Results: {stats['total_results']}\n"
                    formatted_response += f"- Search time: {stats['search_time_seconds']}s\n"
                    formatted_response += f"- Average score: {stats['average_score']}\n"
            
            # Truncate if too long
            max_length = self.config.response.max_response_length
            if max_length > 0 and len(formatted_response) > max_length:
                formatted_response = formatted_response[:max_length] + "\n\n... (response truncated)"
            
            return formatted_response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return self.templates.error.format(error=str(e))
    
    def _generate_base_content(self, query: str, results: List[SearchResult], 
                              **kwargs) -> str:
        """Generate base content before formatting."""
        if not results:
            return self.templates.no_results.format(query=query)
        elif len(results) == 1:
            return self.templates.single_result
        else:
            return self.templates.multiple_results.format(count=len(results))
    
    def generate_summary(self, query: str, results: List[SearchResult], 
                        search_time: float) -> str:
        """Generate a brief summary of search results."""
        if not results:
            return f"No results found for '{query}' in {search_time:.2f}s"
        
        # File type distribution
        file_types = {}
        for result in results:
            ext = result.extension or "unknown"
            file_types[ext] = file_types.get(ext, 0) + 1
        
        # Most common file types
        top_types = sorted(file_types.items(), key=lambda x: x[1], reverse=True)[:3]
        type_summary = ", ".join(f"{count} {ext}" for ext, count in top_types)
        
        return (f"Found {len(results)} documents for '{query}' in {search_time:.2f}s. "
                f"Top file types: {type_summary}")
    
    def generate_error_response(self, query: str, error: Exception) -> str:
        """Generate an error response."""
        error_message = str(error)
        
        # Don't expose sensitive information
        safe_error = "An internal error occurred"
        if "not found" in error_message.lower():
            safe_error = "Resource not found"
        elif "timeout" in error_message.lower():
            safe_error = "Search timeout - try a simpler query"
        elif "permission" in error_message.lower():
            safe_error = "Access denied"
        
        return self.templates.error.format(error=safe_error)
    
    def generate_help_response(self) -> str:
        """Generate a help response with usage instructions."""
        help_text = """
# Search Bot Help

## Basic Search
- `machine learning` - Search for documents containing these terms
- `"exact phrase"` - Search for an exact phrase in quotes
- `python AND data` - Boolean search with AND, OR, NOT operators

## Advanced Search
- `title:python` - Search in specific fields (title, content, filename, tags)
- `*.py` - Wildcard search for patterns
- `machne~` - Fuzzy search (finds "machine" even with typos)

## Filters
- Search supports filtering by file type, size, and modification date
- Use the index command to add new documents: `index /path/to/documents`

## Commands
- `help` or `h` - Show this help message
- `quit`, `exit`, or `q` - Exit the search bot
- `index <path>` - Index documents from a path

## Tips
- Use specific terms for better results
- Try different keywords if you don't find what you're looking for
- Use quotes for exact phrases
- Combine terms with AND/OR for complex queries
"""
        return help_text.strip()
    
    def format_file_preview(self, file_path: str, max_lines: int = 10) -> str:
        """Generate a preview of a file's contents."""
        try:
            path = Path(file_path)
            if not path.exists():
                return "File not found"
            
            if path.stat().st_size > 1024 * 1024:  # 1MB limit
                return "File too large to preview"
            
            # Read file content
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
            except:
                return "Unable to read file content"
            
            # Generate preview
            preview_lines = lines[:max_lines]
            preview = "".join(preview_lines)
            
            if len(lines) > max_lines:
                preview += f"\n... ({len(lines) - max_lines} more lines)"
            
            return f"```\n{preview}\n```"
            
        except Exception as e:
            self.logger.error(f"Error generating file preview: {e}")
            return "Error generating preview"
    
    def validate_response(self, response: str) -> bool:
        """Validate that a response is appropriate and safe."""
        if not response or not response.strip():
            return False
        
        # Check length limits
        if len(response) > self.config.response.max_response_length * 2:
            return False
        
        # Basic content validation
        # Add more validation rules as needed
        
        return True