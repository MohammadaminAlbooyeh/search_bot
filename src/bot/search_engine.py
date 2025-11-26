"""
Search Engine Module

This module provides intelligent search functionality for the search bot.
It handles query processing, ranking, filtering, and result formatting
with support for various search modes and techniques.
"""

import logging
import re
import time
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# Search engine imports
from whoosh.index import open_dir, exists_in
from whoosh.qparser import MultifieldParser, QueryParser, OrGroup, AndGroup
from whoosh.query import Query, Term, And, Or, Not, Phrase, Wildcard, FuzzyTerm
from whoosh.scoring import BM25F, TF_IDF, Frequency
from whoosh.analysis import StemmingAnalyzer, StandardAnalyzer
from whoosh.highlight import SimpleFragmenter, ContextFragmenter, highlight
from whoosh.collectors import TimeLimitCollector, TermsCollector
import whoosh.qparser as qparser

# NLP imports
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Optional spaCy import
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

# Configuration
from ..utils.config import get_config
from ..data.indexer import DocumentIndexer


@dataclass
class SearchResult:
    """Represents a single search result."""
    path: str
    filename: str
    title: str
    content_snippet: str
    score: float
    extension: str
    size: int
    modified_time: str
    word_count: int
    tags: List[str]
    metadata: Dict[str, Any]
    highlights: Dict[str, str] = field(default_factory=dict)
    rank: int = 0


def _default_search_fields():
    return ["title", "content", "filename", "tags"]

@dataclass
class SearchQuery:
    """Represents a processed search query."""
    original: str
    processed: str
    terms: List[str]
    phrases: List[str]
    filters: Dict[str, Any]
    search_type: str = "standard"  # standard, fuzzy, wildcard, phrase, boolean
    fields: List[str] = field(default_factory=_default_search_fields)


@dataclass
class SearchStats:
    """Search operation statistics."""
    query: str
    total_results: int
    search_time: float
    processing_time: float
    timestamp: datetime
    filters_applied: Dict[str, Any]


class QueryProcessor:
    """Handles query parsing and processing."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLP tools
        self._setup_nlp()
        
        # Query patterns
        self.phrase_pattern = re.compile(r'"([^"]*)"')
        self.field_pattern = re.compile(r'(\w+):(\S+)')
        self.wildcard_pattern = re.compile(r'\*|\?')
        self.boolean_pattern = re.compile(r'\b(AND|OR|NOT)\b', re.IGNORECASE)
    
    def _setup_nlp(self):
        """Initialize NLP components."""
        try:
            # Download NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            
            # Initialize stemmer and stopwords
            self.stemmer = PorterStemmer()
            self.stop_words = set(stopwords.words('english'))
            if self.config.nlp.custom_stopwords:
                self.stop_words.update(self.config.nlp.custom_stopwords)
            
            # Initialize spaCy if available
            if SPACY_AVAILABLE:
                try:
                    self.nlp = spacy.load(self.config.nlp.spacy_model)
                except OSError:
                    self.logger.warning("SpaCy model not available, using basic processing")
                    self.nlp = None
            else:
                self.logger.info("SpaCy not available. Using NLTK for query processing.")
                self.nlp = None
                
        except Exception as e:
            self.logger.error(f"Error setting up NLP: {e}")
            self.stemmer = None
            self.stop_words = set()
            self.nlp = None
    
    def process_query(self, query: str) -> SearchQuery:
        """
        Process and analyze a search query.
        
        Args:
            query: Raw search query string
            
        Returns:
            Processed SearchQuery object
        """
        if not query or not query.strip():
            return SearchQuery(
                original="",
                processed="",
                terms=[],
                phrases=[],
                filters={}
            )
        
        original_query = query.strip()
        
        # Extract phrases (quoted text)
        phrases = self.phrase_pattern.findall(query)
        
        # Extract field filters (field:value)
        filters = {}
        field_matches = self.field_pattern.findall(query)
        for field, value in field_matches:
            filters[field] = value
        
        # Remove phrases and field filters from query for term extraction
        processed_query = query
        for phrase in phrases:
            processed_query = processed_query.replace(f'"{phrase}"', '')
        for field, value in field_matches:
            processed_query = processed_query.replace(f'{field}:{value}', '')
        
        # Determine search type
        search_type = self._determine_search_type(original_query)
        
        # Extract and process terms
        terms = self._extract_terms(processed_query)
        
        # Clean up processed query
        processed_query = ' '.join(terms) if terms else processed_query.strip()
        
        return SearchQuery(
            original=original_query,
            processed=processed_query,
            terms=terms,
            phrases=phrases,
            filters=filters,
            search_type=search_type
        )
    
    def _determine_search_type(self, query: str) -> str:
        """Determine the type of search based on query patterns."""
        if self.phrase_pattern.search(query):
            return "phrase"
        elif self.wildcard_pattern.search(query):
            return "wildcard"
        elif self.boolean_pattern.search(query):
            return "boolean"
        elif self.config.search.fuzzy_matching and len(query.split()) == 1:
            return "fuzzy"
        else:
            return "standard"
    
    def _extract_terms(self, query: str) -> List[str]:
        """Extract and process search terms."""
        if not query.strip():
            return []
        
        # Tokenize
        try:
            tokens = word_tokenize(query.lower())
        except:
            tokens = query.lower().split()
        
        # Filter terms
        terms = []
        for token in tokens:
            # Skip if too short or too long
            if len(token) < self.config.indexing.min_word_length:
                continue
            if len(token) > self.config.indexing.max_word_length:
                continue
            
            # Skip stopwords if configured
            if self.config.nlp.remove_stopwords and token in self.stop_words:
                continue
            
            # Skip punctuation
            if not re.match(r'\w+', token):
                continue
            
            # Apply stemming if configured
            if self.stemmer and self.config.nlp.enable_lemmatization:
                token = self.stemmer.stem(token)
            
            terms.append(token)
        
        return terms


class SearchCache:
    """Simple in-memory cache for search results."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached result if still valid."""
        with self.lock:
            if key in self.cache:
                result, timestamp = self.cache[key]
                if datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                    return result
                else:
                    del self.cache[key]
            return None
    
    def set(self, key: str, value: Any):
        """Cache a result."""
        with self.lock:
            # Remove oldest entries if cache is full
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]
            
            self.cache[key] = (value, datetime.now())
    
    def clear(self):
        """Clear all cached results."""
        with self.lock:
            self.cache.clear()


class SearchEngine:
    """Main search engine class."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.query_processor = QueryProcessor()
        
        # Initialize cache if enabled
        if self.config.performance.cache_enabled:
            self.cache = SearchCache(
                max_size=1000,
                ttl_seconds=self.config.performance.cache_ttl_seconds
            )
        else:
            self.cache = None
        
        # Initialize search index
        self._init_search_index()
        
        # Search statistics
        self.search_stats: List[SearchStats] = []
        self.stats_lock = threading.Lock()
    
    def _init_search_index(self):
        """Initialize connection to the search index."""
        try:
            index_dir = self.config.paths.index_dir
            if exists_in(index_dir):
                self.index = open_dir(index_dir)
                self.logger.info(f"Connected to search index at {index_dir}")
            else:
                self.logger.warning(f"Search index not found at {index_dir}. Creating new index...")
                # Create new index using DocumentIndexer
                indexer = DocumentIndexer()
                self.index = indexer.index
        except Exception as e:
            self.logger.error(f"Error initializing search index: {e}")
            raise
    
    def search(self, query: str, **kwargs) -> List[SearchResult]:
        """
        Perform a search query.
        
        Args:
            query: Search query string
            **kwargs: Additional search parameters
            
        Returns:
            List of SearchResult objects
        """
        if not query or not query.strip():
            return []
        
        start_time = time.time()
        
        # Check cache first
        cache_key = self._generate_cache_key(query, kwargs)
        if self.cache:
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                self.logger.debug(f"Cache hit for query: {query[:50]}...")
                return cached_result
        
        try:
            # Process query
            processed_query = self.query_processor.process_query(query)
            
            # Validate query
            if not self._validate_query(processed_query):
                return []
            
            # Execute search
            results = self._execute_search(processed_query, **kwargs)
            
            # Post-process results
            results = self._post_process_results(results, processed_query, **kwargs)
            
            # Cache results
            if self.cache and results:
                self.cache.set(cache_key, results)
            
            # Record statistics
            search_time = time.time() - start_time
            self._record_search_stats(processed_query, results, search_time, kwargs)
            
            self.logger.info(f"Search completed: '{query[:50]}...' returned {len(results)} results in {search_time:.3f}s")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during search: {e}")
            return []
    
    def _generate_cache_key(self, query: str, kwargs: Dict[str, Any]) -> str:
        """Generate a cache key for the search."""
        key_data = {
            'query': query,
            'kwargs': sorted(kwargs.items())
        }
        return str(hash(json.dumps(key_data, sort_keys=True)))
    
    def _validate_query(self, query: SearchQuery) -> bool:
        """Validate the processed query."""
        # Check query length
        if len(query.original) > self.config.security.max_query_length:
            self.logger.warning(f"Query too long: {len(query.original)} characters")
            return False
        
        # Check if query has any meaningful content
        if not query.terms and not query.phrases:
            return False
        
        return True
    
    def _execute_search(self, query: SearchQuery, **kwargs) -> List[Dict[str, Any]]:
        """Execute the actual search against the index."""
        try:
            with self.index.searcher() as searcher:
                # Build Whoosh query
                whoosh_query = self._build_whoosh_query(query, searcher.schema)
                
                # Configure search parameters
                limit = kwargs.get('limit', self.config.search.max_results)
                
                # Set up scorer
                if self.config.search.engine_type == "bm25":
                    scorer = BM25F()
                else:
                    scorer = TF_IDF()
                
                # Execute search with timeout
                try:
                    if self.config.performance.search_timeout_seconds > 0:
                        collector = TimeLimitCollector(
                            searcher.collector(limit=limit),
                            timelimit=self.config.performance.search_timeout_seconds
                        )
                        results = searcher.search_with_collector(whoosh_query, collector)
                    else:
                        results = searcher.search(whoosh_query, limit=limit, scored=scorer)
                    
                except FuturesTimeoutError:
                    self.logger.warning("Search timed out")
                    return []
                
                # Convert results
                formatted_results = []
                for i, result in enumerate(results):
                    doc = {
                        'path': result.get('path', ''),
                        'filename': result.get('filename', ''),
                        'title': result.get('title', ''),
                        'extension': result.get('extension', ''),
                        'size': result.get('size', 0),
                        'modified_time': result.get('modified_time', ''),
                        'word_count': result.get('word_count', 0),
                        'tags': result.get('tags', '').split(',') if result.get('tags') else [],
                        'score': result.score if hasattr(result, 'score') else 0.0,
                        'rank': i + 1,
                        'metadata': json.loads(result.get('metadata', '{}')) if result.get('metadata') else {}
                    }
                    
                    # Add highlights if enabled
                    if self.config.search.highlight_matches:
                        doc['highlights'] = self._extract_highlights(result, query)
                        doc['content_snippet'] = doc['highlights'].get('content', '')[:self.config.search.snippet_length]
                    else:
                        doc['content_snippet'] = ''
                    
                    # Apply minimum score filter
                    if doc['score'] >= self.config.search.min_score:
                        formatted_results.append(doc)
                
                return formatted_results
                
        except Exception as e:
            self.logger.error(f"Error executing search: {e}")
            return []
    
    def _build_whoosh_query(self, query: SearchQuery, schema) -> Query:
        """Build a Whoosh query from processed search query."""
        if query.search_type == "phrase" and query.phrases:
            # Phrase search
            phrase_queries = []
            for phrase in query.phrases:
                phrase_queries.append(Phrase("content", phrase.split()))
            return Or(phrase_queries)
        
        elif query.search_type == "boolean":
            # Boolean search - let Whoosh parser handle it
            parser = MultifieldParser(query.fields, schema)
            try:
                return parser.parse(query.original)
            except:
                # Fall back to standard search
                return self._build_standard_query(query, schema)
        
        elif query.search_type == "wildcard":
            # Wildcard search
            wildcard_queries = []
            for term in query.terms:
                if '*' in term or '?' in term:
                    wildcard_queries.append(Wildcard("content", term))
                else:
                    wildcard_queries.append(Term("content", term))
            return Or(wildcard_queries)
        
        elif query.search_type == "fuzzy":
            # Fuzzy search
            fuzzy_queries = []
            for term in query.terms:
                fuzzy_queries.append(FuzzyTerm("content", term))
            return Or(fuzzy_queries)
        
        else:
            # Standard search
            return self._build_standard_query(query, schema)
    
    def _build_standard_query(self, query: SearchQuery, schema) -> Query:
        """Build a standard multi-field query."""
        if not query.terms:
            return Term("content", "")
        
        # Create term queries for each field
        field_queries = []
        for field in query.fields:
            if field in schema:
                term_queries = []
                for term in query.terms:
                    term_queries.append(Term(field, term))
                if term_queries:
                    field_queries.append(Or(term_queries))
        
        # Combine field queries
        if field_queries:
            return Or(field_queries)
        else:
            # Fallback to content-only search
            term_queries = [Term("content", term) for term in query.terms]
            return Or(term_queries)
    
    def _extract_highlights(self, result, query: SearchQuery) -> Dict[str, str]:
        """Extract highlighted snippets from search result."""
        highlights = {}
        
        try:
            # Highlight content
            if hasattr(result, 'highlights'):
                content_highlight = result.highlights('content', maxchars=self.config.search.snippet_length)
                if content_highlight:
                    highlights['content'] = content_highlight
            
            # Highlight title
            if hasattr(result, 'highlights'):
                title_highlight = result.highlights('title', maxchars=100)
                if title_highlight:
                    highlights['title'] = title_highlight
                    
        except Exception as e:
            self.logger.debug(f"Error extracting highlights: {e}")
        
        return highlights
    
    def _post_process_results(self, results: List[Dict[str, Any]], query: SearchQuery, **kwargs) -> List[SearchResult]:
        """Post-process search results."""
        processed_results = []
        
        for result_data in results:
            # Create SearchResult object
            search_result = SearchResult(
                path=result_data['path'],
                filename=result_data['filename'],
                title=result_data['title'],
                content_snippet=result_data['content_snippet'],
                score=result_data['score'],
                extension=result_data['extension'],
                size=result_data['size'],
                modified_time=result_data['modified_time'],
                word_count=result_data['word_count'],
                tags=result_data['tags'],
                metadata=result_data['metadata'],
                highlights=result_data.get('highlights', {}),
                rank=result_data['rank']
            )
            
            processed_results.append(search_result)
        
        # Apply additional filters
        processed_results = self._apply_filters(processed_results, query, **kwargs)
        
        # Sort results
        processed_results = self._sort_results(processed_results, **kwargs)
        
        return processed_results
    
    def _apply_filters(self, results: List[SearchResult], query: SearchQuery, **kwargs) -> List[SearchResult]:
        """Apply additional filters to results."""
        filtered_results = results
        
        # Apply file type filter
        if 'file_types' in kwargs:
            file_types = kwargs['file_types']
            filtered_results = [r for r in filtered_results if r.extension in file_types]
        
        # Apply size filter
        if 'max_size' in kwargs:
            max_size = kwargs['max_size']
            filtered_results = [r for r in filtered_results if r.size <= max_size]
        
        # Apply date filter
        if 'modified_after' in kwargs:
            modified_after = kwargs['modified_after']
            filtered_results = [r for r in filtered_results if r.modified_time >= modified_after]
        
        return filtered_results
    
    def _sort_results(self, results: List[SearchResult], **kwargs) -> List[SearchResult]:
        """Sort search results."""
        sort_by = kwargs.get('sort_by', 'relevance')
        reverse = kwargs.get('reverse', True)
        
        if sort_by == 'relevance':
            results.sort(key=lambda x: x.score, reverse=reverse)
        elif sort_by == 'filename':
            results.sort(key=lambda x: x.filename.lower(), reverse=reverse)
        elif sort_by == 'size':
            results.sort(key=lambda x: x.size, reverse=reverse)
        elif sort_by == 'modified_time':
            results.sort(key=lambda x: x.modified_time, reverse=reverse)
        elif sort_by == 'word_count':
            results.sort(key=lambda x: x.word_count, reverse=reverse)
        
        # Update ranks after sorting
        for i, result in enumerate(results):
            result.rank = i + 1
        
        return results
    
    def _record_search_stats(self, query: SearchQuery, results: List[SearchResult], 
                           search_time: float, kwargs: Dict[str, Any]):
        """Record search statistics."""
        with self.stats_lock:
            stats = SearchStats(
                query=query.original,
                total_results=len(results),
                search_time=search_time,
                processing_time=search_time,  # For now, same as search_time
                timestamp=datetime.now(),
                filters_applied=kwargs
            )
            
            self.search_stats.append(stats)
            
            # Keep only recent stats (last 1000 searches)
            if len(self.search_stats) > 1000:
                self.search_stats = self.search_stats[-1000:]
    
    def get_suggestions(self, partial_query: str, max_suggestions: int = 5) -> List[str]:
        """Get search suggestions based on partial query."""
        if not partial_query.strip():
            return []
        
        try:
            with self.index.searcher() as searcher:
                # Simple suggestion based on indexed terms
                suggestions = []
                
                # Get terms from the index
                reader = searcher.reader()
                for fieldname in ["title", "content", "filename"]:
                    field_terms = []
                    try:
                        field_reader = reader.field_terms(fieldname)
                        for term in field_reader:
                            if term.startswith(partial_query.lower()):
                                field_terms.append(term)
                        field_terms.sort()
                        suggestions.extend(field_terms[:max_suggestions])
                    except:
                        continue
                
                # Remove duplicates and limit
                suggestions = list(set(suggestions))[:max_suggestions]
                return suggestions
                
        except Exception as e:
            self.logger.error(f"Error getting suggestions: {e}")
            return []
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search engine statistics."""
        with self.stats_lock:
            if not self.search_stats:
                return {}
            
            total_searches = len(self.search_stats)
            avg_search_time = sum(s.search_time for s in self.search_stats) / total_searches
            avg_results = sum(s.total_results for s in self.search_stats) / total_searches
            
            return {
                'total_searches': total_searches,
                'average_search_time': avg_search_time,
                'average_results_per_search': avg_results,
                'cache_enabled': self.cache is not None,
                'recent_queries': [s.query for s in self.search_stats[-10:]]
            }
    
    def clear_cache(self):
        """Clear the search cache."""
        if self.cache:
            self.cache.clear()
            self.logger.info("Search cache cleared")
    
    def reindex(self):
        """Trigger a reindex of all documents."""
        try:
            indexer = DocumentIndexer()
            result = indexer.index_documents(self.config.paths.documents_dir)
            self.logger.info(f"Reindexing completed: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Error during reindexing: {e}")
            return None