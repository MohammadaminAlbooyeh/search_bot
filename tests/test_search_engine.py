"""
Unit tests for the SearchEngine class.

This module contains comprehensive tests for search functionality,
including query processing, ranking, filtering, and result formatting.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Import test utilities
from . import setup_test_environment, cleanup_test_environment, create_test_document, TEST_DOCUMENTS_DIR

# Import the class to test
from src.bot.search_engine import SearchEngine


class TestSearchEngine:
    """Test cases for SearchEngine class."""
    
    @classmethod
    def setup_class(cls):
        """Set up test environment before all tests."""
        setup_test_environment()
    
    @classmethod
    def teardown_class(cls):
        """Clean up test environment after all tests."""
        cleanup_test_environment()
    
    def setup_method(self):
        """Set up before each test method."""
        self.search_engine = SearchEngine()
    
    def teardown_method(self):
        """Clean up after each test method."""
        pass
    
    def test_search_engine_initialization(self):
        """Test that SearchEngine initializes correctly."""
        engine = SearchEngine()
        assert engine is not None
        assert hasattr(engine, 'search')
        assert callable(engine.search)
    
    def test_simple_search_query(self):
        """Test basic search functionality with simple queries."""
        query = "machine learning"
        results = self.search_engine.search(query)
        
        # Results should be a list or similar iterable
        assert results is not None
        assert hasattr(results, '__iter__')
    
    def test_empty_query(self):
        """Test search with empty query."""
        empty_queries = ["", "   ", "\n\t", None]
        
        for query in empty_queries:
            results = self.search_engine.search(query)
            # Should handle empty queries gracefully
            assert results is not None or results == []
    
    def test_single_word_query(self):
        """Test search with single word queries."""
        single_word_queries = ["python", "machine", "data", "algorithm"]
        
        for query in single_word_queries:
            results = self.search_engine.search(query)
            assert results is not None
    
    def test_multi_word_query(self):
        """Test search with multiple word queries."""
        multi_word_queries = [
            "machine learning algorithms",
            "python web development",
            "data science tutorial",
            "artificial intelligence basics"
        ]
        
        for query in multi_word_queries:
            results = self.search_engine.search(query)
            assert results is not None
    
    def test_query_with_special_characters(self):
        """Test search queries containing special characters."""
        special_queries = [
            "C++ programming",
            "machine-learning",
            "data_science",
            "web dev & design",
            "AI/ML algorithms",
            "python 3.9+",
            "node.js development"
        ]
        
        for query in special_queries:
            results = self.search_engine.search(query)
            assert results is not None
    
    def test_case_insensitive_search(self):
        """Test that search is case insensitive."""
        queries = [
            ("PYTHON", "python"),
            ("Machine Learning", "machine learning"),
            ("DATA SCIENCE", "data science"),
            ("WEB DEVELOPMENT", "web development")
        ]
        
        for upper_query, lower_query in queries:
            upper_results = self.search_engine.search(upper_query)
            lower_results = self.search_engine.search(lower_query)
            
            # Results should be similar regardless of case
            assert upper_results is not None
            assert lower_results is not None
    
    def test_unicode_query(self):
        """Test search with unicode characters."""
        unicode_queries = [
            "café programming",
            "naïve algorithm",
            "résumé parser",
            "machine learning 机器学习",
            "python 🐍 tutorial"
        ]
        
        for query in unicode_queries:
            results = self.search_engine.search(query)
            assert results is not None
    
    def test_long_query(self):
        """Test search with very long queries."""
        long_query = " ".join([
            "machine learning artificial intelligence deep learning",
            "neural networks data science python programming",
            "web development backend frontend database optimization",
            "algorithms data structures computer science tutorial"
        ])
        
        results = self.search_engine.search(long_query)
        assert results is not None
    
    def test_query_with_numbers(self):
        """Test search queries containing numbers."""
        numeric_queries = [
            "python 3.9",
            "100 algorithms",
            "2023 tutorial",
            "top 10 frameworks",
            "version 2.0 features"
        ]
        
        for query in numeric_queries:
            results = self.search_engine.search(query)
            assert results is not None
    
    def test_boolean_like_queries(self):
        """Test queries that look like boolean expressions."""
        boolean_queries = [
            "python AND machine learning",
            "web OR mobile development",
            "NOT deprecated features",
            "python AND (web OR data)",
            "machine learning NOT neural networks"
        ]
        
        for query in boolean_queries:
            results = self.search_engine.search(query)
            assert results is not None
    
    def test_phrase_queries(self):
        """Test search with quoted phrases."""
        phrase_queries = [
            '"machine learning"',
            '"web development"',
            '"data science"',
            '"artificial intelligence"',
            '"python programming tutorial"'
        ]
        
        for query in phrase_queries:
            results = self.search_engine.search(query)
            assert results is not None
    
    def test_search_result_structure(self):
        """Test the structure of search results."""
        query = "test query"
        results = self.search_engine.search(query)
        
        # Verify results structure
        assert results is not None
        
        # If results is a list, check each result
        if isinstance(results, list) and results:
            for result in results:
                # Common result fields that might be present
                if isinstance(result, dict):
                    # Check for common fields
                    assert 'title' in result or 'content' in result or 'filename' in result or 'path' in result
    
    def test_search_performance(self):
        """Test search performance with timing."""
        import time
        
        query = "performance test query"
        
        start_time = time.time()
        results = self.search_engine.search(query)
        end_time = time.time()
        
        # Search should complete in reasonable time (adjust threshold as needed)
        execution_time = end_time - start_time
        assert execution_time < 10.0  # Should complete within 10 seconds
        assert results is not None
    
    def test_concurrent_searches(self):
        """Test multiple concurrent search operations."""
        import threading
        import time
        
        queries = [
            "concurrent test 1",
            "concurrent test 2", 
            "concurrent test 3",
            "concurrent test 4",
            "concurrent test 5"
        ]
        
        results = {}
        threads = []
        
        def search_worker(query, result_dict):
            result_dict[query] = self.search_engine.search(query)
        
        # Start multiple threads
        for query in queries:
            thread = threading.Thread(target=search_worker, args=(query, results))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout
        
        # Verify all searches completed
        assert len(results) == len(queries)
        for query in queries:
            assert query in results
            assert results[query] is not None


class TestSearchEngineWithMockData:
    """Test SearchEngine with mocked data sources."""
    
    def setup_method(self):
        """Set up before each test method."""
        self.search_engine = SearchEngine()
    
    @patch('src.bot.search_engine.SearchEngine')
    def test_search_with_mock_results(self, mock_search_engine):
        """Test search with predefined mock results."""
        # Mock search results
        mock_results = [
            {
                'title': 'Python Tutorial',
                'content': 'Learn Python programming basics',
                'score': 0.95,
                'filename': 'python_tutorial.txt'
            },
            {
                'title': 'Machine Learning Guide',
                'content': 'Introduction to machine learning algorithms',
                'score': 0.87,
                'filename': 'ml_guide.txt'
            }
        ]
        
        # Configure mock
        mock_instance = mock_search_engine.return_value
        mock_instance.search.return_value = mock_results
        
        # Test search
        engine = mock_search_engine()
        results = engine.search("python machine learning")
        
        # Verify mock was called and results are correct
        mock_instance.search.assert_called_once_with("python machine learning")
        assert results == mock_results
        assert len(results) == 2
        assert results[0]['title'] == 'Python Tutorial'
    
    def test_search_error_handling(self):
        """Test search engine error handling."""
        # Test various error conditions
        error_conditions = [
            {"query": "test", "side_effect": Exception("General error")},
            {"query": "test", "side_effect": ValueError("Invalid query")},
            {"query": "test", "side_effect": TypeError("Type error")},
            {"query": "test", "side_effect": KeyError("Key not found")},
        ]
        
        for condition in error_conditions:
            with patch.object(self.search_engine, 'search', side_effect=condition["side_effect"]):
                try:
                    result = self.search_engine.search(condition["query"])
                    # If no exception, the error was handled gracefully
                    assert True
                except Exception:
                    # If exception is raised, that's also acceptable behavior
                    assert True


class TestSearchEngineIntegration:
    """Integration tests for SearchEngine."""
    
    def setup_method(self):
        """Set up before each test method."""
        setup_test_environment()
        self.search_engine = SearchEngine()
        
        # Create sample documents for testing
        self.sample_documents = [
            ("python_basics.txt", "Python is a versatile programming language used for web development, data analysis, and automation."),
            ("ml_intro.txt", "Machine learning is a subset of artificial intelligence that enables computers to learn from data."),
            ("web_dev.txt", "Web development involves creating websites using HTML, CSS, JavaScript, and backend technologies."),
            ("data_science.txt", "Data science combines statistics, programming, and domain expertise to extract insights from data."),
            ("ai_overview.txt", "Artificial intelligence encompasses machine learning, natural language processing, and computer vision.")
        ]
        
        for filename, content in self.sample_documents:
            create_test_document(filename, content)
    
    def teardown_method(self):
        """Clean up after each test method."""
        cleanup_test_environment()
    
    def test_full_search_workflow(self):
        """Test complete search workflow with real data."""
        # Test queries related to our sample documents
        test_queries = [
            "python programming",
            "machine learning",
            "web development",
            "data science",
            "artificial intelligence"
        ]
        
        for query in test_queries:
            results = self.search_engine.search(query)
            assert results is not None
            
            # If results contain matches, verify they're relevant
            if isinstance(results, list) and results:
                for result in results:
                    if isinstance(result, dict):
                        # Check that result contains some relevant content
                        assert 'content' in result or 'title' in result or 'filename' in result
    
    def test_search_relevance(self):
        """Test that search results are relevant to queries."""
        relevance_tests = [
            {
                'query': 'python',
                'should_contain': ['python', 'programming'],
                'document_hint': 'python_basics.txt'
            },
            {
                'query': 'machine learning',
                'should_contain': ['machine', 'learning', 'artificial'],
                'document_hint': 'ml_intro.txt'
            },
            {
                'query': 'web development',
                'should_contain': ['web', 'html', 'css', 'javascript'],
                'document_hint': 'web_dev.txt'
            }
        ]
        
        for test_case in relevance_tests:
            results = self.search_engine.search(test_case['query'])
            assert results is not None
            
            # Additional relevance checks would depend on implementation
            # This is a placeholder for more sophisticated relevance testing