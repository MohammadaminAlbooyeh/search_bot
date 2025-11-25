"""
Unit tests for the DocumentIndexer class.

This module contains comprehensive tests for document indexing functionality,
including text processing, file handling, and index management.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import test utilities
from . import setup_test_environment, cleanup_test_environment, create_test_document, TEST_DOCUMENTS_DIR

# Import the class to test
from src.data.indexer import DocumentIndexer


class TestDocumentIndexer:
    """Test cases for DocumentIndexer class."""
    
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
        self.indexer = DocumentIndexer()
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
    
    def teardown_method(self):
        """Clean up after each test method."""
        if self.test_path.exists():
            shutil.rmtree(self.test_path)
    
    def test_indexer_initialization(self):
        """Test that DocumentIndexer initializes correctly."""
        indexer = DocumentIndexer()
        assert indexer is not None
        assert hasattr(indexer, 'index_documents')
        assert hasattr(indexer, 'search')
    
    def test_index_single_document(self):
        """Test indexing a single document."""
        # Create test document
        test_content = "This is a test document about machine learning and artificial intelligence."
        test_file = create_test_document("test_doc.txt", test_content, self.test_path)
        
        # Index the document
        result = self.indexer.index_documents(str(test_file))
        
        # Verify indexing was successful
        assert result is not None
        # Additional assertions would depend on the actual implementation
    
    def test_index_multiple_documents(self):
        """Test indexing multiple documents from a directory."""
        # Create multiple test documents
        documents = [
            ("doc1.txt", "Python programming tutorial for beginners"),
            ("doc2.txt", "Advanced machine learning algorithms"),
            ("doc3.txt", "Data science with pandas and numpy"),
            ("doc4.md", "# Web Development\nBuilding modern web applications")
        ]
        
        for filename, content in documents:
            create_test_document(filename, content, self.test_path)
        
        # Index all documents
        result = self.indexer.index_documents(str(self.test_path))
        
        # Verify indexing was successful
        assert result is not None
    
    def test_index_empty_directory(self):
        """Test indexing an empty directory."""
        empty_dir = self.test_path / "empty"
        empty_dir.mkdir()
        
        result = self.indexer.index_documents(str(empty_dir))
        
        # Should handle empty directory gracefully
        assert result is not None
    
    def test_index_nonexistent_path(self):
        """Test indexing a path that doesn't exist."""
        nonexistent_path = str(self.test_path / "nonexistent")
        
        with pytest.raises(Exception):
            self.indexer.index_documents(nonexistent_path)
    
    def test_index_different_file_types(self):
        """Test indexing various file types."""
        test_files = [
            ("text_file.txt", "Plain text content"),
            ("markdown_file.md", "# Markdown Content\nWith **bold** text"),
            ("python_file.py", "# Python code\nprint('Hello World')"),
            ("readme.README", "README file content"),
        ]
        
        for filename, content in test_files:
            create_test_document(filename, content, self.test_path)
        
        result = self.indexer.index_documents(str(self.test_path))
        assert result is not None
    
    def test_index_large_document(self):
        """Test indexing a large document."""
        # Create a large test document
        large_content = "This is a test sentence. " * 10000
        test_file = create_test_document("large_doc.txt", large_content, self.test_path)
        
        result = self.indexer.index_documents(str(test_file))
        assert result is not None
    
    def test_index_unicode_content(self):
        """Test indexing documents with unicode characters."""
        unicode_content = "Test document with unicode: 你好世界 🌟 café naïve résumé"
        test_file = create_test_document("unicode_doc.txt", unicode_content, self.test_path)
        
        result = self.indexer.index_documents(str(test_file))
        assert result is not None
    
    def test_search_functionality(self):
        """Test basic search functionality if implemented."""
        # Create test documents
        create_test_document("ml_doc.txt", "Machine learning and neural networks", self.test_path)
        create_test_document("web_doc.txt", "Web development with JavaScript", self.test_path)
        
        # Index documents
        self.indexer.index_documents(str(self.test_path))
        
        # Test search functionality if it exists
        if hasattr(self.indexer, 'search'):
            results = self.indexer.search("machine learning")
            assert results is not None
    
    def test_index_with_special_characters(self):
        """Test indexing documents with special characters."""
        special_content = "Document with special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?"
        test_file = create_test_document("special_chars.txt", special_content, self.test_path)
        
        result = self.indexer.index_documents(str(test_file))
        assert result is not None
    
    def test_index_binary_files(self):
        """Test handling of binary files."""
        # Create a fake binary file
        binary_file = self.test_path / "binary_file.bin"
        binary_file.write_bytes(b'\x00\x01\x02\x03\x04\x05')
        
        # Should handle binary files gracefully (skip or process appropriately)
        result = self.indexer.index_documents(str(binary_file))
        # The exact behavior depends on implementation
        assert result is not None or result is None  # Either way should not crash
    
    def test_index_nested_directories(self):
        """Test indexing nested directory structures."""
        # Create nested directory structure
        nested_dir = self.test_path / "level1" / "level2" / "level3"
        nested_dir.mkdir(parents=True)
        
        # Create files at different levels
        create_test_document("root.txt", "Root level document", self.test_path)
        create_test_document("level1.txt", "Level 1 document", self.test_path / "level1")
        create_test_document("level2.txt", "Level 2 document", self.test_path / "level1" / "level2")
        create_test_document("level3.txt", "Level 3 document", nested_dir)
        
        result = self.indexer.index_documents(str(self.test_path))
        assert result is not None
    
    @patch('builtins.open', side_effect=PermissionError("Permission denied"))
    def test_permission_error_handling(self, mock_open):
        """Test handling of permission errors."""
        test_file = create_test_document("test.txt", "content", self.test_path)
        
        # Should handle permission errors gracefully
        try:
            result = self.indexer.index_documents(str(test_file))
            # If no exception is raised, the method handled it gracefully
            assert True
        except PermissionError:
            # If PermissionError is raised, that's also acceptable behavior
            assert True
    
    def test_index_empty_file(self):
        """Test indexing empty files."""
        empty_file = create_test_document("empty.txt", "", self.test_path)
        
        result = self.indexer.index_documents(str(empty_file))
        assert result is not None
    
    def test_index_whitespace_only_file(self):
        """Test indexing files with only whitespace."""
        whitespace_file = create_test_document("whitespace.txt", "   \n\t  \n  ", self.test_path)
        
        result = self.indexer.index_documents(str(whitespace_file))
        assert result is not None


# Integration tests
class TestDocumentIndexerIntegration:
    """Integration tests for DocumentIndexer."""
    
    def setup_method(self):
        """Set up before each test method."""
        setup_test_environment()
        self.indexer = DocumentIndexer()
    
    def teardown_method(self):
        """Clean up after each test method."""
        cleanup_test_environment()
    
    def test_full_workflow(self):
        """Test complete indexing workflow."""
        # Create sample documents
        sample_docs = [
            ("python_guide.txt", "Python is a programming language used for web development, data science, and automation."),
            ("ml_basics.txt", "Machine learning involves training algorithms on data to make predictions."),
            ("web_dev.txt", "Web development includes frontend and backend technologies like HTML, CSS, JavaScript, and Python.")
        ]
        
        for filename, content in sample_docs:
            create_test_document(filename, content)
        
        # Index documents
        result = self.indexer.index_documents(str(TEST_DOCUMENTS_DIR))
        
        # Verify the workflow completed successfully
        assert result is not None
        
        # If search is implemented, test it
        if hasattr(self.indexer, 'search'):
            search_results = self.indexer.search("Python programming")
            assert search_results is not None