"""
Document Indexer Module

This module handles document indexing functionality for the search bot.
It processes various file types, extracts text content, and builds searchable
indexes using Whoosh or other search engines.
"""

import os
import logging
import hashlib
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Any, Iterator, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import fnmatch

# Text processing imports
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Optional spaCy import
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

# File processing imports
import PyPDF2
from docx import Document as DocxDocument
from bs4 import BeautifulSoup
import markdown

# Search engine imports
from whoosh.index import create_index, open_dir, exists_in
from whoosh.fields import Schema, TEXT, ID, DATETIME, NUMERIC, KEYWORD
from whoosh.analysis import StandardAnalyzer, StemmingAnalyzer
from whoosh.writing import IndexWriter

# Configuration
from ..utils.config import get_config


def _default_tags():
    return []

def _default_metadata():
    return {}

@dataclass
class DocumentInfo:
    """Information about an indexed document."""
    path: str
    filename: str
    extension: str
    size: int
    modified_time: datetime
    content_hash: str
    title: Optional[str] = None
    content: str = ""
    word_count: int = 0
    language: str = "en"
    tags: List[str] = field(default_factory=_default_tags)
    metadata: Dict[str, Any] = field(default_factory=_default_metadata)


class DocumentProcessor:
    """Handles processing of different document types."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLP tools
        self._setup_nlp()
    
    def _setup_nlp(self):
        """Initialize NLP libraries and download required data."""
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            
            # Initialize stemmer
            self.stemmer = PorterStemmer()
            
            # Initialize stopwords
            self.stop_words = set(stopwords.words('english'))
            if self.config.nlp.custom_stopwords:
                self.stop_words.update(self.config.nlp.custom_stopwords)
            
            # Initialize spaCy if available
            if SPACY_AVAILABLE:
                try:
                    self.nlp = spacy.load(self.config.nlp.spacy_model)
                except OSError:
                    self.logger.warning(f"SpaCy model '{self.config.nlp.spacy_model}' not found. Install with: python -m spacy download {self.config.nlp.spacy_model}")
                    self.nlp = None
            else:
                self.logger.info("SpaCy not available. Using NLTK for text processing.")
                self.nlp = None
                
        except Exception as e:
            self.logger.error(f"Error setting up NLP tools: {e}")
            self.stemmer = None
            self.stop_words = set()
            self.nlp = None
    
    def process_file(self, file_path: Path) -> Optional[DocumentInfo]:
        """
        Process a single file and extract its content.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            DocumentInfo object or None if processing failed
        """
        try:
            if not file_path.exists() or not file_path.is_file():
                return None
            
            # Check file size
            file_size = file_path.stat().st_size
            max_size = self.config.indexing.max_file_size_mb * 1024 * 1024
            if file_size > max_size:
                self.logger.warning(f"File too large: {file_path} ({file_size} bytes)")
                return None
            
            # Get file info
            stat = file_path.stat()
            extension = file_path.suffix.lower()
            
            # Check if file type is supported
            if extension not in self.config.indexing.supported_formats:
                return None
            
            # Extract content based on file type
            content = self._extract_content(file_path)
            if not content or not content.strip():
                return None
            
            # Create document info
            doc_info = DocumentInfo(
                path=str(file_path.absolute()),
                filename=file_path.name,
                extension=extension,
                size=file_size,
                modified_time=datetime.fromtimestamp(stat.st_mtime),
                content_hash=self._calculate_hash(content),
                content=content,
                word_count=len(word_tokenize(content)) if content else 0
            )
            
            # Extract title
            doc_info.title = self._extract_title(file_path, content)
            
            # Process content with NLP if enabled
            if self.nlp and self.config.nlp.enable_ner:
                doc_info.tags = self._extract_entities(content)
            
            # Add metadata
            doc_info.metadata = self._extract_metadata(file_path, content)
            
            return doc_info
            
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}")
            return None
    
    def _extract_content(self, file_path: Path) -> str:
        """Extract text content from various file types."""
        extension = file_path.suffix.lower()
        
        try:
            if extension == '.pdf':
                return self._extract_pdf_content(file_path)
            elif extension == '.docx':
                return self._extract_docx_content(file_path)
            elif extension in ['.html', '.htm']:
                return self._extract_html_content(file_path)
            elif extension == '.md':
                return self._extract_markdown_content(file_path)
            elif extension in ['.txt', '.py', '.js', '.css', '.json', '.xml', '.csv']:
                return self._extract_text_content(file_path)
            else:
                # Try to read as text
                return self._extract_text_content(file_path)
                
        except Exception as e:
            self.logger.error(f"Error extracting content from {file_path}: {e}")
            return ""
    
    def _extract_pdf_content(self, file_path: Path) -> str:
        """Extract text from PDF files."""
        content = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    content += page.extract_text() + "\n"
        except Exception as e:
            self.logger.error(f"Error reading PDF {file_path}: {e}")
        return content
    
    def _extract_docx_content(self, file_path: Path) -> str:
        """Extract text from DOCX files."""
        try:
            doc = DocxDocument(file_path)
            content = []
            for paragraph in doc.paragraphs:
                content.append(paragraph.text)
            return "\n".join(content)
        except Exception as e:
            self.logger.error(f"Error reading DOCX {file_path}: {e}")
            return ""
    
    def _extract_html_content(self, file_path: Path) -> str:
        """Extract text from HTML files."""
        try:
            with open(file_path, 'r', encoding=self.config.indexing.text_encoding, errors='ignore') as file:
                html_content = file.read()
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text content
                text = soup.get_text()
                
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                return text
        except Exception as e:
            self.logger.error(f"Error reading HTML {file_path}: {e}")
            return ""
    
    def _extract_markdown_content(self, file_path: Path) -> str:
        """Extract text from Markdown files."""
        try:
            with open(file_path, 'r', encoding=self.config.indexing.text_encoding, errors='ignore') as file:
                md_content = file.read()
                
                # Convert markdown to HTML then extract text
                html = markdown.markdown(md_content)
                soup = BeautifulSoup(html, 'html.parser')
                return soup.get_text()
        except Exception as e:
            self.logger.error(f"Error reading Markdown {file_path}: {e}")
            return ""
    
    def _extract_text_content(self, file_path: Path) -> str:
        """Extract text from plain text files."""
        try:
            with open(file_path, 'r', encoding=self.config.indexing.text_encoding, errors='ignore') as file:
                return file.read()
        except Exception as e:
            self.logger.error(f"Error reading text file {file_path}: {e}")
            return ""
    
    def _extract_title(self, file_path: Path, content: str) -> str:
        """Extract title from document content or filename."""
        # Try to extract title from content
        if content:
            lines = content.strip().split('\n')
            first_line = lines[0].strip() if lines else ""
            
            # Check if first line looks like a title
            if first_line and len(first_line) < 100 and not first_line.endswith('.'):
                return first_line
        
        # Fallback to filename without extension
        return file_path.stem
    
    def _extract_entities(self, content: str) -> List[str]:
        """Extract named entities from content using spaCy."""
        if not self.nlp or not content:
            return []
        
        try:
            doc = self.nlp(content[:10000])  # Limit content length for performance
            entities = []
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT']:
                    entities.append(ent.text)
            return list(set(entities))  # Remove duplicates
        except Exception as e:
            self.logger.error(f"Error extracting entities: {e}")
            return []
    
    def _extract_metadata(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Extract metadata from file and content."""
        metadata = {}
        
        # File metadata
        stat = file_path.stat()
        metadata['file_type'] = file_path.suffix.lower()
        metadata['created_time'] = datetime.fromtimestamp(stat.st_ctime).isoformat()
        metadata['accessed_time'] = datetime.fromtimestamp(stat.st_atime).isoformat()
        
        # Content metadata
        if content:
            metadata['line_count'] = len(content.split('\n'))
            metadata['char_count'] = len(content)
            metadata['paragraph_count'] = len([p for p in content.split('\n\n') if p.strip()])
        
        # MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type:
            metadata['mime_type'] = mime_type
        
        return metadata
    
    def _calculate_hash(self, content: str) -> str:
        """Calculate SHA-256 hash of content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()


class DocumentIndexer:
    """Main document indexer class."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.processor = DocumentProcessor()
        
        # Initialize search index
        self._init_index()
    
    def _init_index(self):
        """Initialize the Whoosh search index."""
        try:
            # Create index directory if it doesn't exist
            index_dir = Path(self.config.paths.index_dir)
            index_dir.mkdir(parents=True, exist_ok=True)
            
            # Define index schema
            analyzer = StemmingAnalyzer() if self.config.search.stemming else StandardAnalyzer()
            
            self.schema = Schema(
                path=ID(stored=True, unique=True),
                filename=TEXT(stored=True),
                title=TEXT(stored=True, analyzer=analyzer),
                content=TEXT(analyzer=analyzer),
                extension=KEYWORD(stored=True),
                size=NUMERIC(stored=True),
                modified_time=DATETIME(stored=True),
                content_hash=ID(stored=True),
                word_count=NUMERIC(stored=True),
                tags=KEYWORD(stored=True, commas=True),
                metadata=TEXT(stored=True)  # JSON-serialized metadata
            )
            
            # Create or open index
            if exists_in(str(index_dir)):
                from whoosh.index import open_dir
                self.index = open_dir(str(index_dir))
                self.logger.info(f"Opened existing search index at {index_dir}")
            else:
                self.index = create_index(self.schema, str(index_dir))
                self.logger.info(f"Created new search index at {index_dir}")
                
        except Exception as e:
            self.logger.error(f"Error initializing search index: {e}")
            raise
    
    def index_documents(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Index documents from a file or directory.
        
        Args:
            path: Path to file or directory to index
            
        Returns:
            Dictionary with indexing results
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        
        start_time = datetime.now()
        results = {
            'total_files': 0,
            'indexed_files': 0,
            'skipped_files': 0,
            'errors': 0,
            'start_time': start_time.isoformat(),
            'files_processed': []
        }
        
        try:
            writer = self.index.writer()
            
            if path.is_file():
                # Index single file
                files_to_process = [path]
            else:
                # Index directory
                files_to_process = self._get_files_to_index(path)
            
            results['total_files'] = len(files_to_process)
            
            # Process files in batches
            batch_size = self.config.indexing.batch_size
            for i in range(0, len(files_to_process), batch_size):
                batch = files_to_process[i:i + batch_size]
                
                for file_path in batch:
                    try:
                        if self._should_skip_file(file_path):
                            results['skipped_files'] += 1
                            continue
                        
                        # Check if file needs updating
                        if self._is_file_up_to_date(file_path):
                            results['skipped_files'] += 1
                            continue
                        
                        # Process document
                        doc_info = self.processor.process_file(file_path)
                        if doc_info:
                            self._add_to_index(writer, doc_info)
                            results['indexed_files'] += 1
                            results['files_processed'].append({
                                'path': str(file_path),
                                'status': 'indexed',
                                'word_count': doc_info.word_count
                            })
                        else:
                            results['skipped_files'] += 1
                            results['files_processed'].append({
                                'path': str(file_path),
                                'status': 'skipped',
                                'reason': 'processing_failed'
                            })
                    
                    except Exception as e:
                        results['errors'] += 1
                        self.logger.error(f"Error indexing {file_path}: {e}")
                        results['files_processed'].append({
                            'path': str(file_path),
                            'status': 'error',
                            'error': str(e)
                        })
                
                # Commit batch
                writer.commit()
                writer = self.index.writer()
                
                self.logger.info(f"Processed batch {i//batch_size + 1}, "
                              f"indexed: {results['indexed_files']}, "
                              f"skipped: {results['skipped_files']}")
            
            writer.commit()
            
            # Update results
            end_time = datetime.now()
            results['end_time'] = end_time.isoformat()
            results['duration_seconds'] = (end_time - start_time).total_seconds()
            
            self.logger.info(f"Indexing completed: {results['indexed_files']} files indexed, "
                          f"{results['skipped_files']} skipped, {results['errors']} errors")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during indexing: {e}")
            raise
    
    def _get_files_to_index(self, directory: Path) -> List[Path]:
        """Get list of files to index from directory."""
        files = []
        
        if self.config.indexing.recursive_scan:
            pattern = "**/*"
        else:
            pattern = "*"
        
        for file_path in directory.glob(pattern):
            if file_path.is_file():
                files.append(file_path)
        
        return files
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped."""
        # Skip hidden files if configured
        if self.config.indexing.ignore_hidden_files and file_path.name.startswith('.'):
            return True
        
        # Check exclude patterns
        for pattern in self.config.indexing.exclude_patterns:
            if fnmatch.fnmatch(file_path.name, pattern) or fnmatch.fnmatch(str(file_path), pattern):
                return True
        
        # Check file extension
        if file_path.suffix.lower() not in self.config.indexing.supported_formats:
            return True
        
        return False
    
    def _is_file_up_to_date(self, file_path: Path) -> bool:
        """Check if file is already indexed and up to date."""
        try:
            with self.index.searcher() as searcher:
                doc = searcher.document(path=str(file_path.absolute()))
                if doc:
                    # Check if file has been modified since last index
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    indexed_mtime = doc['modified_time']
                    return file_mtime <= indexed_mtime
                return False
        except Exception:
            return False
    
    def _add_to_index(self, writer: IndexWriter, doc_info: DocumentInfo):
        """Add document to search index."""
        writer.update_document(
            path=doc_info.path,
            filename=doc_info.filename,
            title=doc_info.title or "",
            content=doc_info.content,
            extension=doc_info.extension,
            size=doc_info.size,
            modified_time=doc_info.modified_time,
            content_hash=doc_info.content_hash,
            word_count=doc_info.word_count,
            tags=",".join(doc_info.tags) if doc_info.tags else "",
            metadata=json.dumps(doc_info.metadata)
        )
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search the index for documents matching the query.
        
        Args:
            query: Search query string
            
        Returns:
            List of matching documents with metadata
        """
        try:
            with self.index.searcher() as searcher:
                from whoosh.qparser import MultifieldParser
                
                # Create query parser
                parser = MultifieldParser(
                    ["title", "content", "filename", "tags"],
                    schema=self.index.schema
                )
                
                # Parse query
                parsed_query = parser.parse(query)
                
                # Execute search
                results = searcher.search(
                    parsed_query,
                    limit=self.config.search.max_results
                )
                
                # Format results
                formatted_results = []
                for result in results:
                    doc = {
                        'path': result['path'],
                        'filename': result['filename'],
                        'title': result['title'],
                        'extension': result['extension'],
                        'size': result['size'],
                        'modified_time': result['modified_time'].isoformat() if result['modified_time'] else None,
                        'word_count': result['word_count'],
                        'tags': result['tags'].split(',') if result['tags'] else [],
                        'score': result.score,
                        'metadata': json.loads(result['metadata']) if result['metadata'] else {}
                    }
                    
                    # Add content snippet if highlighting is enabled
                    if self.config.search.highlight_matches:
                        doc['snippet'] = result.highlights('content', maxchars=self.config.search.snippet_length)
                    
                    formatted_results.append(doc)
                
                return formatted_results
                
        except Exception as e:
            self.logger.error(f"Error searching index: {e}")
            return []
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the search index."""
        try:
            with self.index.searcher() as searcher:
                doc_count = searcher.doc_count()
                field_names = list(self.index.schema.names())
                
                return {
                    'document_count': doc_count,
                    'field_names': field_names,
                    'index_path': self.config.paths.index_dir,
                    'schema_version': getattr(self.index, 'schema_version', 'unknown')
                }
        except Exception as e:
            self.logger.error(f"Error getting index stats: {e}")
            return {}
    
    def clear_index(self) -> bool:
        """Clear all documents from the index."""
        try:
            writer = self.index.writer()
            writer.commit(mergetype=self.index.writer.CLEAR)
            self.logger.info("Search index cleared")
            return True
        except Exception as e:
            self.logger.error(f"Error clearing index: {e}")
            return False