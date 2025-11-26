"""
Configuration Management Module

This module handles loading, validating, and managing configuration settings
for the search bot application. It supports JSON config files, environment
variables, and provides type-safe access to configuration values.
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field


@dataclass
class PathConfig:
    """Configuration for file paths and directories."""
    documents_dir: str = "./data/documents"
    index_dir: str = "./data/index"
    cache_dir: str = "./data/cache"
    logs_dir: str = "./logs"
    temp_dir: str = "./tmp"


@dataclass
class SearchConfig:
    """Configuration for search engine settings."""
    engine_type: str = "whoosh"
    max_results: int = 20
    min_score: float = 0.1
    fuzzy_matching: bool = True
    stemming: bool = True
    case_sensitive: bool = False
    phrase_search: bool = True
    wildcard_search: bool = True
    boolean_operators: bool = True
    highlight_matches: bool = True
    snippet_length: int = 200


@dataclass
class IndexingConfig:
    """Configuration for document indexing."""
    supported_formats: List[str] = field(default_factory=lambda: [
        ".txt", ".md", ".py", ".js", ".html", ".css", ".json", ".xml", ".csv"
    ])
    max_file_size_mb: int = 50
    batch_size: int = 100
    auto_index: bool = True
    recursive_scan: bool = True
    ignore_hidden_files: bool = True
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "*.pyc", "*.log", "*.tmp", "__pycache__", ".git", ".vscode", "node_modules"
    ])
    text_encoding: str = "utf-8"
    min_word_length: int = 3
    max_word_length: int = 50


@dataclass
class NLPConfig:
    """Configuration for NLP processing."""
    language: str = "en"
    spacy_model: str = "en_core_web_sm"
    enable_ner: bool = True
    enable_pos_tagging: bool = True
    remove_stopwords: bool = True
    enable_lemmatization: bool = True
    custom_stopwords: List[str] = field(default_factory=lambda: [
        "bot", "search", "document", "file"
    ])


@dataclass
class ResponseConfig:
    """Configuration for response generation."""
    max_response_length: int = 1000
    include_source_info: bool = True
    include_relevance_score: bool = True
    format: str = "markdown"
    templates: Dict[str, str] = field(default_factory=lambda: {
        "no_results": "Sorry, I couldn't find any documents matching your query: '{query}'",
        "single_result": "I found 1 document related to your query:",
        "multiple_results": "I found {count} documents related to your query:",
        "error": "An error occurred while processing your request: {error}"
    })


@dataclass
class PerformanceConfig:
    """Configuration for performance settings."""
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    max_concurrent_searches: int = 5
    search_timeout_seconds: int = 30
    index_update_interval: int = 300
    memory_limit_mb: int = 512


@dataclass
class SecurityConfig:
    """Configuration for security settings."""
    max_query_length: int = 500
    rate_limit_requests_per_minute: int = 60
    allowed_file_types: List[str] = field(default_factory=lambda: [
        ".txt", ".md", ".py", ".js", ".html", ".css", ".json", ".xml", ".csv"
    ])
    blocked_extensions: List[str] = field(default_factory=lambda: [
        ".exe", ".bat", ".sh", ".ps1"
    ])
    sanitize_input: bool = True
    log_queries: bool = True


class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass


class Config:
    """
    Main configuration class that loads and manages all application settings.
    
    This class provides a centralized way to access configuration values
    with type safety and validation.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_file: Path to configuration file. If None, uses default location.
        """
        self.config_file = config_file or self._find_config_file()
        self._raw_config: Dict[str, Any] = {}
        self._logger = logging.getLogger(__name__)
        
        # Configuration sections
        self.paths: PathConfig = PathConfig()
        self.search: SearchConfig = SearchConfig()
        self.indexing: IndexingConfig = IndexingConfig()
        self.nlp: NLPConfig = NLPConfig()
        self.response: ResponseConfig = ResponseConfig()
        self.performance: PerformanceConfig = PerformanceConfig()
        self.security: SecurityConfig = SecurityConfig()
        
        # Load configuration
        self.load()
    
    def _find_config_file(self) -> str:
        """Find the configuration file in standard locations."""
        possible_paths = [
            "./config/settings.json",
            "../config/settings.json",
            "../../config/settings.json",
            os.path.expanduser("~/.search_bot/settings.json"),
            "/etc/search_bot/settings.json"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Default to the main config location
        return "./config/settings.json"
    
    def load(self) -> None:
        """Load configuration from file and environment variables."""
        try:
            self._load_from_file()
            self._load_from_environment()
            self._validate_config()
            self._ensure_directories()
            self._logger.info(f"Configuration loaded successfully from {self.config_file}")
        except Exception as e:
            self._logger.error(f"Failed to load configuration: {e}")
            raise ConfigurationError(f"Configuration loading failed: {e}")
    
    def _load_from_file(self) -> None:
        """Load configuration from JSON file."""
        if not os.path.exists(self.config_file):
            self._logger.warning(f"Config file not found: {self.config_file}. Using defaults.")
            return
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self._raw_config = json.load(f)
            
            # Load each configuration section
            self._load_section('paths', PathConfig)
            self._load_section('search', SearchConfig)
            self._load_section('indexing', IndexingConfig)
            self._load_section('nlp', NLPConfig)
            self._load_section('response', ResponseConfig)
            self._load_section('performance', PerformanceConfig)
            self._load_section('security', SecurityConfig)
            
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in config file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error reading config file: {e}")
    
    def _load_section(self, section_name: str, config_class: type) -> None:
        """Load a configuration section."""
        if section_name in self._raw_config:
            section_data = self._raw_config[section_name]
            # Update the configuration object with values from file
            config_obj = getattr(self, section_name)
            for key, value in section_data.items():
                if hasattr(config_obj, key):
                    setattr(config_obj, key, value)
    
    def _load_from_environment(self) -> None:
        """Load configuration overrides from environment variables."""
        env_mappings = {
            'SEARCH_BOT_DEBUG': ('application', 'debug', bool),
            'SEARCH_BOT_LOG_LEVEL': ('application', 'log_level', str),
            'SEARCH_BOT_DOCUMENTS_DIR': ('paths', 'documents_dir', str),
            'SEARCH_BOT_INDEX_DIR': ('paths', 'index_dir', str),
            'SEARCH_BOT_MAX_RESULTS': ('search', 'max_results', int),
            'SEARCH_BOT_CACHE_ENABLED': ('performance', 'cache_enabled', bool),
            'SEARCH_BOT_MAX_FILE_SIZE_MB': ('indexing', 'max_file_size_mb', int),
        }
        
        for env_var, (section, key, value_type) in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                try:
                    if value_type == bool:
                        parsed_value = env_value.lower() in ('true', '1', 'yes', 'on')
                    elif value_type == int:
                        parsed_value = int(env_value)
                    elif value_type == float:
                        parsed_value = float(env_value)
                    else:
                        parsed_value = env_value
                    
                    # Set the value in the appropriate configuration section
                    config_section = getattr(self, section)
                    setattr(config_section, key, parsed_value)
                    
                except (ValueError, TypeError) as e:
                    self._logger.warning(f"Invalid environment variable {env_var}: {e}")
    
    def _validate_config(self) -> None:
        """Validate configuration values."""
        # Validate paths exist or can be created
        required_paths = [
            self.paths.documents_dir,
            self.paths.index_dir,
            self.paths.cache_dir,
            self.paths.logs_dir
        ]
        
        for path in required_paths:
            try:
                Path(path).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ConfigurationError(f"Cannot create directory {path}: {e}")
        
        # Validate numeric ranges
        if self.search.max_results <= 0:
            raise ConfigurationError("search.max_results must be positive")
        
        if self.indexing.max_file_size_mb <= 0:
            raise ConfigurationError("indexing.max_file_size_mb must be positive")
        
        if self.performance.memory_limit_mb <= 0:
            raise ConfigurationError("performance.memory_limit_mb must be positive")
    
    def _ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        directories = [
            self.paths.documents_dir,
            self.paths.index_dir,
            self.paths.cache_dir,
            self.paths.logs_dir,
            self.paths.temp_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'search.max_results')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            parts = key.split('.')
            if len(parts) != 2:
                return default
            
            section, attr = parts
            config_section = getattr(self, section, None)
            if config_section is None:
                return default
            
            return getattr(config_section, attr, default)
        except (AttributeError, KeyError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'search.max_results')
            value: Value to set
        """
        parts = key.split('.')
        if len(parts) != 2:
            raise ValueError("Key must be in format 'section.attribute'")
        
        section, attr = parts
        config_section = getattr(self, section, None)
        if config_section is None:
            raise ValueError(f"Unknown configuration section: {section}")
        
        if not hasattr(config_section, attr):
            raise ValueError(f"Unknown configuration attribute: {attr}")
        
        setattr(config_section, attr, value)
    
    def reload(self) -> None:
        """Reload configuration from file."""
        self.load()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'paths': self.paths.__dict__,
            'search': self.search.__dict__,
            'indexing': self.indexing.__dict__,
            'nlp': self.nlp.__dict__,
            'response': self.response.__dict__,
            'performance': self.performance.__dict__,
            'security': self.security.__dict__,
        }
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"Config(file={self.config_file})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"Config(file='{self.config_file}', sections={list(self.to_dict().keys())})"


# Global configuration instance
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration instance.
    
    Returns:
        Global Config instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


def reload_config() -> None:
    """Reload the global configuration."""
    global _config_instance
    if _config_instance:
        _config_instance.reload()
    else:
        _config_instance = Config()