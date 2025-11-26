#!/usr/bin/env python3
"""
Search Bot - Main Entry Point

This is the main entry point for the search bot application.
It handles initialization, user interaction, and coordinates between
different components of the search system.
"""

import sys
import os
from pathlib import Path

# Check Python version compatibility
if sys.version_info < (3, 8):
    print("Error: This application requires Python 3.8 or higher.")
    print(f"Current Python version: {sys.version}")
    sys.exit(1)

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src import SearchEngine, ResponseHandler, DocumentIndexer, Config


def main():
    """
    Main function to run the search bot application.
    """
    print("🤖 Welcome to Search Bot!")
    print("=" * 40)
    
    try:
        # Initialize configuration
        config = Config()
        print("✅ Configuration loaded")
        
        # Initialize components
        indexer = DocumentIndexer()
        search_engine = SearchEngine()
        response_handler = ResponseHandler()
        
        print("✅ Search bot components initialized")
        print("\nType 'quit' or 'exit' to stop the bot")
        print("Type 'help' for available commands")
        print("-" * 40)
        
        # Main interaction loop
        while True:
            try:
                user_input = input("\n🔍 Search Bot > ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye! Thanks for using Search Bot!")
                    break
                    
                if user_input.lower() in ['help', 'h']:
                    show_help()
                    continue
                    
                if user_input.lower().startswith('index'):
                    handle_indexing_command(user_input, indexer)
                    continue
                
                # Process search query
                print(f"🔎 Searching for: '{user_input}'")
                
                # Perform search
                search_results = search_engine.search(user_input)
                
                # Generate response
                response = response_handler.generate_response(user_input, search_results)
                
                # Display response
                print(f"💬 Response: {response}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Thanks for using Search Bot!")
                break
            except Exception as e:
                print(f"❌ Error processing request: {e}")
                
    except Exception as e:
        print(f"❌ Failed to initialize search bot: {e}")
        sys.exit(1)


def show_help():
    """
    Display help information for available commands.
    """
    help_text = """
📖 Available Commands:
    
    help, h              - Show this help message
    quit, exit, q        - Exit the search bot
    index <path>         - Index documents from specified path
    <search query>       - Search for documents matching your query
    
📝 Examples:
    index ./data/documents
    What is machine learning?
    Find documents about Python programming
    """
    print(help_text)


def handle_indexing_command(command, indexer):
    """
    Handle document indexing commands.
    
    Args:
        command (str): The indexing command
        indexer (DocumentIndexer): The indexer instance
    """
    parts = command.split()
    if len(parts) < 2:
        print("❌ Usage: index <path>")
        return
        
    path = parts[1]
    if not os.path.exists(path):
        print(f"❌ Path not found: {path}")
        return
        
    try:
        print(f"📚 Indexing documents from: {path}")
        result = indexer.index_documents(path)
        print(f"✅ Indexing completed: {result}")
    except Exception as e:
        print(f"❌ Indexing failed: {e}")


if __name__ == "__main__":
    main()