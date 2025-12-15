#!/usr/bin/env python3
"""
RAG Command Line Interface
==========================
Interactive chat with your documents!

Author: Sanjeev
Date: December 2025
"""

import os
import sys
import time
import argparse

# Import our RAG components from the modular structure
from document_store import DocumentStore
from local_llm import LocalLLM
from rag_pipeline import RAGPipeline


# ============================================================
# COLORS - Makes the terminal output pretty
# ============================================================

class Colors:
    """ANSI color codes for terminal."""
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_color(text: str, color: str = ""):
    """Print text with color."""
    print(f"{color}{text}{Colors.END}")


def print_banner():
    """Show welcome message."""
    banner = """
╔═══════════════════════════════════════════╗
║         LOCAL RAG SYSTEM                  ║
║   Ask questions about your documents!     ║
╚═══════════════════════════════════════════╝
"""
    print_color(banner, Colors.CYAN)


def print_help():
    """Show available commands."""
    print("""
Commands:
  /help     Show this help message
  /sources  Toggle source display on/off
  /quit     Exit the program

Just type your question and press Enter!
""")


# ============================================================
# LOADING FUNCTIONS
# ============================================================

def load_documents(data_dir: str):
    """
    Load all .txt and .md files from a directory.
    
    Returns:
        texts: List of file contents
        sources: List of filenames
    """
    texts = []
    sources = []
    
    print(f"\nLoading documents from: {data_dir}")
    
    for filename in os.listdir(data_dir):
        # Only load text and markdown files
        if filename.endswith(('.txt', '.md')):
            filepath = os.path.join(data_dir, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                texts.append(content)
                sources.append(filename)
                print(f"  ✓ Loaded: {filename}")
                
            except Exception as e:
                print(f"  ✗ Error loading {filename}: {e}")
    
    return texts, sources


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    """Main entry point."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="RAG System - Chat with your documents"
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to the .gguf model file"
    )
    parser.add_argument(
        "--data", "-d",
        required=True,
        help="Path to folder containing documents"
    )
    
    args = parser.parse_args()
    
    # Check that paths exist
    if not os.path.exists(args.model):
        print_color(f"Error: Model not found: {args.model}", Colors.RED)
        sys.exit(1)
    
    if not os.path.isdir(args.data):
        print_color(f"Error: Data folder not found: {args.data}", Colors.RED)
        sys.exit(1)
    
    # Show welcome message
    print_banner()
    
    # ========================================
    # Initialize the RAG system
    # ========================================
    
    print_color("\n📚 Initializing RAG system...\n", Colors.YELLOW)
    
    # Step 1: Create document store
    print("Step 1/3: Loading embedding model...")
    doc_store = DocumentStore()
    
    # Step 2: Load documents
    print("\nStep 2/3: Loading documents...")
    texts, sources = load_documents(args.data)
    
    if not texts:
        print_color("\nError: No documents found!", Colors.RED)
        print("Make sure you have .txt or .md files in your data folder.")
        sys.exit(1)
    
    # Add documents to store
    doc_store.add_documents(texts, sources)
    
    # Step 3: Load the AI model
    print("\nStep 3/3: Loading AI model (this takes about a minute)...")
    llm = LocalLLM(model_path=args.model)
    
    # Create the pipeline
    pipeline = RAGPipeline(doc_store, llm)
    
    print_color("\n✓ System ready!\n", Colors.GREEN)
    print_help()
    
    # Settings
    show_sources = True
    
    # ========================================
    # Main chat loop
    # ========================================
    
    while True:
        try:
            # Get user input
            user_input = input(f"{Colors.CYAN}You: {Colors.END}").strip()
            
            # Skip empty input
            if not user_input:
                continue
            
            # Handle commands (start with /)
            if user_input.startswith("/"):
                cmd = user_input.lower()
                
                if cmd in ["/quit", "/exit", "/q"]:
                    print_color("\nGoodbye! 👋", Colors.CYAN)
                    break
                    
                elif cmd == "/help":
                    print_help()
                    
                elif cmd == "/sources":
                    show_sources = not show_sources
                    status = "ON" if show_sources else "OFF"
                    print_color(f"Source display: {status}", Colors.YELLOW)
                    
                else:
                    print_color(f"Unknown command: {cmd}", Colors.RED)
                    print("Type /help for available commands")
                
                continue
            
            # Process the question
            print_color("\n🤔 Thinking...", Colors.YELLOW)
            
            start_time = time.time()
            response = pipeline.query(user_input)
            elapsed = time.time() - start_time
            
            # Display the answer
            print()
            print_color("Answer:", Colors.GREEN + Colors.BOLD)
            print(response.answer)
            
            # Display confidence bar
            bar_filled = int(response.confidence * 10)
            bar_empty = 10 - bar_filled
            bar = "█" * bar_filled + "░" * bar_empty
            print_color(f"\nConfidence: [{bar}] {response.confidence:.0%}", Colors.CYAN)
            
            # Display sources (if enabled)
            if show_sources and response.sources:
                print_color("\nSources:", Colors.CYAN)
                for i, src in enumerate(response.sources, 1):
                    score = src.similarity_score
                    print(f"  [{i}] {src.source} (relevance: {score:.2f})")
            
            # Display timing
            print_color(f"\n⏱ Response time: {elapsed:.1f}s\n", Colors.YELLOW)
            
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print_color("\n\nType /quit to exit", Colors.YELLOW)
            
        except Exception as e:
            print_color(f"\nError: {e}", Colors.RED)
            print("Try again or type /quit to exit\n")


# Run the main function when script is executed
if __name__ == "__main__":
    main()