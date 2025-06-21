#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for the modular RAG pipeline

This script provides a command-line interface for the RAG pipeline with support for:
- Interactive mode
- Sample queries
- Single query execution
- Configuration via command line or config file

Usage:
    python main.py --interactive
    python main.py --sample-queries
    python main.py --query "Your question here"
    python main.py --config config.json --interactive
"""

import os
import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

# Add the current directory to the path to import our modules
sys.path.insert(0, str(Path(__file__).parent))

from rag_pipeline import RAGPipeline, AppConfig
from rag_pipeline.interface import interactive_mode, sample_queries_mode, single_query_mode
from rag_pipeline.utils.logging import setup_logger

# Load environment variables
load_dotenv()

# Initialize rich console
console = Console()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Modular Local RAG Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --interactive
  python main.py --sample-queries
  python main.py --query "What are the macronutrients?"
  python main.py --config custom_config.json --interactive
  python main.py --pdf document.pdf --model microsoft/DialoGPT-medium --interactive
        """
    )
    
    # Configuration options
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--pdf', type=str, help='Path to the PDF document')
    parser.add_argument('--pdf-url', type=str, help='URL to download PDF if not present')
    parser.add_argument('--embeddings', type=str, help='Path to save/load embeddings')
    parser.add_argument('--model', type=str, help='LLM model ID from Hugging Face')
    parser.add_argument('--embedding-model', type=str, help='Embedding model name')
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'], help='Device preference')
    parser.add_argument('--no-quantization', action='store_true', help='Disable 4-bit quantization')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level')
    
    # Execution modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    mode_group.add_argument('--sample-queries', action='store_true', help='Run with sample queries')
    mode_group.add_argument('--query', type=str, help='Run with a single query')
    
    return parser.parse_args()


def load_config_from_file(config_path: str) -> dict:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        console.print(f"[green]Configuration loaded from {config_path}[/green]")
        return config_data
    except Exception as e:
        console.print(f"[red]Error loading config file {config_path}: {e}[/red]")
        return {}


def create_config(args) -> AppConfig:
    """Create AppConfig from command line arguments and config file"""
    # Start with default config
    config_dict = {}
    
    # Load from config file if provided
    if args.config and os.path.exists(args.config):
        config_dict.update(load_config_from_file(args.config))
    
    # Override with command line arguments (these take precedence)
    if args.pdf:
        config_dict['pdf_path'] = args.pdf
    if args.pdf_url:
        config_dict['pdf_download_url'] = args.pdf_url
    if args.embeddings:
        config_dict['embeddings_save_path'] = args.embeddings
    if args.model:
        config_dict['llm_model_id'] = args.model
    if args.embedding_model:
        config_dict['embedding_model_name'] = args.embedding_model
    if args.device:
        config_dict['embedding_device'] = args.device
        config_dict['llm_device'] = args.device
    if args.no_quantization:
        config_dict['use_quantization'] = False
    if args.verbose:
        config_dict['verbose'] = True
    if args.log_level:
        config_dict['log_level'] = args.log_level
    
    # Create config object
    return AppConfig(**config_dict)


def show_welcome_message():
    """Display welcome message"""
    welcome_text = """
[bold blue]Modular RAG Pipeline v2.0[/bold blue]

A local Retrieval Augmented Generation system for PDF documents
- Document processing and chunking
- Vector embeddings and similarity search  
- LLM-powered answer generation
- Fully modular and extensible architecture
    """
    
    console.print(Panel.fit(welcome_text.strip(), border_style="green"))


def main():
    """Main entry point for the RAG pipeline application"""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Show welcome message
        show_welcome_message()
        
        # Create configuration
        config = create_config(args)
        
        # Setup logging
        setup_logger(log_level=config.log_level)
        
        # Display configuration summary
        console.print(f"\n[bold cyan]Configuration Summary:[/bold cyan]")
        console.print(f"PDF Path: {config.pdf_path}")
        console.print(f"LLM Model: {config.llm_model_id}")
        console.print(f"Embedding Model: {config.embedding_model_name}")
        console.print(f"Embeddings Path: {config.embeddings_save_path}")
        console.print(f"Device Settings: {config.embedding_device}/{config.llm_device}")
        console.print(f"Quantization: {'Enabled' if config.use_quantization else 'Disabled'}")
        
        # Initialize pipeline
        console.print(f"\n[bold cyan]Initializing Pipeline...[/bold cyan]")
        pipeline = RAGPipeline(config)
        pipeline.initialize_pipeline()
        
        # Execute based on mode
        if args.sample_queries:
            sample_queries_mode(pipeline)
        elif args.interactive:
            interactive_mode(pipeline)
        elif args.query:
            single_query_mode(pipeline, args.query)
        else:
            console.print("\n[yellow]No execution mode specified. Use --interactive, --sample-queries, or --query[/yellow]")
            console.print("[yellow]Starting interactive mode by default...[/yellow]\n")
            interactive_mode(pipeline)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Process interrupted by user. Exiting...[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]Fatal error: {e}[/bold red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
