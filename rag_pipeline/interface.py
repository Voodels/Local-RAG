"""
Interactive interface for the RAG pipeline
"""

from typing import List
from rich.console import Console
from rich.panel import Panel

from rag_pipeline.core.pipeline import RAGPipeline
from rag_pipeline.utils.logging import get_logger

console = Console()
logger = get_logger()


def interactive_mode(pipeline: RAGPipeline) -> None:
    """Run the RAG pipeline in interactive mode"""
    console.print("\n[bold green]Starting interactive mode - type 'exit' or 'quit' to end the session[/bold green]\n")
    
    while True:
        query = console.input("[bold cyan]Enter your query: [/bold cyan]")
        
        if query.lower() in ['exit', 'quit']:
            console.print("[yellow]Exiting interactive mode...[/yellow]")
            break
            
        try:
            pipeline.ask(query)
        except Exception as e:
            logger.exception(f"Error processing query: {query}")
            console.print(f"[bold red]Error: {e}[/bold red]")
            console.print("[yellow]Continuing with next query...[/yellow]")


def sample_queries_mode(pipeline: RAGPipeline) -> None:
    """Run the pipeline with some sample queries"""
    # Nutrition-related sample queries
    sample_queries = [
        "What are the macronutrients, and what roles do they play in the human body?",
        "How do vitamins and minerals differ in their roles and importance for health?",
        "Describe the process of digestion and absorption of nutrients in the human body.",
        "What role does fibre play in digestion? Name five fibre containing foods.",
        "Explain the concept of energy balance and its importance in weight management.",
        "How often should infants be breastfed?",
        "What are symptoms of pellagra?",
        "How does saliva help with digestion?",
        "What is the RDI for protein per day?",
        "What are water soluble vitamins?"
    ]
    
    console.print(f"\n[bold green]Running {len(sample_queries)} sample queries...[/bold green]\n")
    
    for i, query in enumerate(sample_queries, 1):
        console.print(f"\n[bold blue]Sample Query {i}/{len(sample_queries)}:[/bold blue] {query}")
        
        try:
            pipeline.ask(query)
        except Exception as e:
            logger.exception(f"Error processing sample query: {query}")
            console.print(f"[bold red]Error: {e}[/bold red]")
            console.print("[yellow]Continuing with next query...[/yellow]")
        
        # Add a break between queries
        if i < len(sample_queries):
            console.print("\n" + "-" * 100 + "\n")


def single_query_mode(pipeline: RAGPipeline, query: str) -> None:
    """Run the pipeline with a single query"""
    try:
        pipeline.ask(query)
    except Exception as e:
        logger.exception(f"Error processing query: {query}")
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise
