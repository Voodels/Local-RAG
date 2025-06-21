"""
Main RAG pipeline module that orchestrates the entire process
"""

import os
from typing import Tuple, List, Dict
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from rag_pipeline.core.config import AppConfig, PipelineStage
from rag_pipeline.processors.document import DocumentProcessor
from rag_pipeline.processors.embedding import EmbeddingCreator
from rag_pipeline.processors.retrieval import RetrievalSystem
from rag_pipeline.processors.llm import LLMProcessor
from rag_pipeline.utils.logging import get_logger

console = Console()
logger = get_logger()


class RAGPipeline:
    """Main RAG pipeline that orchestrates the entire process"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.document_processor = None
        self.embedding_creator = None
        self.retrieval_system = None
        self.llm_processor = None
        self.current_stage = PipelineStage.DOCUMENT_LOADING
        
        logger.info(f"Initializing RAG pipeline with config: {config}")
        
    def initialize_pipeline(self) -> None:
        """Initialize and prepare the pipeline components"""
        try:
            # Initialize document processor
            console.print(f"\n[bold cyan]Step 1: Document Processing[/bold cyan]")
            self.document_processor = DocumentProcessor(self.config)
            self.document_processor.download_pdf()
            self.document_processor.read_pdf()
            self.document_processor.process_to_sentences()
            self.document_processor.create_chunks()
            
            # Check if embeddings file exists
            if os.path.exists(self.config.embeddings_save_path):
                console.print(f"\n[bold cyan]Step 2: Loading Existing Embeddings[/bold cyan]")
                # Skip embedding creation, will be loaded in retrieval system
            else:
                # Create embeddings
                console.print(f"\n[bold cyan]Step 2: Embedding Creation[/bold cyan]")
                self.embedding_creator = EmbeddingCreator(
                    self.config, 
                    self.document_processor.pages_and_chunks_over_min_token_len
                )
                self.embedding_creator.initialize_model()
                self.embedding_creator.create_embeddings()
                self.embedding_creator.save_embeddings()
            
            # Initialize retrieval system
            console.print(f"\n[bold cyan]Step 3: Setting up Retrieval System[/bold cyan]")
            self.retrieval_system = RetrievalSystem(self.config)
            self.retrieval_system.load_data()
            
            # Initialize LLM
            console.print(f"\n[bold cyan]Step 4: Setting up LLM[/bold cyan]")
            self.llm_processor = LLMProcessor(self.config)
            self.llm_processor.initialize_model()
            
            console.print(f"\n[bold green]Pipeline initialized successfully![/bold green]")
            
        except Exception as e:
            console.print(f"[bold red]Failed to initialize pipeline: {e}[/bold red]")
            logger.exception("Pipeline initialization failed")
            raise
    
    def ask(self, 
           query: str,
           temperature: float = None,
           max_new_tokens: int = None,
           print_results: bool = True) -> Tuple[str, List[Dict]]:
        """
        Process a query through the complete RAG pipeline
        
        Args:
            query: User query string
            temperature: Temperature for LLM generation
            max_new_tokens: Max tokens to generate
            print_results: Whether to print results to console
            
        Returns:
            Tuple of (answer, context_items)
        """
        try:
            self.current_stage = PipelineStage.QUERY_PROCESSING
            
            # Retrieve relevant resources
            scores, indices = self.retrieval_system.retrieve_relevant_resources(
                query=query,
                n_resources=self.config.n_resources_to_return
            )
            
            # Create context items
            context_items = [self.retrieval_system.pages_and_chunks[i] for i in indices]
            
            # Add score to context items
            for i, item in enumerate(context_items):
                item["score"] = scores[i].cpu().item()  # convert to Python float
            
            # Print search results if requested
            if print_results:
                self.retrieval_system.print_top_results(query, scores, indices)
            
            # Generate answer
            answer = self.llm_processor.generate_answer(
                query=query,
                context_items=context_items,
                temperature=temperature,
                max_new_tokens=max_new_tokens
            )
            
            # Print answer if requested
            if print_results:
                console.print(f"\n[bold blue]Query:[/bold blue] {query}")
                console.print(Panel(Markdown(answer), title="Generated Answer", border_style="green"))
            
            return answer, context_items
            
        except Exception as e:
            console.print(f"[bold red]Error processing query: {e}[/bold red]")
            logger.exception(f"Error processing query: {query}")
            raise
