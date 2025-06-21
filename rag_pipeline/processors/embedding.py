"""
Embedding processing module for creating and managing text embeddings
"""

import time
import pandas as pd
from typing import List, Dict
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from rag_pipeline.core.config import AppConfig, PipelineStage
from rag_pipeline.utils.device import get_device, print_device_info
from rag_pipeline.utils.logging import get_logger

console = Console()
logger = get_logger()


class EmbeddingCreator:
    """Creates and manages embeddings for text chunks"""
    
    def __init__(self, config: AppConfig, text_chunks: List[Dict]):
        self.config = config
        self.text_chunks = text_chunks
        self.current_stage = PipelineStage.EMBEDDING_CREATION
        self.embedding_model = None
        self.device = get_device(config.embedding_device)
        
    def initialize_model(self) -> None:
        """Initialize the embedding model"""
        try:
            console.print(f"[yellow]Loading embedding model: {self.config.embedding_model_name}...[/yellow]")
            self.embedding_model = SentenceTransformer(
                model_name_or_path=self.config.embedding_model_name,
                device=self.device
            )
            console.print(f"[green]Embedding model loaded successfully on {self.device}[/green]")
            
            # Show device info
            print_device_info(self.device)
            
        except Exception as e:
            console.print(f"[red]Error initializing embedding model: {e}[/red]")
            raise
    
    def create_embeddings(self) -> List[Dict]:
        """Create embeddings for all text chunks"""
        try:
            # Extract text chunks
            text_chunk_strings = [item["sentence_chunk"] for item in self.text_chunks]
            
            console.print(f"[yellow]Creating embeddings for {len(text_chunk_strings)} text chunks...[/yellow]")
            console.print(f"[yellow]Using device: {self.device}[/yellow]")
            
            start_time = time.time()
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Creating embeddings..."),
                BarColumn(),
                TextColumn("[bold]{task.percentage:.2f}%"),
                console=console
            ) as progress:
                progress.add_task("Embedding", total=1)
                
                # Create embeddings in batch
                batch_embeddings = self.embedding_model.encode(
                    text_chunk_strings,
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
            
            end_time = time.time()
            console.print(f"[green]Time taken for batch embedding: {end_time-start_time:.5f} seconds[/green]")
            console.print(f"[green]Embeddings shape: {batch_embeddings.shape}[/green]")
            
            # Add embeddings back to the text chunks
            for i, item in enumerate(self.text_chunks):
                item["embedding"] = batch_embeddings[i].cpu().numpy()
            
            return self.text_chunks
            
        except Exception as e:
            console.print(f"[red]Error creating embeddings: {e}[/red]")
            raise
    
    def save_embeddings(self) -> None:
        """Save embeddings to a file"""
        try:
            text_chunks_and_embeddings_df = pd.DataFrame(self.text_chunks)
            console.print(f"[yellow]Saving embeddings to {self.config.embeddings_save_path}...[/yellow]")
            text_chunks_and_embeddings_df.to_csv(self.config.embeddings_save_path, index=False)
            console.print(f"[green]Embeddings saved successfully[/green]")
        except Exception as e:
            console.print(f"[red]Error saving embeddings: {e}[/red]")
            raise
