"""
Retrieval system module for vector search and content retrieval
"""

import os
import textwrap
from typing import List, Dict, Tuple
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from rich.console import Console
from rich.table import Table

from rag_pipeline.core.config import AppConfig, PipelineStage
from rag_pipeline.utils.device import get_device
from rag_pipeline.utils.text import wrap_text
from rag_pipeline.utils.logging import get_logger

console = Console()
logger = get_logger()


class RetrievalSystem:
    """Manages vector search and retrieval of relevant content"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.current_stage = PipelineStage.RETRIEVAL_SETUP
        self.embedding_model = None
        self.device = get_device(config.embedding_device)
        self.embeddings = None
        self.pages_and_chunks = None
        
    def load_data(self) -> None:
        """Load embeddings and chunks from file"""
        try:
            if not os.path.exists(self.config.embeddings_save_path):
                console.print(f"[red]Embeddings file {self.config.embeddings_save_path} not found[/red]")
                raise FileNotFoundError(f"Embeddings file {self.config.embeddings_save_path} not found")
                
            console.print(f"[yellow]Loading embeddings from {self.config.embeddings_save_path}...[/yellow]")
            
            # Load DataFrame
            text_chunks_and_embedding_df = pd.read_csv(self.config.embeddings_save_path)
            
            # Convert embedding column back to np.array
            text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(
                lambda x: np.fromstring(x.strip("[]"), sep=" ")
            )
            
            # Convert texts and embedding df to list of dicts
            self.pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")
            
            # Convert embeddings to torch tensor and send to device
            self.embeddings = torch.tensor(
                np.array(text_chunks_and_embedding_df["embedding"].tolist()), 
                dtype=torch.float32
            ).to(self.device)
            
            console.print(f"[green]Loaded {len(self.pages_and_chunks)} chunks with embeddings of shape {self.embeddings.shape}[/green]")
            
            # Initialize embedding model for queries
            self.embedding_model = SentenceTransformer(
                model_name_or_path=self.config.embedding_model_name,
                device=self.device
            )
            console.print(f"[green]Embedding model loaded successfully for retrieval[/green]")
            
        except Exception as e:
            console.print(f"[red]Error loading embeddings: {e}[/red]")
            raise
    
    def retrieve_relevant_resources(self, 
                                   query: str,
                                   n_resources: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve relevant resources based on query
        
        Args:
            query: The search query
            n_resources: Number of resources to return
            
        Returns:
            Tuple of (scores, indices)
        """
        if n_resources is None:
            n_resources = self.config.n_resources_to_return
            
        try:
            # Embed the query
            query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
            
            # Get dot product scores
            import time
            start_time = time.time()
            dot_scores = util.dot_score(query_embedding, self.embeddings)[0]
            end_time = time.time()
            
            if self.config.verbose:
                logger.debug(f"Time taken to get scores on {len(self.embeddings)} embeddings: {end_time-start_time:.5f} seconds")
            
            # Get top scores and indices
            scores, indices = torch.topk(input=dot_scores, k=n_resources)
            return scores, indices
            
        except Exception as e:
            console.print(f"[red]Error retrieving resources: {e}[/red]")
            raise
    
    def print_top_results(self, query: str, scores: torch.Tensor, indices: torch.Tensor) -> None:
        """Print top results with their scores and content"""
        console.print(f"\n[bold blue]Query:[/bold blue] {query}\n")
        
        table = Table(title="Search Results", show_header=True, header_style="bold magenta")
        table.add_column("Score", style="dim", width=8)
        table.add_column("Page", justify="right", width=6)
        table.add_column("Content", style="green")
        
        for score, index in zip(scores, indices):
            table.add_row(
                f"{score:.4f}", 
                str(self.pages_and_chunks[index]["page_number"]),
                wrap_text(self.pages_and_chunks[index]["sentence_chunk"], width=80)
            )
        
        console.print(table)
