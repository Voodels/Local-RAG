#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LocalRAG - A simple local RAG (Retrieval Augmented Generation) pipeline

This script implements a complete RAG pipeline that runs locally on your hardware:
1. Document preprocessing and embedding creation
2. Retrieval system for finding relevant information based on queries
3. LLM for generating answers based on the retrieved context

Author: Based on code by Daniel Bourke
License: MIT
"""

import os
import re
import time
import sys
import json
import random
import textwrap
import argparse
from enum import Enum
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
from datetime import datetime

# Standard data processing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Rich for beautiful terminal output
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, TaskID, TextColumn, BarColumn, SpinnerColumn
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.table import Table
from rich import print as rprint
from rich.tree import Tree

# Logging
from loguru import logger

# PDF processing
import fitz  # PyMuPDF
import requests

# Deep learning
import torch
from torch.cuda.amp import autocast
from sentence_transformers import SentenceTransformer, util

# LLM
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    GenerationConfig
)
from transformers.utils import is_flash_attn_2_available
import spacy

# Configuration using Pydantic
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize rich console
console = Console()


class PipelineStage(Enum):
    """Enum for tracking pipeline stages"""
    DOCUMENT_LOADING = "Document Loading"
    TEXT_PROCESSING = "Text Processing"
    EMBEDDING_CREATION = "Embedding Creation"
    RETRIEVAL_SETUP = "Retrieval Setup"
    LLM_SETUP = "LLM Setup"
    QUERY_PROCESSING = "Query Processing"
    COMPLETE = "Pipeline Complete"


class AppConfig(BaseSettings):
    """Configuration for the RAG application"""
    # PDF settings
    pdf_path: str = Field("document.pdf", description="Path to PDF document")
    pdf_download_url: Optional[str] = Field(None, description="URL to download PDF if not present")
    
    # Processing settings
    min_token_length: int = Field(30, description="Minimum token length for chunks to keep")
    page_offset: int = Field(0, description="Page offset for PDF numbering")
    num_sentence_chunk_size: int = Field(10, description="Number of sentences per chunk")
    
    # Model settings
    embedding_model_name: str = Field("all-mpnet-base-v2", description="Name of embedding model")
    embedding_device: str = Field("auto", description="Device for embedding model (auto, cpu, cuda)")
    
    # LLM settings
    llm_model_id: str = Field("google/gemma-2b-it", description="HuggingFace model ID for LLM")
    use_quantization: bool = Field(True, description="Whether to use 4-bit quantization")
    llm_device: str = Field("auto", description="Device for LLM (auto, cpu, cuda)")
    
    # RAG settings
    n_resources_to_return: int = Field(5, description="Number of resources to retrieve per query")
    temperature: float = Field(0.7, description="Temperature for LLM generation")
    max_new_tokens: int = Field(512, description="Maximum number of new tokens to generate")
    
    # File paths
    embeddings_save_path: str = Field("text_chunks_and_embeddings_df.csv", description="Path to save embeddings")
    
    # Run settings
    verbose: bool = Field(True, description="Verbose output")
    log_level: str = Field("INFO", description="Logging level")
    
    class Config:
        env_prefix = "RAG_"
        case_sensitive = False


# Setup logger
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
logger.add("rag_pipeline_{time}.log", rotation="10 MB", retention="7 days")


class DocumentProcessor:
    """Handles document loading and text processing"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.current_stage = PipelineStage.DOCUMENT_LOADING
        self._download_progress = None
        self._download_task = None
        self.nlp = None
        self.pages_and_texts = []
        self.pages_and_chunks = []
        self.pages_and_chunks_over_min_token_len = []
        
    def download_pdf(self) -> None:
        """Download PDF if it doesn't exist"""
        if os.path.exists(self.config.pdf_path):
            console.print(f"[green]File {self.config.pdf_path} exists.[/green]")
            return
            
        if not self.config.pdf_download_url:
            console.print("[red]PDF file doesn't exist and no download URL provided[/red]")
            raise FileNotFoundError(f"PDF file {self.config.pdf_path} not found")
            
        console.print("[yellow]File doesn't exist, downloading...[/yellow]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[bold]{task.completed}/{task.total}"),
            console=console
        ) as progress:
            self._download_progress = progress
            self._download_task = progress.add_task(f"Downloading {self.config.pdf_path}...", total=None)
            
            try:
                response = requests.get(self.config.pdf_download_url, stream=True)
                response.raise_for_status()
                
                # Get total size if available
                total_size = int(response.headers.get('content-length', 0))
                if total_size:
                    progress.update(self._download_task, total=total_size)
                
                # Save the content to the file
                with open(self.config.pdf_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
                        if total_size:
                            progress.update(self._download_task, advance=len(chunk))
                
                console.print(f"[green]The file has been downloaded and saved as {self.config.pdf_path}[/green]")
            except requests.RequestException as e:
                console.print(f"[red]Failed to download the file: {e}[/red]")
                raise
    
    def _text_formatter(self, text: str) -> str:
        """Clean and format text from PDF"""
        cleaned_text = text.replace("\n", " ").strip()
        # Remove extra spaces
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        return cleaned_text
    
    def read_pdf(self) -> List[Dict]:
        """Read PDF and extract text with metadata"""
        try:
            doc = fitz.open(self.config.pdf_path)
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[bold]{task.percentage:.2f}% • {task.completed}/{task.total}"),
                console=console
            ) as progress:
                task = progress.add_task(f"Reading PDF document ({doc.page_count} pages)...", total=doc.page_count)
                
                for page_number, page in enumerate(doc):
                    text = page.get_text()
                    text = self._text_formatter(text)
                    self.pages_and_texts.append({
                        "page_number": page_number - self.config.page_offset,
                        "page_char_count": len(text),
                        "page_word_count": len(text.split(" ")),
                        "page_sentence_count_raw": len(text.split(". ")),
                        "page_token_count": len(text) / 4,  # 1 token ~= 4 chars
                        "text": text
                    })
                    progress.update(task, advance=1)
            
            console.print(f"[green]Successfully processed {len(self.pages_and_texts)} pages from {self.config.pdf_path}[/green]")
            return self.pages_and_texts
            
        except Exception as e:
            console.print(f"[red]Error reading PDF: {e}[/red]")
            raise
    
    def process_to_sentences(self) -> None:
        """Process pages into sentences using spaCy"""
        self.current_stage = PipelineStage.TEXT_PROCESSING
        
        try:
            # Load spaCy
            if self.nlp is None:
                console.print("[yellow]Loading spaCy English model...[/yellow]")
                self.nlp = spacy.load("en_core_web_sm")
                if not self.nlp.has_pipe("sentencizer"):
                    self.nlp.add_pipe("sentencizer")
                console.print("[green]spaCy model loaded successfully[/green]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[bold]{task.percentage:.2f}% • {task.completed}/{task.total}"),
                console=console
            ) as progress:
                task = progress.add_task("Processing text into sentences...", total=len(self.pages_and_texts))
                
                for item in self.pages_and_texts:
                    item["sentences"] = list(self.nlp(item["text"]).sents)
                    # Make sure all sentences are strings
                    item["sentences"] = [str(sentence) for sentence in item["sentences"]]
                    # Count the sentences 
                    item["page_sentence_count_spacy"] = len(item["sentences"])
                    progress.update(task, advance=1)
                    
            console.print(f"[green]Successfully processed sentences for {len(self.pages_and_texts)} pages[/green]")
        except Exception as e:
            console.print(f"[red]Error in sentence processing: {e}[/red]")
            raise
    
    def _split_list(self, input_list: List, slice_size: int) -> List[List]:
        """Split a list into chunks of specified size"""
        return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]
    
    def create_chunks(self) -> None:
        """Create chunks of sentences from pages"""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[bold]{task.percentage:.2f}% • {task.completed}/{task.total}"),
                console=console
            ) as progress:
                task1 = progress.add_task("Creating sentence chunks...", total=len(self.pages_and_texts))
                
                # Create chunks from sentences
                for item in self.pages_and_texts:
                    item["sentence_chunks"] = self._split_list(
                        input_list=item["sentences"],
                        slice_size=self.config.num_sentence_chunk_size
                    )
                    item["num_chunks"] = len(item["sentence_chunks"])
                    progress.update(task1, advance=1)
                    
                # Convert chunks to individual items
                progress.add_task("Converting chunks to individual items...")
                
                for item in self.pages_and_texts:
                    for sentence_chunk in item["sentence_chunks"]:
                        chunk_dict = {}
                        chunk_dict["page_number"] = item["page_number"]
                        
                        # Join the sentences together into a paragraph
                        joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
                        joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)
                        chunk_dict["sentence_chunk"] = joined_sentence_chunk
                        
                        # Get stats about the chunk
                        chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
                        chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
                        chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4  # 1 token ~= 4 chars
                        
                        self.pages_and_chunks.append(chunk_dict)
                
                # Filter by minimum token length
                task3 = progress.add_task("Filtering chunks by minimum token length...", total=len(self.pages_and_chunks))
                self.pages_and_chunks_over_min_token_len = []
                
                for chunk in self.pages_and_chunks:
                    if chunk["chunk_token_count"] > self.config.min_token_length:
                        self.pages_and_chunks_over_min_token_len.append(chunk)
                    progress.update(task3, advance=1)
            
            console.print(f"[green]Created {len(self.pages_and_chunks)} chunks, {len(self.pages_and_chunks_over_min_token_len)} chunks after filtering[/green]")
        except Exception as e:
            console.print(f"[red]Error in chunk creation: {e}[/red]")
            raise


class EmbeddingCreator:
    """Creates and manages embeddings for text chunks"""
    
    def __init__(self, config: AppConfig, text_chunks: List[Dict]):
        self.config = config
        self.text_chunks = text_chunks
        self.current_stage = PipelineStage.EMBEDDING_CREATION
        self.embedding_model = None
        self.device = self._get_device(config.embedding_device)
        
    def _get_device(self, device_preference: str) -> str:
        """Determine the actual device to use based on preference and availability"""
        if device_preference.lower() == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device_preference.lower()
    
    def initialize_model(self) -> None:
        """Initialize the embedding model"""
        try:
            console.print(f"[yellow]Loading embedding model: {self.config.embedding_model_name}...[/yellow]")
            self.embedding_model = SentenceTransformer(
                model_name_or_path=self.config.embedding_model_name,
                device=self.device
            )
            console.print(f"[green]Embedding model loaded successfully on {self.device}[/green]")
            
            # Show GPU info if available
            if self.device == "cuda":
                console.print(f"[blue]GPU: {torch.cuda.get_device_name(0)}[/blue]")
                console.print(f"[blue]GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB[/blue]")
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


class RetrievalSystem:
    """Manages vector search and retrieval of relevant content"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.current_stage = PipelineStage.RETRIEVAL_SETUP
        self.embedding_model = None
        self.device = self._get_device(config.embedding_device)
        self.embeddings = None
        self.pages_and_chunks = None
        
    def _get_device(self, device_preference: str) -> str:
        """Determine the actual device to use based on preference and availability"""
        if device_preference.lower() == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device_preference.lower()
    
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
                self._wrap_text(self.pages_and_chunks[index]["sentence_chunk"], width=80)
            )
        
        console.print(table)
    
    def _wrap_text(self, text: str, width: int = 80) -> str:
        """Wrap text to specified width"""
        return "\n".join(textwrap.wrap(text, width=width))


class LLMProcessor:
    """Manages LLM for generating responses based on retrieved content"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.current_stage = PipelineStage.LLM_SETUP
        self.tokenizer = None
        self.llm_model = None
        self.device = self._get_device(config.llm_device)
        
    def _get_device(self, device_preference: str) -> str:
        """Determine the actual device to use based on preference and availability"""
        if device_preference.lower() == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device_preference.lower()
    
    def initialize_model(self) -> None:
        """Initialize LLM and tokenizer"""
        try:
            console.print(f"[yellow]Loading LLM: {self.config.llm_model_id}...[/yellow]")
            
            # Create quantization config if needed
            quantization_config = None
            if self.config.use_quantization:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                console.print("[yellow]Using 4-bit quantization[/yellow]")
            
            # Setup attention implementation
            if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
                attn_implementation = "flash_attention_2"
                console.print("[green]Using Flash Attention 2[/green]")
            else:
                attn_implementation = "sdpa"
                console.print("[yellow]Using scaled dot product attention[/yellow]")
            
            # Instantiate tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=self.config.llm_model_id
            )
            
            # Instantiate the model
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=self.config.llm_model_id,
                torch_dtype=torch.float16,  # datatype to use
                quantization_config=quantization_config if self.config.use_quantization else None,
                low_cpu_mem_usage=False,  # use full memory
                attn_implementation=attn_implementation  # which attention version to use
            )
            
            if not self.config.use_quantization:
                self.llm_model.to(self.device)
            
            # Model stats
            num_params = sum([param.numel() for param in self.llm_model.parameters()])
            mem_params = sum([param.nelement() * param.element_size() for param in self.llm_model.parameters()])
            mem_buffers = sum([buf.nelement() * buf.element_size() for buf in self.llm_model.buffers()])
            model_mem_gb = (mem_params + mem_buffers) / (1024**3)
            
            console.print(f"[green]LLM loaded successfully:[/green]")
            console.print(f"[blue]Parameters: {num_params:,}[/blue]")
            console.print(f"[blue]Memory usage: {model_mem_gb:.2f} GB[/blue]")
            
        except Exception as e:
            console.print(f"[red]Error initializing LLM: {e}[/red]")
            raise
    
    def prompt_formatter(self, query: str, context_items: List[Dict]) -> str:
        """Format prompt with query and context items"""
        # Join context items into one dotted paragraph
        context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])

        # Create a base prompt with examples
        base_prompt = """Based on the following context items, please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
Use the following examples as reference for the ideal answer style.
\nExample 1:
Query: What are the fat-soluble vitamins?
Answer: The fat-soluble vitamins include Vitamin A, Vitamin D, Vitamin E, and Vitamin K. These vitamins are absorbed along with fats in the diet and can be stored in the body's fatty tissue and liver for later use. Vitamin A is important for vision, immune function, and skin health. Vitamin D plays a critical role in calcium absorption and bone health. Vitamin E acts as an antioxidant, protecting cells from damage. Vitamin K is essential for blood clotting and bone metabolism.
\nExample 2:
Query: What are the causes of type 2 diabetes?
Answer: Type 2 diabetes is often associated with overnutrition, particularly the overconsumption of calories leading to obesity. Factors include a diet high in refined sugars and saturated fats, which can lead to insulin resistance, a condition where the body's cells do not respond effectively to insulin. Over time, the pancreas cannot produce enough insulin to manage blood sugar levels, resulting in type 2 diabetes. Additionally, excessive caloric intake without sufficient physical activity exacerbates the risk by promoting weight gain and fat accumulation, particularly around the abdomen, further contributing to insulin resistance.
\nExample 3:
Query: What is the importance of hydration for physical performance?
Answer: Hydration is crucial for physical performance because water plays key roles in maintaining blood volume, regulating body temperature, and ensuring the transport of nutrients and oxygen to cells. Adequate hydration is essential for optimal muscle function, endurance, and recovery. Dehydration can lead to decreased performance, fatigue, and increased risk of heat-related illnesses, such as heat stroke. Drinking sufficient water before, during, and after exercise helps ensure peak physical performance and recovery.
\nNow use the following context items to answer the user query:
{context}
\nRelevant passages: <extract relevant passages from the context here>
User query: {query}
Answer:"""

        # Update base prompt with context items and query   
        base_prompt = base_prompt.format(context=context, query=query)

        # Create prompt template for instruction-tuned model
        dialogue_template = [
            {"role": "user",
             "content": base_prompt}
        ]

        # Apply the chat template
        prompt = self.tokenizer.apply_chat_template(
            conversation=dialogue_template,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt
    
    def generate_answer(self, 
                       query: str, 
                       context_items: List[Dict],
                       temperature: float = None,
                       max_new_tokens: int = None) -> str:
        """
        Generate an answer based on the query and context items
        
        Args:
            query: The user query
            context_items: List of relevant context items
            temperature: Generation temperature (randomness)
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            Generated answer text
        """
        if temperature is None:
            temperature = self.config.temperature
            
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens
            
        try:
            # Format the prompt
            prompt = self.prompt_formatter(query=query, context_items=context_items)
            
            # Tokenize the prompt
            input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Generating answer..."),
                BarColumn(),
                console=console
            ) as progress:
                progress.add_task("Generating", total=1)
                
                # Generate output
                outputs = self.llm_model.generate(
                    **input_ids,
                    temperature=temperature,
                    do_sample=True,
                    max_new_tokens=max_new_tokens
                )
            
            # Decode output
            output_text = self.tokenizer.decode(outputs[0])
            
            # Clean output
            formatted_answer = output_text.replace(prompt, "").replace("<bos>", "").replace("<eos>", "")
            formatted_answer = formatted_answer.replace("Sure, here is the answer to the user query:\n\n", "")
            formatted_answer = formatted_answer.strip()
            
            return formatted_answer
            
        except Exception as e:
            console.print(f"[red]Error generating answer: {e}[/red]")
            raise


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


def main():
    """Main entry point for the RAG pipeline application"""
    # Setup command line arguments
    parser = argparse.ArgumentParser(description='Local RAG Pipeline')
    parser.add_argument('--pdf', type=str, help='Path to the PDF document')
    parser.add_argument('--pdf-url', type=str, help='URL to download PDF if not present')
    parser.add_argument('--embeddings', type=str, help='Path to save/load embeddings')
    parser.add_argument('--model', type=str, help='LLM model ID from Hugging Face')
    parser.add_argument('--sample-queries', action='store_true', help='Run with sample queries')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--query', type=str, help='Run with a single query')
    parser.add_argument('--config', type=str, help='Path to config file')
    args = parser.parse_args()
    
    # Show welcome message
    console.print(Panel.fit(
        "[bold blue]Local RAG Pipeline[/bold blue]\nA local Retrieval Augmented Generation system for PDF documents",
        border_style="green"
    ))
    
    # Load config
    config = AppConfig()
    
    # Override with config file if provided
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                config_data = json.load(f)
                for key, value in config_data.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
        except Exception as e:
            console.print(f"[red]Error loading config file: {e}[/red]")
    
    # Override config with command line arguments
    if args.pdf:
        config.pdf_path = args.pdf
    if args.pdf_url:
        config.pdf_download_url = args.pdf_url
    if args.embeddings:
        config.embeddings_save_path = args.embeddings
    if args.model:
        config.llm_model_id = args.model
    
    # Initialize pipeline
    pipeline = RAGPipeline(config)
    pipeline.initialize_pipeline()
    
    # Execute based on mode
    if args.sample_queries:
        sample_queries_mode(pipeline)
    elif args.interactive:
        interactive_mode(pipeline)
    elif args.query:
        pipeline.ask(args.query)
    else:
        console.print("\n[yellow]No execution mode specified. Use --interactive, --sample-queries, or --query[/yellow]")
        console.print("[yellow]Starting interactive mode by default...[/yellow]\n")
        interactive_mode(pipeline)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Process interrupted by user. Exiting...[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]Fatal error: {e}[/bold red]")
        logger.exception("Fatal error in main")
        sys.exit(1)
