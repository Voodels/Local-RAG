"""
Document processing module for PDF handling and text extraction
"""

import os
import re
import requests
import spacy
from typing import List, Dict
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import fitz  # PyMuPDF

from rag_pipeline.core.config import AppConfig, PipelineStage
from rag_pipeline.utils.text import text_formatter, split_list
from rag_pipeline.utils.logging import get_logger

console = Console()
logger = get_logger()


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
                    text = text_formatter(text)
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
                    item["sentence_chunks"] = split_list(
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
