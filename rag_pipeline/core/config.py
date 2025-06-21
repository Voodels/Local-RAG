"""
Configuration module for the RAG pipeline

Contains configuration classes and enums used throughout the pipeline.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


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
