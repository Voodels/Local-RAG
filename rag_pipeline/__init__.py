"""
RAG Pipeline - A modular local RAG (Retrieval Augmented Generation) system

This package implements a complete RAG pipeline with the following components:
- Document processing and text chunking
- Embedding creation and management
- Vector-based retrieval system
- LLM-based answer generation

Author: Based on code by Daniel Bourke
License: MIT
"""

__version__ = "2.0.0"
__author__ = "RAG Pipeline Team"

from .core.pipeline import RAGPipeline
from .core.config import AppConfig, PipelineStage
from .interface import interactive_mode, sample_queries_mode, single_query_mode

__all__ = [
    "RAGPipeline",
    "AppConfig", 
    "PipelineStage",
    "interactive_mode",
    "sample_queries_mode", 
    "single_query_mode"
]
