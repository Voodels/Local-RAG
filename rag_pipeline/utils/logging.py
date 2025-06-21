"""
Logging utilities for the RAG pipeline
"""

import sys
from loguru import logger


def setup_logger(log_level: str = "INFO", log_file: str = None) -> None:
    """
    Setup logger configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    # Remove default logger
    logger.remove()
    
    # Add stderr logger
    logger.add(
        sys.stderr, 
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level
    )
    
    # Add file logger if specified
    if log_file:
        logger.add(
            log_file,
            rotation="10 MB", 
            retention="7 days",
            level=log_level
        )
    else:
        # Default log file with timestamp
        logger.add(
            "rag_pipeline_{time}.log", 
            rotation="10 MB", 
            retention="7 days",
            level=log_level
        )


def get_logger():
    """Get the configured logger instance"""
    return logger
