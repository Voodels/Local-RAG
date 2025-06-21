"""
Text processing utilities
"""

import re
import textwrap
from typing import List


def text_formatter(text: str) -> str:
    """
    Clean and format text from PDF
    
    Args:
        text: Raw text string
        
    Returns:
        str: Cleaned text
    """
    cleaned_text = text.replace("\n", " ").strip()
    # Remove extra spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text


def split_list(input_list: List, slice_size: int) -> List[List]:
    """
    Split a list into chunks of specified size
    
    Args:
        input_list: List to split
        slice_size: Size of each chunk
        
    Returns:
        List[List]: List of chunks
    """
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]


def wrap_text(text: str, width: int = 80) -> str:
    """
    Wrap text to specified width
    
    Args:
        text: Text to wrap
        width: Maximum width
        
    Returns:
        str: Wrapped text
    """
    return "\n".join(textwrap.wrap(text, width=width))


def clean_generated_text(text: str, prompt: str = "") -> str:
    """
    Clean generated text by removing prompt and special tokens
    
    Args:
        text: Generated text
        prompt: Original prompt to remove
        
    Returns:
        str: Cleaned text
    """
    # Remove prompt
    cleaned = text.replace(prompt, "")
    
    # Remove special tokens
    cleaned = cleaned.replace("<bos>", "").replace("<eos>", "")
    
    # Remove common generation artifacts
    cleaned = cleaned.replace("Sure, here is the answer to the user query:\n\n", "")
    
    return cleaned.strip()
