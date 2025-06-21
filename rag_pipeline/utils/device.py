"""
Device utilities for managing GPU/CPU operations
"""

import torch
from rich.console import Console

console = Console()


def get_device(device_preference: str) -> str:
    """
    Determine the actual device to use based on preference and availability
    
    Args:
        device_preference: Device preference ('auto', 'cpu', 'cuda')
        
    Returns:
        str: Actual device to use
    """
    if device_preference.lower() == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_preference.lower()
    
    return device


def print_device_info(device: str) -> None:
    """
    Print information about the device being used
    
    Args:
        device: Device string ('cpu' or 'cuda')
    """
    if device == "cuda" and torch.cuda.is_available():
        console.print(f"[blue]GPU: {torch.cuda.get_device_name(0)}[/blue]")
        console.print(f"[blue]GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB[/blue]")
    else:
        console.print(f"[blue]Using device: {device}[/blue]")


def get_model_memory_info(model) -> dict:
    """
    Get memory usage information for a model
    
    Args:
        model: PyTorch model
        
    Returns:
        dict: Memory information
    """
    num_params = sum([param.numel() for param in model.parameters()])
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_buffers = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    model_mem_gb = (mem_params + mem_buffers) / (1024**3)
    
    return {
        "num_parameters": num_params,
        "memory_gb": model_mem_gb,
        "memory_params": mem_params,
        "memory_buffers": mem_buffers
    }
