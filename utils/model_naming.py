#!/usr/bin/env python3.12
"""Common model naming utilities for consistent naming across the project.

This module provides standardized model naming conventions following the pattern:
    wildfire_<size>_<accelerator>_<precision>.<format>

Where:
    - size: nano, small, medium, large, xlarge
    - accelerator: tensorrt, hailo, coral, openvino, tflite, onnx
    - precision: int8, fp16, fp32
    - format: engine, hef, tflite, xml, onnx
"""

from typing import Tuple, Optional, Dict
from pathlib import Path


# Model size categories based on resolution
SIZE_CATEGORIES = {
    'nano': {'resolution': 320, 'params': '3M'},
    'small': {'resolution': 416, 'params': '8M'},
    'medium': {'resolution': 640, 'params': '25M'},
    'large': {'resolution': 640, 'params': '50M'},
    'xlarge': {'resolution': 1280, 'params': '100M'}
}

# Accelerator format mappings
ACCELERATOR_FORMATS = {
    'tensorrt': 'engine',
    'hailo': 'hef',
    'coral': 'tflite',
    'openvino': 'xml',
    'tflite': 'tflite',
    'onnx': 'onnx'
}

# Precision mappings
PRECISION_MAP = {
    'fp16': 'fp16',
    'fp32': 'fp32',
    'int8': 'int8',
    'int8_qat': 'int8',  # QAT models are still INT8
    'int8_ptq': 'int8'   # PTQ models are still INT8
}


def get_size_category(resolution: int) -> str:
    """Determine model size category based on resolution.
    
    Args:
        resolution: Model input resolution (width or height)
        
    Returns:
        str: Size category (nano, small, medium, large, xlarge)
    """
    if resolution <= 320:
        return 'nano'
    elif resolution <= 416:
        return 'small'
    elif resolution <= 640:
        # Distinguish between medium and large based on model complexity
        # This would need additional info in practice
        return 'medium'
    elif resolution <= 1024:
        return 'large'
    else:
        return 'xlarge'


def get_model_filename(size: str, accelerator: str, precision: str) -> str:
    """Generate standardized model filename.
    
    Args:
        size: Model size category (nano, small, medium, large, xlarge)
        accelerator: Target accelerator (tensorrt, hailo, coral, etc.)
        precision: Model precision (int8, fp16, fp32)
        
    Returns:
        str: Standardized filename like 'wildfire_large_tensorrt_int8.engine'
    """
    # Normalize inputs
    size = size.lower()
    accelerator = accelerator.lower()
    precision = PRECISION_MAP.get(precision.lower(), precision.lower())
    
    # Get file extension
    extension = ACCELERATOR_FORMATS.get(accelerator, 'bin')
    
    return f"wildfire_{size}_{accelerator}_{precision}.{extension}"


def get_model_path(model_dir: Path, size: str, accelerator: str, precision: str) -> Path:
    """Get full model path.
    
    Args:
        model_dir: Directory where models are stored
        size: Model size category
        accelerator: Target accelerator
        precision: Model precision
        
    Returns:
        Path: Full path to model file
    """
    filename = get_model_filename(size, accelerator, precision)
    return model_dir / filename


def parse_model_filename(filename: str) -> Optional[Dict[str, str]]:
    """Parse a model filename to extract components.
    
    Args:
        filename: Model filename to parse
        
    Returns:
        dict: Components (size, accelerator, precision) or None if invalid
    """
    parts = Path(filename).stem.split('_')
    
    # Expected format: wildfire_<size>_<accelerator>_<precision>
    if len(parts) != 4 or parts[0] != 'wildfire':
        return None
        
    return {
        'size': parts[1],
        'accelerator': parts[2],
        'precision': parts[3]
    }


def get_model_url(repo_base: str, size: str, accelerator: str, precision: str) -> str:
    """Generate model download URL.
    
    Args:
        repo_base: Base repository URL
        size: Model size category
        accelerator: Target accelerator
        precision: Model precision
        
    Returns:
        str: Full URL to download model
    """
    filename = get_model_filename(size, accelerator, precision)
    return f"{repo_base}/models/{filename}"


def determine_model_size_for_hardware(gpu_memory_mb: int = 0, 
                                     coral_count: int = 0,
                                     has_hailo: bool = False) -> str:
    """Determine optimal model size based on hardware capabilities.
    
    Args:
        gpu_memory_mb: GPU memory in MB
        coral_count: Number of Coral TPUs
        has_hailo: Whether Hailo accelerator is present
        
    Returns:
        str: Recommended model size category
    """
    # GPU-based selection
    if gpu_memory_mb > 0:
        vram_gb = gpu_memory_mb / 1024
        if vram_gb >= 8:
            return 'xlarge'
        elif vram_gb >= 4:
            return 'large'
        else:
            return 'medium'
    
    # Hailo can handle larger models
    if has_hailo:
        return 'large'
    
    # Coral TPU selection
    if coral_count >= 2:
        return 'medium'  # Multiple TPUs can pipeline
    elif coral_count == 1:
        return 'small'   # Single TPU is memory limited
    
    # Default to nano for CPU
    return 'nano'


def list_available_models(model_dir: Path) -> Dict[str, list]:
    """List all available models organized by size.
    
    Args:
        model_dir: Directory to search for models
        
    Returns:
        dict: Models organized by size category
    """
    models = {}
    
    for model_file in model_dir.glob("wildfire_*.???"):
        parsed = parse_model_filename(model_file.name)
        if parsed:
            size = parsed['size']
            if size not in models:
                models[size] = []
            models[size].append({
                'filename': model_file.name,
                'accelerator': parsed['accelerator'],
                'precision': parsed['precision'],
                'size_mb': model_file.stat().st_size / (1024 * 1024)
            })
    
    return models