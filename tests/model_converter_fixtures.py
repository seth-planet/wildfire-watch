#!/usr/bin/env python3.12
"""
Model Converter Test Fixtures with Caching Support

This module provides pytest fixtures for model converter tests that use
caching to dramatically reduce test execution time.
"""

import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Optional
import urllib.request

# Add converted_models to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'converted_models'))

from cached_model_converter import CachedModelConverter
from model_cache_manager import get_global_cache


@pytest.fixture(scope="session")
def model_cache():
    """Provide global model cache for all tests"""
    cache_dir = os.environ.get('MODEL_CACHE_DIR', 'cache')
    cache = get_global_cache(cache_dir)
    
    # Show cache stats at start
    stats = cache.get_cache_stats()
    print(f"\nModel cache initialized: {stats['total_entries']} entries, "
          f"{stats['total_size_mb']:.1f} MB used")
    
    yield cache
    
    # Show cache stats at end
    stats = cache.get_cache_stats()
    print(f"\nModel cache final: {stats['total_entries']} entries, "
          f"{stats['total_size_mb']:.1f} MB used")


@pytest.fixture(scope="session")
def test_models_dir(tmp_path_factory):
    """Create directory for test models"""
    models_dir = tmp_path_factory.mktemp("test_models")
    
    # Download minimal test models if not cached
    test_models = {
        'yolov8n.pt': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt',
        'yolov5n.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5n.pt'
    }
    
    for filename, url in test_models.items():
        model_path = models_dir / filename
        if not model_path.exists():
            print(f"Downloading test model: {filename}")
            try:
                urllib.request.urlretrieve(url, model_path)
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
    
    yield models_dir
    
    # Cleanup handled by pytest


@pytest.fixture(scope="session")
def calibration_data_dir(tmp_path_factory, model_cache):
    """Provide calibration data directory with caching"""
    # Check cache first
    cached_path = model_cache.get_cached_calibration('default')
    if cached_path:
        # Extract to temp directory
        cal_dir = tmp_path_factory.mktemp("calibration")
        if cached_path.endswith('.tgz') or cached_path.endswith('.tar.gz'):
            import tarfile
            with tarfile.open(cached_path, 'r:gz') as tar:
                tar.extractall(cal_dir)
        return cal_dir
    
    # Create minimal calibration data
    cal_dir = tmp_path_factory.mktemp("calibration")
    images_dir = cal_dir / "images"
    images_dir.mkdir()
    
    # Create dummy calibration images
    try:
        import numpy as np
        import cv2
        
        for i in range(10):
            img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(images_dir / f"cal_{i:04d}.jpg"), img)
    except ImportError:
        # Create empty files if cv2 not available
        for i in range(10):
            (images_dir / f"cal_{i:04d}.jpg").touch()
    
    return cal_dir


@pytest.fixture
def cached_converter(test_models_dir, tmp_path):
    """Create a cached model converter instance"""
    def _create_converter(model_name='yolov8n.pt', **kwargs):
        model_path = test_models_dir / model_name
        if not model_path.exists():
            pytest.skip(f"Test model {model_name} not available")
        
        output_dir = tmp_path / "output"
        output_dir.mkdir(exist_ok=True)
        
        converter = CachedModelConverter(
            model_path=str(model_path),
            output_dir=str(output_dir),
            use_cache=True,
            **kwargs
        )
        
        return converter
    
    return _create_converter


@pytest.fixture
def mock_heavy_conversions(monkeypatch):
    """Mock heavy conversion operations for fast unit tests"""
    
    def mock_tflite_conversion(*args, **kwargs):
        # Return a dummy TFLite model path
        output_path = Path(args[0]) if args else Path("dummy.tflite")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"DUMMY_TFLITE_MODEL")
        return str(output_path)
    
    def mock_tensorrt_conversion(*args, **kwargs):
        # Return a dummy TensorRT engine path
        output_path = Path(args[0]) if args else Path("dummy.engine")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"DUMMY_TENSORRT_ENGINE")
        return str(output_path)
    
    def mock_hailo_conversion(*args, **kwargs):
        # Return a dummy HEF path
        output_path = Path(args[0]) if args else Path("dummy.hef")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"DUMMY_HAILO_HEF")
        return str(output_path)
    
    # Apply mocks
    monkeypatch.setattr("convert_model.convert_to_tflite", mock_tflite_conversion)
    monkeypatch.setattr("convert_model.convert_to_tensorrt", mock_tensorrt_conversion)
    monkeypatch.setattr("convert_model.compile_hailo_model", mock_hailo_conversion)


@pytest.fixture
def pre_converted_models(model_cache, cached_converter):
    """Provide pre-converted models for fast testing"""
    # Models to pre-convert
    conversions = [
        ('yolov8n.pt', ['onnx'], [640]),
        ('yolov8n.pt', ['tflite'], [320]),  # Smaller size for faster conversion
    ]
    
    models = {}
    
    for model_name, formats, sizes in conversions:
        converter = cached_converter(model_name)
        results = converter.convert_all(formats=formats, sizes=sizes, validate=False)
        
        # Extract model paths from results
        for size_key, size_data in results.get('sizes', {}).items():
            for format_name, format_data in size_data.get('models', {}).items():
                if isinstance(format_data, dict) and 'path' in format_data:
                    key = f"{model_name}_{size_key}_{format_name}"
                    models[key] = format_data['path']
    
    return models


# Test markers for model converter tests
converter_quick = pytest.mark.timeout(60)  # 1 minute for quick tests
converter_slow = pytest.mark.timeout(300)  # 5 minutes for slow tests
converter_heavy = pytest.mark.timeout(1800)  # 30 minutes for heavy tests


def requires_gpu():
    """Skip test if GPU is not available"""
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
    except ImportError:
        pytest.skip("PyTorch not installed")


def requires_docker():
    """Skip test if Docker is not available"""
    import subprocess
    try:
        subprocess.run(['docker', '--version'], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("Docker not available")


def requires_hailo_sdk():
    """Skip test if Hailo SDK is not available"""
    hailo_sdk = os.environ.get('HAILO_SDK_PATH')
    if not hailo_sdk or not Path(hailo_sdk).exists():
        pytest.skip("Hailo SDK not available")