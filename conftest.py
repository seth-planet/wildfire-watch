#!/usr/bin/env python3.12
"""
Root conftest.py for strict test collection control.

This file ensures that pytest collection respects Python version boundaries
and doesn't accidentally collect tests from incompatible directories.
"""

import os
import sys
from pathlib import Path

def pytest_ignore_collect(collection_path, config):
    """
    Programmatically ignore test collection from problematic directories.
    
    This is the most aggressive way to prevent pytest from collecting tests
    from directories that have incompatible Python version dependencies.
    """
    # Convert to absolute path string for consistent comparison
    path_str = str(collection_path.absolute()) if hasattr(collection_path, 'absolute') else str(collection_path)
    
    # Get the current Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    
    # Directories that should NEVER be collected by any Python version
    always_ignore = [
        '__pycache__',
        '.git',
        '.pytest_cache',
        'venv',
        '.venv',
        'node_modules',
        'build',
        'dist',
        '.tox'
    ]
    
    # Check for always-ignored directories
    for ignore_dir in always_ignore:
        if f"/{ignore_dir}/" in path_str or path_str.endswith(f"/{ignore_dir}"):
            return True
    
    # Python version-specific exclusions
    if python_version == "3.12":
        # Python 3.12 should collect from tests/ directory
        project_root = str(Path(__file__).parent)
        tests_dir = os.path.join(project_root, "tests")
        
        # If the path is not under the tests directory, ignore it
        if not path_str.startswith(tests_dir):
            # But allow collection from the root directory itself for conftest.py
            if path_str == project_root or path_str.endswith("conftest.py"):
                return False
            return True
        
        # Within tests directory, exclude specific Python version dependent files
        if path_str.endswith("test_yolo_nas_training.py") or \
           path_str.endswith("test_yolo_nas_training_updated.py") or \
           path_str.endswith("test_api_usage.py") or \
           path_str.endswith("test_qat_functionality.py") or \
           path_str.endswith("test_int8_quantization.py"):
            return True
            
    elif python_version == "3.10":
        # Python 3.10 should collect YOLO-NAS tests and specific test files
        yolo_nas_dirs = [
            "converted_models/YOLO-NAS-pytorch/tests",
            "tests/test_yolo_nas_training.py",
            "tests/test_api_usage.py", 
            "tests/test_qat_functionality.py"
        ]
        
        # Allow collection only from specific directories/files
        for allowed_path in yolo_nas_dirs:
            if allowed_path in path_str:
                return False
        
        # If not in allowed paths, check if it's a tests/ directory file that's not version-specific
        if "/tests/" in path_str and not any(x in path_str for x in ["coral", "tflite", "hardware", "deployment", "model_converter"]):
            return False
            
        return True
        
    elif python_version == "3.8":
        # Python 3.8 should collect Coral TPU and hardware tests
        coral_tests = [
            "test_model_converter",
            "test_hardware_integration", 
            "test_deployment",
            "test_int8_quantization",
            "coral",
            "tflite"
        ]
        
        # Allow collection only if path contains coral/hardware-related tests
        for coral_test in coral_tests:
            if coral_test in path_str:
                return False
                
        # Block everything else
        return True
    
    # Default: don't ignore
    return False

def pytest_configure(config):
    """Configure pytest with strict collection rules."""
    # Add markers if they don't exist
    markers_to_add = [
        "python312: Tests requiring Python 3.12",
        "python310: Tests requiring Python 3.10", 
        "python38: Tests requiring Python 3.8",
        "yolo_nas: YOLO-NAS training tests",
        "coral_tpu: Coral TPU tests",
        "slow: Slow tests",
        "integration: Integration tests"
    ]
    
    for marker in markers_to_add:
        config.addinivalue_line("markers", marker)

def pytest_collection_modifyitems(config, items):
    """
    Modify collected items to ensure proper Python version filtering.
    
    This is a final safety check to remove any tests that shouldn't
    be running in the current Python version.
    """
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    
    # For Python 3.12, we don't need to filter in this function
    # The pytest_ignore_collect already handles file-level filtering
    # This function should only filter based on markers
    
    if python_version != "3.12":
        # For other Python versions, keep the existing logic
        items_to_remove = []
        
        for item in items:
            item_path = str(item.fspath)
            should_exclude = False
            
            if python_version == "3.10":
                # Only include YOLO-NAS/super-gradients tests
                include_patterns = ["yolo_nas", "super_gradients", "api_usage", "qat"]
                
                if not any(pattern in item_path.lower() or pattern in item.name.lower() for pattern in include_patterns):
                    # Check markers
                    markers = [marker.name for marker in item.iter_markers()]
                    if not any(marker in ["python310", "yolo_nas", "super_gradients", "api_usage"] for marker in markers):
                        should_exclude = True
                        
            elif python_version == "3.8":
                # Only include Coral/hardware tests
                include_patterns = ["coral", "tflite", "model_converter", "hardware", "deployment", "int8"]
                
                if not any(pattern in item_path.lower() or pattern in item.name.lower() for pattern in include_patterns):
                    # Check markers
                    markers = [marker.name for marker in item.iter_markers()]
                    if not any(marker in ["python38", "coral_tpu", "tflite_runtime"] for marker in markers):
                        should_exclude = True
            
            if should_exclude:
                items_to_remove.append(item)
        
        # Remove excluded items
        for item in items_to_remove:
            items.remove(item)
            
        if items_to_remove:
            print(f"Excluded {len(items_to_remove)} tests incompatible with Python {python_version}")

def pytest_sessionstart(session):
    """Log test session start with Python version info."""
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"Starting pytest session with Python {python_version}")
    print(f"Test collection will be restricted to Python {python_version} compatible tests")