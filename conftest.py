#!/usr/bin/env python3.12
"""
Root conftest.py for strict test collection control.

This file ensures that pytest collection respects Python version boundaries
and doesn't accidentally collect tests from incompatible directories.
"""

import os
import sys
from pathlib import Path
import pytest
import threading
import time

# Ensure project root is in sys.path for imports to work correctly
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Also ensure that subdirectories can be imported without needing relative imports
# This allows tests to import from fire_consensus, gpio_trigger, etc.
for subdir in ['fire_consensus', 'gpio_trigger', 'camera_detector', 'cam_telemetry', 'security_nvr']:
    subdir_path = os.path.join(project_root, subdir)
    if os.path.exists(subdir_path) and subdir_path not in sys.path:
        sys.path.insert(0, subdir_path)

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
            "tests/test_yolo_nas_training_updated.py",
            "tests/test_yolo_nas_qat_hailo_e2e.py",
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


# GPIO State Isolation Fixtures
@pytest.fixture(scope="session")
def gpio_state_manager():
    """Session-level fixture to manage GPIO state across all tests."""
    class GPIOStateManager:
        def __init__(self):
            self.initial_states = {}
            self.lock = threading.Lock()
            
        def save_initial_state(self):
            """Save the initial GPIO state at the beginning of the session."""
            try:
                from gpio_trigger.trigger import GPIO
                if hasattr(GPIO, '_state') and hasattr(GPIO, '_mode'):
                    with self.lock:
                        self.initial_states = {
                            'state': GPIO._state.copy() if GPIO._state else {},
                            'mode': GPIO._mode.copy() if GPIO._mode else {},
                            'pull': GPIO._pull.copy() if hasattr(GPIO, '_pull') and GPIO._pull else {}
                        }
            except Exception as e:
                print(f"Could not save initial GPIO state: {e}")
                
        def restore_initial_state(self):
            """Restore GPIO to initial state."""
            try:
                from gpio_trigger.trigger import GPIO
                if hasattr(GPIO, '_state') and hasattr(GPIO, '_mode'):
                    with GPIO._lock:
                        # Clear current state
                        GPIO._state.clear()
                        GPIO._mode.clear()
                        if hasattr(GPIO, '_pull'):
                            GPIO._pull.clear()
                        
                        # Restore initial state
                        if self.initial_states:
                            GPIO._state.update(self.initial_states.get('state', {}))
                            GPIO._mode.update(self.initial_states.get('mode', {}))
                            if hasattr(GPIO, '_pull'):
                                GPIO._pull.update(self.initial_states.get('pull', {}))
            except Exception as e:
                print(f"Could not restore GPIO state: {e}")
    
    manager = GPIOStateManager()
    manager.save_initial_state()
    
    yield manager
    
    # Session cleanup
    manager.restore_initial_state()


@pytest.fixture(scope="module", autouse=True)
def gpio_module_cleanup(gpio_state_manager):
    """Module-level fixture to clean GPIO state between test modules."""
    yield
    
    # Clean up after each module
    try:
        from gpio_trigger.trigger import GPIO
        if hasattr(GPIO, '_state') and hasattr(GPIO, '_lock'):
            with GPIO._lock:
                # Reset all pins to LOW
                for pin in list(GPIO._state.keys()):
                    if pin in GPIO._mode and GPIO._mode.get(pin) == GPIO.OUT:
                        GPIO._state[pin] = GPIO.LOW
                        
                # Call cleanup to reset hardware
                if hasattr(GPIO, 'cleanup'):
                    GPIO.cleanup()
                    
                # IMPORTANT: Force clear all internal state dictionaries after cleanup
                # This ensures no state persists between modules
                if hasattr(GPIO, '_state'):
                    GPIO._state.clear()
                if hasattr(GPIO, '_mode'):
                    GPIO._mode.clear()
                if hasattr(GPIO, '_pull'):
                    GPIO._pull.clear()
                if hasattr(GPIO, '_edge_callbacks'):
                    GPIO._edge_callbacks.clear()
                    
                # Re-initialize for next module
                if hasattr(GPIO, 'setmode'):
                    GPIO.setmode(GPIO.BCM)
                    GPIO.setwarnings(False)
    except Exception as e:
        print(f"GPIO module cleanup error: {e}")


@pytest.fixture(autouse=True)
def ensure_gpio_cleanup():
    """Function-level fixture to ensure GPIO is properly cleaned up after each test."""
    # Pre-test setup
    initial_gpio_state = None
    try:
        from gpio_trigger.trigger import GPIO
        if hasattr(GPIO, '_state') and hasattr(GPIO, '_lock'):
            with GPIO._lock:
                initial_gpio_state = GPIO._state.copy() if GPIO._state else {}
    except Exception:
        pass
    
    yield
    
    # Post-test cleanup
    try:
        from gpio_trigger.trigger import GPIO
        if hasattr(GPIO, '_state') and hasattr(GPIO, '_lock'):
            with GPIO._lock:
                # Reset all output pins to LOW
                for pin in list(GPIO._state.keys()):
                    if pin in GPIO._mode and GPIO._mode.get(pin) == GPIO.OUT:
                        GPIO._state[pin] = GPIO.LOW
                        
                # If we had an initial state, restore non-output pins
                if initial_gpio_state:
                    for pin, value in initial_gpio_state.items():
                        if pin in GPIO._mode and GPIO._mode.get(pin) == GPIO.IN:
                            GPIO._state[pin] = value
    except Exception as e:
        print(f"GPIO cleanup error: {e}")


@pytest.fixture(autouse=True)
def cleanup_controller_instances():
    """Ensure controller instances are cleaned up between tests."""
    yield
    
    # Clean up any lingering controller instances
    try:
        import gpio_trigger.trigger as trigger_module
        if hasattr(trigger_module, 'controller') and trigger_module.controller:
            try:
                trigger_module.controller._shutdown = True
                trigger_module.controller.cleanup()
                trigger_module.controller = None
            except Exception:
                pass
    except Exception:
        pass