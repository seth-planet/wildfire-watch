# Python Version Testing Configuration

## Overview

The Wildfire Watch test suite automatically routes tests to the correct Python version based on their dependencies and requirements. This ensures that tests run with the appropriate Python environment without manual intervention.

## Python Version Requirements

### Python 3.12 (Default - Most Tests)
**Used for**: Core wildfire-watch services and integration tests
- **Services**: camera_detector, fire_consensus, gpio_trigger, cam_telemetry
- **Libraries**: paho-mqtt, opencv-python, onvif, netifaces, pytest
- **Test Types**: Unit tests, integration tests, MQTT tests, service tests

**Test Files**:
- `test_consensus.py` - Fire consensus logic
- `test_detect.py` - Camera detection and discovery  
- `test_trigger.py` - GPIO trigger and pump control
- `test_telemetry.py` - Telemetry and health monitoring
- `test_mqtt_broker.py` - MQTT broker functionality
- `test_integration_e2e.py` - End-to-end integration
- `test_simplified_integration.py` - Simplified integration tests
- `test_tls_integration.py` - TLS security tests

### Python 3.10 (YOLO-NAS and Training)
**Used for**: AI model training and super-gradients library
- **Libraries**: super-gradients, torch, torchvision (specific versions)
- **Test Types**: Model training, API validation, quantization-aware training

**Test Files**:
- `test_yolo_nas_training.py` - YOLO-NAS model training
- `test_api_usage.py` - API usage validation
- `test_qat_functionality.py` - Quantization-aware training

**Markers**: `yolo_nas`, `super_gradients`, `api_usage`, `qat_functionality`

### Python 3.8 (Coral TPU and Edge Hardware)
**Used for**: Edge hardware integration and model deployment
- **Libraries**: tflite-runtime, pycoral, edgetpu (Coral-specific)
- **Test Types**: Hardware integration, model conversion, deployment

**Test Files**:
- `test_model_converter.py` - Model format conversion
- `test_hardware_integration.py` - Hardware-specific tests
- `test_deployment.py` - Production deployment tests
- `test_int8_quantization.py` - INT8 model quantization

**Markers**: `coral_tpu`, `tflite_runtime`, `model_converter`, `hardware_integration`, `deployment`

## Configuration Files

### 1. Version-Specific pytest.ini Files

#### `pytest-python312.ini`
```ini
[tool:pytest]
testpaths = tests
addopts = 
    -v --tb=short --strict-markers --color=yes
    -m "not (yolo_nas or super_gradients or coral_tpu or tflite_runtime)"
timeout = 3600
```

#### `pytest-python310.ini`
```ini
[tool:pytest]
testpaths = tests
addopts = 
    -v --tb=short --strict-markers --color=yes
    -m "yolo_nas or super_gradients or api_usage or qat_functionality"
timeout = 7200  # Longer for training
```

#### `pytest-python38.ini`
```ini
[tool:pytest]
testpaths = tests
addopts = 
    -v --tb=short --strict-markers --color=yes
    -m "coral_tpu or tflite_runtime or model_converter or hardware_integration"
timeout = 7200  # Longer for model conversion
```

### 2. Automatic Test Routing Plugin

**`tests/pytest_python_versions.py`** - Analyzes test files and automatically assigns Python version markers based on:
- Import statements (e.g., `super_gradients` → Python 3.10)
- Test file names (e.g., `test_yolo_nas_*` → Python 3.10)  
- Test function names (e.g., `test_coral_*` → Python 3.8)
- Keywords in test content

## Usage

### 1. Automatic Test Runner (Recommended)

```bash
# Run all tests with correct Python versions automatically
./scripts/run_tests_by_python_version.sh --all

# Run specific Python version tests
./scripts/run_tests_by_python_version.sh --python312
./scripts/run_tests_by_python_version.sh --python310  
./scripts/run_tests_by_python_version.sh --python38

# Run specific test with auto-detection
./scripts/run_tests_by_python_version.sh --test tests/test_yolo_nas_training.py

# Validate Python environment
./scripts/run_tests_by_python_version.sh --validate
```

### 2. Manual Configuration-Based Testing

```bash
# Python 3.12 tests (most tests)
python3.12 -m pytest -c pytest-python312.ini

# Python 3.10 tests (YOLO-NAS)
python3.10 -m pytest -c pytest-python310.ini

# Python 3.8 tests (Coral TPU)
python3.8 -m pytest -c pytest-python38.ini
```

### 3. Marker-Based Testing

```bash
# Run tests marked for specific Python versions
python3.12 -m pytest -m "python312"
python3.10 -m pytest -m "python310" 
python3.8 -m pytest -m "python38"

# Run specific technology tests
python3.10 -m pytest -m "yolo_nas"
python3.8 -m pytest -m "coral_tpu"
```

## Environment Setup

### Prerequisites

Install required Python versions:

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.12 python3.10 python3.8
sudo apt install python3.12-venv python3.10-venv python3.8-venv

# Or use pyenv for version management
pyenv install 3.12.11
pyenv install 3.10.14
pyenv install 3.8.19
```

### Python-Specific Dependencies

#### Python 3.12 (Default)
```bash
python3.12 -m pip install -r requirements-base.txt
python3.12 -m pip install pytest pytest-timeout pytest-mock
```

#### Python 3.10 (YOLO-NAS)
```bash
python3.10 -m pip install super-gradients torch torchvision
python3.10 -m pip install pytest pytest-timeout pytest-mock
```

#### Python 3.8 (Coral TPU)
```bash
python3.8 -m pip install tflite-runtime pycoral
python3.8 -m pip install pytest pytest-timeout pytest-mock
```

## Adding New Tests

### 1. Automatic Detection

Tests are automatically assigned to the correct Python version based on:

```python
# This test will automatically use Python 3.10
import super_gradients
def test_yolo_nas_training():
    pass

# This test will automatically use Python 3.8  
import tflite_runtime
def test_coral_inference():
    pass
```

### 2. Explicit Markers

For manual control, use pytest markers:

```python
import pytest

@pytest.mark.python310
@pytest.mark.yolo_nas
def test_custom_training():
    """Test that explicitly requires Python 3.10"""
    pass

@pytest.mark.python38
@pytest.mark.coral_tpu  
def test_hardware_specific():
    """Test that explicitly requires Python 3.8"""
    pass
```

### 3. Test File Naming Conventions

Use naming patterns for automatic detection:

- `test_yolo_nas_*.py` → Python 3.10
- `test_coral_*.py` → Python 3.8
- `test_tflite_*.py` → Python 3.8
- `test_model_converter*.py` → Python 3.8
- Everything else → Python 3.12

## Continuous Integration

### GitHub Actions Example

```yaml
name: Test Suite
on: [push, pull_request]

jobs:
  test-python312:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - run: ./scripts/run_tests_by_python_version.sh --python312

  test-python310:
    runs-on: ubuntu-latest  
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: ./scripts/run_tests_by_python_version.sh --python310

  test-python38:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.8'
      - run: ./scripts/run_tests_by_python_version.sh --python38
```

## Troubleshooting

### Problem: Test runs with wrong Python version
**Solution**: Check imports and add explicit marker
```python
@pytest.mark.python310  # Force specific version
def test_function():
    pass
```

### Problem: Python version not found
**Solution**: Install missing Python version or skip tests
```bash
# Check available versions
./scripts/run_tests_by_python_version.sh --validate

# Install missing version (Ubuntu)
sudo apt install python3.10-dev python3.10-venv
```

### Problem: Dependencies missing for Python version
**Solution**: Install version-specific dependencies
```bash
# Python 3.10 for YOLO-NAS
python3.10 -m pip install super-gradients

# Python 3.8 for Coral
python3.8 -m pip install tflite-runtime
```

### Problem: Tests marked for wrong version
**Solution**: Update test markers or configuration
```bash
# Check what version a test requires
python3.12 scripts/run_tests_with_correct_python.py --analyze tests/test_example.py
```

## Benefits

1. ✅ **Automatic version selection** - No manual Python version management
2. ✅ **Dependency isolation** - Each Python version has its own dependencies  
3. ✅ **CI/CD friendly** - Easy to run tests in parallel by version
4. ✅ **Developer friendly** - Single command runs all tests correctly
5. ✅ **Maintainable** - Clear separation of version-specific requirements
6. ✅ **Scalable** - Easy to add new Python versions or requirements

The configuration ensures that tests always run with the correct Python version, preventing dependency conflicts and ensuring accurate test results across all components of the Wildfire Watch system.