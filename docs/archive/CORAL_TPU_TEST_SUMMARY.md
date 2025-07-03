# Coral TPU Test Summary

## Overview
Enhanced the existing `test_hardware_inference.py` file with comprehensive Coral TPU hardware tests that use real hardware without mocking.

## Key Improvements

### 1. Enhanced Hardware Detection
- Added `test_coral_hardware_detection()` that checks for:
  - USB Coral devices (vendor IDs: 1a6e:089a, 18d1:9302)
  - PCIe Coral devices via lspci
  - /dev/apex* device files
- Provides detailed hardware inventory

### 2. Runtime Dependency Verification
- Added `test_coral_runtime_dependencies()` to verify:
  - tflite_runtime is installed (Python 3.8 only)
  - pycoral is installed with all required adapters
- Clear error messages if dependencies are missing

### 3. Edge TPU Compiler Testing
- Added `test_coral_model_compilation()` that:
  - Finds TFLite models without Edge TPU compilation
  - Runs edgetpu_compiler with optimization flags
  - Parses compilation statistics
  - Handles unsupported operations gracefully

### 4. Multi-Model Size Support
- Added `test_coral_multi_model_sizes()` to test:
  - 320x320 models (recommended for Coral TPU)
  - 416x416 models (balanced performance)
  - 640x640 models (maximum accuracy)
- Provides recommendations based on available models

### 5. Camera Integration Improvements
- Updated `_capture_camera_frame()` to use CameraDetector
- Discovers cameras on correct subnet (192.168.5.0/24)
- No hardcoded IPs - uses dynamic discovery
- Proper credential handling via environment variables

## Setup Script
Created `scripts/setup_coral_tpu.sh` that:
- Checks Python 3.8 availability
- Installs system dependencies
- Adds Coral repository
- Installs Edge TPU runtime and compiler
- Installs Python packages (tflite-runtime, pycoral)
- Sets up USB permissions
- Downloads test models
- Runs verification tests

## Running the Tests

### With Python 3.8 (Required for Coral TPU):
```bash
# Run all Coral TPU tests
python3.8 -m pytest tests/test_hardware_inference.py::TestCoralTPUInference -v --timeout=1800

# Run specific test
python3.8 -m pytest tests/test_hardware_inference.py::TestCoralTPUInference::test_coral_tpu_inference_performance -v
```

### With scripts/run_tests_by_python_version.sh:
```bash
# Automatically selects Python 3.8 for Coral tests
./scripts/run_tests_by_python_version.sh --test tests/test_hardware_inference.py::TestCoralTPUInference
```

## Performance Targets
- Average inference: < 25ms
- 95th percentile: < 30ms
- 320x320 models: < 20ms average

## Hardware Requirements
- Google Coral USB Accelerator or PCIe Accelerator
- USB 3.0 port (for USB version)
- Python 3.8 (tflite_runtime compatibility)
- Linux x86_64 or aarch64

## Next Steps
- Continue with TensorRT GPU tests
- Create comprehensive E2E hardware integration tests
- Test with real fire detection scenarios