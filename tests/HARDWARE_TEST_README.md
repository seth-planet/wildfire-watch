# Hardware Integration Test Guide

## Python Version Requirements

Different hardware accelerators require specific Python versions due to library compatibility:

### Coral TPU Tests
- **Required Python Version**: 3.8
- **Reason**: `tflite_runtime` and `pycoral` only support Python 3.8
- **Run with**: `python3.8 -m pytest tests/test_*coral*.py`
- **Example**: 
  ```bash
  CAMERA_CREDENTIALS=admin:password python3.8 -m pytest tests/test_e2e_coral_frigate.py -xvs
  ```

### Hailo Tests  
- **Required Python Version**: 3.10
- **Reason**: `hailo-python` SDK requires Python 3.10
- **Run with**: `python3.10 -m pytest tests/test_*hailo*.py`
- **Example**:
  ```bash
  python3.10 -m pytest tests/test_hailo_nms_working.py -xvs
  ```

### TensorRT/GPU Tests
- **Compatible Python Versions**: 3.10 or 3.12
- **Reason**: TensorRT and PyTorch support both versions
- **Run with**: `python3.12 -m pytest tests/test_*tensorrt*.py` (recommended)
- **Example**:
  ```bash
  python3.12 -m pytest tests/test_tensorrt_gpu_integration.py -xvs
  ```

### General Tests
- **Default Python Version**: 3.12
- **Reason**: Project standard, best compatibility with latest libraries
- **Run with**: `python3.12 -m pytest tests/`

## Common Issues

### Wrong Python Version
If you see errors like:
- `ModuleNotFoundError: No module named 'pycoral'` - You're using Python 3.12 instead of 3.8
- `Failed to load delegate from libedgetpu.so.1` - May be Python version mismatch
- Test skipped with "Coral requires Python 3.8" - Run with python3.8

### Hardware In Use
If you see:
- `Coral TPU devices are in use by other processes` - Stop Frigate or other services
- Use `ps aux | grep coral` to find processes
- Use `docker stop frigate` if Frigate is running

### Camera Credentials
Always pass camera credentials via environment:
```bash
CAMERA_CREDENTIALS=admin:password python3.8 -m pytest tests/test_e2e_*.py
```

## Test Scripts Helper

Use the provided test runner script that automatically selects the correct Python version:
```bash
./scripts/run_tests_by_python_version.sh --test tests/test_e2e_coral_frigate.py
```

This script will:
1. Detect which Python version is required
2. Check if it's installed
3. Run the test with the correct interpreter