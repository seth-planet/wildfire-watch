# Wildfire Watch Test Suite

## Python Version Requirements

**IMPORTANT**: Different tests require different Python versions due to library dependencies:

### Python 3.12 (Default)
Most tests use Python 3.12. This includes:
- Core logic tests: `test_consensus.py`, `test_trigger.py`, `test_telemetry.py`, `test_core_logic.py`
- Integration tests: `test_integration_e2e.py`, `test_integration_docker.py`, `test_simplified_integration.py`
- E2E tests: `test_e2e_fire_detection_full.py`, `test_e2e_working_integration.py`
- Service tests: `test_camera_detector.py`, `test_mqtt_broker.py`
- Other tests: `test_scripts.py`, `test_new_features.py`, `test_consensus_debug.py`, `test_consensus_enhanced.py`
- TLS tests: `test_tls_integration.py`, `test_tls_integration_consolidated.py`

### Python 3.10 (YOLO-NAS/super-gradients)
Tests requiring `super-gradients` library:
- `test_api_usage.py`
- `test_yolo_nas_training.py`
- `test_yolo_nas_training_updated.py`
- `test_qat_functionality.py`
- `test_int8_quantization.py` (also uses tflite_runtime)
- `test_frigate_integration.py` (also uses tflite_runtime)

### Python 3.8 (Coral TPU/tflite_runtime)
Tests requiring `tflite_runtime` library:
- `test_model_converter.py`
- `test_model_converter_e2e.py`
- `test_model_converter_hardware.py`
- `test_hardware_integration.py`
- `test_deployment.py`
- `test_security_nvr_integration.py`
- `test_security_nvr_documentation.py`

## Running Tests

### Run all Python 3.12 tests:
```bash
python3.12 -m pytest tests/ -v --timeout=300 -k "not (api_usage or yolo_nas or qat_functionality or int8_quantization or frigate_integration or model_converter or hardware_integration or deployment or security_nvr)"
```

### Run Python 3.10 tests (YOLO-NAS):
```bash
python3.10 -m pytest tests/ -v --timeout=300 -k "api_usage or yolo_nas or qat_functionality"
```

### Run Python 3.8 tests (Coral TPU):
**Note**: Python 3.8 typically only has `tflite_runtime` installed, not pytest. These tests should be run with Python 3.12 unless they specifically import `tflite_runtime`. If a test needs both pytest and tflite_runtime:
1. Install pytest for Python 3.8: `python3.8 -m pip install pytest`
2. Or refactor the test to separate tflite_runtime imports
3. Or run with Python 3.12 and mock the tflite_runtime imports

```bash
# If pytest is installed for Python 3.8:
python3.8 -m pytest tests/ -v --timeout=300 -k "model_converter or hardware_integration or deployment"

# Otherwise, run with Python 3.12:
python3.12 -m pytest tests/ -v --timeout=300 -k "model_converter or hardware_integration or deployment"
```

### Run tests that require both Python 3.10 and 3.8:
For tests like `test_int8_quantization.py` and `test_frigate_integration.py` that import both super_gradients and tflite_runtime:
- These tests may need to be refactored to separate the dependencies
- Or run with the Python version that has both libraries available
- Consider using conditional imports or splitting into separate test files

### Run specific test file:
```bash
python3.12 -m pytest tests/test_consensus.py -v --timeout=300
```

## Test Timeouts

Some tests may require longer timeouts, especially:
- Model conversion tests (up to 60 minutes for TensorRT)
- Integration tests with multiple services
- Hardware tests with real devices

Use `--timeout=600` or higher for these tests.

## Skipping Tests

Tests can be skipped if they require:
- Specific hardware (Coral TPU, GPIO pins)
- Docker environment
- Network access
- Large model files

Use pytest markers to skip:
```python
@pytest.mark.skipif(not has_coral_tpu(), reason="Requires Coral TPU hardware")
```

## Common Issues

1. **ModuleNotFoundError: No module named 'super_gradients'**
   - Use Python 3.10 instead of Python 3.12
   
2. **ModuleNotFoundError: No module named 'tflite_runtime'**
   - Use Python 3.8 instead of Python 3.12
   
3. **Timeout errors**
   - Increase timeout with `--timeout=600` or higher
   - Some model conversion tests legitimately take 30-60 minutes

4. **Docker-related failures**
   - Ensure Docker daemon is running
   - Check if services can be started with `docker-compose up`

5. **Hardware test failures**
   - These tests require actual hardware (Raspberry Pi, Coral TPU)
   - Can be skipped in CI/CD environments