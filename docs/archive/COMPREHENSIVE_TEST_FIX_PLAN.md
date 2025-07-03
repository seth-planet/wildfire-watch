# Comprehensive Test Fix Plan for Wildfire Watch

## Executive Summary

This plan addresses all test violations to ensure proper integration testing with real hardware and no internal module mocking. The wildfire-watch project requires true integration tests that exercise real hardware capabilities (Coral TPU, TensorRT GPU, cameras) and use real MQTT brokers.

## Key Violations Found

### 1. Internal Module Mocking
- **test_api_usage.py**: Heavy use of mocks for super-gradients components
- **test_yolo_nas_training_updated.py**: Mocking internal trainer and model components
- **test_rtsp_validation_improved.py**: Mocking internal camera detector methods
- **test_rtsp_validation_timeout.py**: Mocking cv2.VideoCapture

### 2. Missing Real Hardware Testing
- No tests actually verify Coral TPU inference
- No tests verify TensorRT GPU performance
- No tests verify real camera RTSP streams
- No tests verify Hailo hardware (correctly skipped as not available)

### 3. MQTT Testing Issues
- All test files correctly use real MQTT broker ✓
- Good use of `test_mqtt_broker` fixture ✓
- Proper topic isolation with `mqtt_topic_factory` ✓

### 4. Environment Variable Issues
- Camera credentials hardcoded in some tests instead of using env vars
- Missing proper environment setup for hardware tests

## Phase 1: Fix Internal Mocking Violations

### 1.1 Fix test_api_usage.py
```python
# REMOVE all mocks of super-gradients components
# REPLACE with:
# - Create real temporary datasets
# - Use actual super-gradients API calls
# - Verify actual model training (even if just 1 epoch)
# - Test with real model architectures
```

### 1.2 Fix test_yolo_nas_training_updated.py
```python
# REMOVE mocking of:
# - super_gradients.Trainer
# - super_gradients.training.models.get
# - Dataloader factory functions
# REPLACE with:
# - Real dataset creation in temp directories
# - Actual model instantiation
# - Real training execution (1 epoch minimum)
```

### 1.3 Fix test_rtsp_validation_improved.py
```python
# REMOVE patches of internal methods
# ADD real camera testing:
# - Test with actual network cameras using CAMERA_CREDENTIALS env var
# - Test with mock RTSP server for CI/CD
# - Verify actual stream validation
```

### 1.4 Fix test_rtsp_validation_timeout.py
```python
# REMOVE cv2.VideoCapture mocking
# REPLACE with:
# - Real timeout testing using non-existent hosts
# - Process-based timeout implementation testing
# - Real network failure scenarios
```

## Phase 2: Add Real Hardware Testing

### 2.1 Coral TPU Tests
```python
# tests/test_coral_tpu_hardware.py
@pytest.mark.skipif(not has_coral_tpu(), reason="Coral TPU not available")
class TestCoralTPUHardware:
    def test_coral_model_loading(self):
        # Load actual .tflite model
        # Run inference on test image
        # Verify detection results
        
    def test_coral_inference_performance(self):
        # Measure actual inference time
        # Verify meets 15-20ms target
```

### 2.2 TensorRT GPU Tests
```python
# tests/test_tensorrt_hardware.py
@pytest.mark.skipif(not has_tensorrt(), reason="TensorRT not available")
class TestTensorRTHardware:
    def test_tensorrt_engine_creation(self):
        # Build actual TensorRT engine
        # Verify optimization levels
        
    def test_tensorrt_inference(self):
        # Run actual GPU inference
        # Verify 8-12ms performance target
```

### 2.3 Camera Hardware Tests
```python
# tests/test_camera_hardware.py
@pytest.mark.skipif(not has_camera_on_network(), reason="No cameras on network")
class TestCameraHardware:
    def test_real_camera_discovery(self):
        # Use CAMERA_CREDENTIALS env var
        # Discover actual cameras
        # Validate RTSP streams
        
    def test_camera_reconnection(self):
        # Test camera disconnect/reconnect
        # Verify MAC address tracking
```

## Phase 3: Fix Model Conversion Tests

### 3.1 Remove Mocking from Conversion Tests
```python
# Fix model conversion tests to use real conversions
# Even if they take 30-60 minutes
# Use pytest marks for slow tests
@pytest.mark.slow
@pytest.mark.timeout(3600)  # 1 hour timeout
def test_real_tflite_conversion():
    # Actual conversion with quantization
    # Real calibration data
    # Verify output model works
```

## Phase 4: Integration Test Suite

### 4.1 End-to-End Hardware Test
```python
# tests/test_e2e_hardware_integration.py
class TestE2EHardwareIntegration:
    def test_full_fire_detection_pipeline(self):
        # Real camera → Real AI inference → Real MQTT → Real GPIO
        # Use available hardware (Coral or TensorRT)
        # Verify complete system operation
```

### 4.2 Multi-Python Version Support
```python
# Ensure tests work with scripts/run_tests_by_python_version.sh
# Python 3.12: Most tests
# Python 3.10: YOLO-NAS training
# Python 3.8: Coral TPU tests
```

## Phase 5: Environment Configuration

### 5.1 Test Environment Setup
```bash
# .env.test
CAMERA_CREDENTIALS=username:password
MQTT_BROKER=localhost
MQTT_PORT=1883
GPIO_SIMULATION=true  # For non-Pi systems
FRIGATE_DETECTOR=auto  # Auto-detect hardware
```

### 5.2 Hardware Detection
```python
# Enhance conftest.py hardware detection
def detect_available_hardware():
    hardware = {
        'coral_tpu': has_coral_tpu(),
        'tensorrt': has_tensorrt(),
        'cameras': find_network_cameras(),
        'hailo': False  # Not available
    }
    return hardware
```

## Phase 6: Fix Specific Issues

### 6.1 Camera Credentials
- Remove all hardcoded credentials
- Always use `os.getenv('CAMERA_CREDENTIALS', '')`
- Pass credentials only through environment

### 6.2 MQTT Optimization
- Already correctly implemented ✓
- Using real broker everywhere ✓
- Proper topic isolation ✓

### 6.3 Timeout Configuration
- Use appropriate timeouts for hardware operations
- Model conversion: 60 minutes
- Camera discovery: 30 seconds
- MQTT operations: 10 seconds

## Phase 7: CI/CD Considerations

### 7.1 Hardware-Dependent Tests
```yaml
# .github/workflows/test.yml
- name: Run hardware tests
  env:
    SKIP_HARDWARE_TESTS: ${{ !contains(runner.labels, 'self-hosted') }}
  run: |
    if [ "$SKIP_HARDWARE_TESTS" = "false" ]; then
      ./scripts/run_tests_by_python_version.sh --all
    else
      ./scripts/run_tests_by_python_version.sh --no-hardware
    fi
```

### 7.2 Test Categories
```python
# pytest.ini markers
markers =
    hardware: requires physical hardware
    slow: test takes >1 minute
    coral_tpu: requires Coral TPU
    tensorrt: requires NVIDIA GPU with TensorRT
    cameras: requires network cameras
```

## Implementation Priority

1. **Critical**: Fix internal mocking (Phase 1)
2. **High**: Add hardware tests (Phase 2)
3. **High**: Fix model conversion tests (Phase 3)
4. **Medium**: Integration tests (Phase 4)
5. **Medium**: Environment setup (Phase 5)
6. **Low**: CI/CD updates (Phase 7)

## Success Criteria

1. ✅ No internal module mocking
2. ✅ All tests use real MQTT broker
3. ✅ Hardware tests verify actual capabilities
4. ✅ Tests pass with `./scripts/run_tests_by_python_version.sh`
5. ✅ Camera credentials from env vars only
6. ✅ Proper hardware detection and skipping
7. ✅ Integration tests exercise full system

## Next Steps

1. Review and approve this plan
2. Implement fixes in priority order
3. Run full test suite on hardware
4. Update CI/CD for hardware tests
5. Document hardware test requirements