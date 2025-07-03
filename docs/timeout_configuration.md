# Timeout Configuration Guide for Wildfire Watch Tests

## Overview

The Wildfire Watch test suite has been configured to handle long timeouts gracefully, particularly for infrastructure setup that can take 15+ seconds. This document explains the timeout configuration and how to use it effectively.

## Configuration Files

### 1. pytest.ini
**Location**: `/home/seth/wildfire-watch/pytest.ini`

Key timeout settings:
- `timeout = 3600` - 1 hour timeout per test (handles slow infrastructure)
- `timeout_method = thread` - More reliable timeout mechanism
- `timeout_func_only = true` - Only timeout test function, not fixture setup
- `session_timeout = 7200` - 2 hour session timeout for entire test run

### Python Version Specific Configurations

#### pytest-python312.ini (Python 3.12 - Most Tests)
**Location**: `/home/seth/wildfire-watch/pytest-python312.ini`
- **Per-test timeout**: 3600 seconds (1 hour)
- **Session timeout**: 7200 seconds (2 hours)
- **Used for**: Most unit and integration tests
- **Tests include**: GPIO trigger, fire consensus, camera detection, MQTT integration

#### pytest-python310.ini (Python 3.10 - YOLO-NAS)
**Location**: `/home/seth/wildfire-watch/pytest-python310.ini`
- **Per-test timeout**: 7200 seconds (2 hours)
- **Session timeout**: 14400 seconds (4 hours)
- **Used for**: YOLO-NAS training and super-gradients tests
- **Tests include**: Model training, quantization-aware training (QAT)

#### pytest-python38.ini (Python 3.8 - Coral TPU)
**Location**: `/home/seth/wildfire-watch/pytest-python38.ini`
- **Per-test timeout**: 7200 seconds (2 hours)
- **Session timeout**: 14400 seconds (4 hours)
- **Used for**: Model conversion and Coral TPU tests
- **Tests include**: TFLite conversion, Edge TPU compilation, INT8 quantization

### 2. conftest.py
**Location**: `/home/seth/wildfire-watch/tests/conftest.py`

Provides:
- Session-scoped MQTT broker fixture
- Timeout-aware test setup
- Performance monitoring
- Graceful timeout handling

### 3. timeout_utils.py
**Location**: `/home/seth/wildfire-watch/tests/timeout_utils.py`

Utility functions and decorators for timeout management.

## Usage Patterns

### 1. Basic Test (No Special Handling Needed)
```python
def test_basic_functionality():
    """Normal test - uses default timeout settings"""
    # Test logic here
    assert True
```

### 2. Tests with Expected Long Timeouts
```python
@pytest.mark.timeout_expected
def test_with_infrastructure():
    """Test that uses slow infrastructure like MQTT broker"""
    # Will use extended timeouts automatically
    pass

@expect_long_timeout(timeout_seconds=1800, reason="MQTT infrastructure setup")
def test_with_explicit_timeout():
    """Test with explicit timeout declaration"""
    pass
```

### 3. MQTT Infrastructure Tests
```python
@mqtt_infrastructure_test(timeout_seconds=900)
def test_mqtt_functionality(test_mqtt_broker):
    """Test that requires MQTT broker - uses optimized session broker"""
    with mqtt_setup_context():
        # MQTT operations here
        pass
```

### 4. Integration Tests
```python
@integration_test(timeout_seconds=1200)  
def test_multi_service_integration():
    """Integration test with multiple services"""
    with service_startup_context("camera_detector"):
        # Service setup
        pass
```

## Timeout Behavior

### Expected Timeouts
These operations are **expected** to take time and won't cause test failures:

1. **MQTT Broker Setup**: 15-30 seconds (session-scoped, amortized)
2. **Service Initialization**: 5-10 seconds per service
3. **Docker Operations**: 30-60 seconds (building + starting containers)
4. **Model Loading**: 10-30 seconds
5. **Hardware Detection**: 5-15 seconds
6. **Model Conversion Operations**:
   - ONNX: 2-5 minutes (includes simplification)
   - TFLite: 15-30 minutes (includes INT8 quantization with calibration data)
   - TensorRT: 30-60 minutes (engine optimization is compute-intensive)
   - Edge TPU: 10-20 minutes (compilation for TPU)
   - OpenVINO: 10-30 minutes (includes IR generation and optimization)
7. **Model Training** (YOLO-NAS):
   - Small dataset: 30-60 minutes
   - QAT (Quantization-Aware Training): Additional 20-30 minutes
8. **Integration Tests**:
   - E2E Tests: 2-5 minutes (camera discovery → fire detection → trigger)
   - Docker Integration: 2-3 minutes (build + run + test)
   - Hardware Integration: 5-10 minutes (depends on hardware)

### Timeout Thresholds

| Operation | Normal | Warning | Error |
|-----------|--------|---------|-------|
| Individual Test | <10s | 10-60s | >1hr |
| MQTT Setup | <30s | 30-60s | >5min |
| Service Startup | <15s | 15-60s | >5min |
| Integration Test | <120s | 2-10min | >1hr |
| Full Test Suite | <30min | 30-60min | >2hr |

## Running Tests with Timeout Configuration

### Automatic Python Version Selection (Recommended)
```bash
# Automatically uses correct Python version and timeout configuration
./scripts/run_tests_by_python_version.sh --all

# Run specific Python version tests
./scripts/run_tests_by_python_version.sh --python312  # Most tests
./scripts/run_tests_by_python_version.sh --python310  # YOLO-NAS/super-gradients  
./scripts/run_tests_by_python_version.sh --python38   # Coral TPU/TensorFlow Lite

# Run specific test with auto-detection
./scripts/run_tests_by_python_version.sh --test tests/test_detect.py
```

### Manual Test Run with Specific Python Version
```bash
# Python 3.12 (most tests) - uses pytest-python312.ini
python3.12 -m pytest -c pytest-python312.ini

# Python 3.10 (YOLO-NAS) - uses pytest-python310.ini with 2hr timeout
python3.10 -m pytest -c pytest-python310.ini

# Python 3.8 (Coral TPU) - uses pytest-python38.ini with 2hr timeout
python3.8 -m pytest -c pytest-python38.ini
```

### Quick Tests Only (Skip Slow Infrastructure)
```bash
# Skip tests marked as slow or requiring infrastructure
python3.12 -m pytest tests/ -v -m "not slow and not infrastructure_dependent"
```

### Test Specific Categories
```bash
# Run only MQTT tests with appropriate timeouts
python3.12 -m pytest tests/ -v -m "mqtt"

# Run integration tests (will use long timeouts)
python3.12 -m pytest tests/ -v -m "integration"
```

### Override Timeout for Debugging
```bash
# Disable timeouts entirely for debugging
python3.12 -m pytest tests/ -v --timeout=0

# Custom timeout
python3.12 -m pytest tests/ -v --timeout=7200  # 2 hours
```

## Performance Monitoring

The timeout configuration includes automatic performance monitoring:

1. **Test Duration Logging**: Tests >30s are logged as slow
2. **Infrastructure Timing**: Setup times are measured and reported
3. **Timeout Patterns**: Repeated slow operations are flagged
4. **Session Summary**: Overall timing summary at end of session

### View Performance Reports
```bash
# Run with verbose logging to see timing information
python3.12 -m pytest tests/ -v --log-level=INFO

# View detailed fixture setup times
python3.12 -m pytest tests/ -v --setup-show
```

## Troubleshooting

### Problem: Tests Still Timing Out
**Solution**: Check if test is properly marked

```python
# Add timeout marker
@pytest.mark.timeout_expected
def test_slow_operation():
    pass

# Or use decorator
@expect_long_timeout(timeout_seconds=1800)
def test_very_slow_operation():
    pass
```

### Problem: Session Takes Too Long
**Solution**: Use session-scoped fixtures and skip slow tests

```bash
# Run without slow tests
python3.12 -m pytest tests/ -m "not slow"

# Use optimized fixtures
python3.12 -m pytest tests/test_detect.py::test_initialization -v
```

### Problem: MQTT Broker Setup Slow
**Solution**: Already optimized with session-scoped broker

The `conftest.py` provides a session-scoped MQTT broker that starts once and is reused across all tests, reducing setup overhead from 15s per test to 15s total.

### Problem: Need Even Longer Timeouts
**Solution**: Modify pytest.ini or use environment variables

```bash
# Set custom timeout via environment
PYTEST_TIMEOUT=7200 python3.12 -m pytest tests/ -v

# Or modify pytest.ini timeout value
# timeout = 7200  # 2 hours
```

## Best Practices

### 1. Mark Slow Tests Appropriately
```python
@pytest.mark.slow
@pytest.mark.timeout_expected
def test_long_operation():
    """Test that inherently takes a long time"""
    pass
```

### 2. Use Context Managers for Timing
```python
def test_with_timing():
    with timeout_context("Operation name", expected_duration=30):
        # Long operation here
        pass
```

### 3. Prefer Session-Scoped Fixtures
```python
@pytest.fixture(scope="session")
def expensive_setup():
    """Share expensive setup across tests"""
    # Setup once, use many times
    yield setup_object
```

### 4. Skip Tests in CI When Appropriate
```python
@pytest.mark.skipif(os.getenv("CI"), reason="Too slow for CI")
@pytest.mark.timeout_expected
def test_very_slow_integration():
    pass
```

## Summary

The timeout configuration ensures that:

1. ✅ **Tests don't fail due to expected infrastructure delays**
2. ✅ **Long-running operations are handled gracefully**  
3. ✅ **Performance is monitored and reported**
4. ✅ **Session-scoped fixtures reduce overall test time**
5. ✅ **Developers can debug without timeout interference**

The configuration strikes a balance between catching truly hanging tests while allowing normal infrastructure setup time that can be 15+ seconds in the Wildfire Watch system.