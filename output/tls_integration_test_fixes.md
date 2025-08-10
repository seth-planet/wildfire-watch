# TLS Integration Test Fixes

## Summary
Fixed failing TLS integration tests in `tests/test_tls_integration_consolidated.py`. All tests now pass or are correctly skipped when TLS infrastructure is not available.

## Issues Fixed

### 1. test_services_read_tls_config (FAILED → PASSED)
**Problem**: The test assumed service modules have a `Config` class, but they use different class names:
- camera_detector has `CameraDetectorConfig`
- fire_consensus has `FireConsensusConfig`
- gpio_trigger uses a global `CONFIG` dictionary

**Fix**: Updated the test to use correct class names:
```python
services = [
    ("camera_detector", "detect.CameraDetectorConfig"),
    ("fire_consensus", "consensus.FireConsensusConfig"),
    ("gpio_trigger", "trigger.CONFIG"),
]
```

Also fixed the assertion to use `config.mqtt_tls` (lowercase) for ConfigBase-derived classes.

### 2. test_services_mount_certificates (FAILED → PASSED)
**Problem**: The test incorrectly accessed the YAML dictionary using dot notation.

**Fix**: Changed from:
```python
if service_name in compose_config.services:
    service = compose_config.services[service_name]
```

To:
```python
if service_name in compose_config['services']:
    service = compose_config['services'][service_name]
```

### 3. TestMQTTBrokerTLS tests (SKIPPED - Correct Behavior)
These tests are correctly skipped when MQTT TLS port (8883) is not available. This is expected behavior in test environments without a running MQTT broker with TLS enabled.

## Test Results
- **Total tests**: 15
- **Passed**: 10
- **Skipped**: 5 (require running MQTT broker with TLS)
- **Failed**: 0

## Running the Tests
```bash
# Run all TLS integration tests
python3.12 -m pytest tests/test_tls_integration_consolidated.py -v

# Run specific test
python3.12 -m pytest tests/test_tls_integration_consolidated.py::TestServiceTLS::test_services_read_tls_config -xvs
```

## Notes
- The skipped tests require a running MQTT broker with TLS enabled on port 8883
- Tests properly handle environment variable modification and restore original values
- Test isolation is maintained for parallel execution environments
- Minor warnings about deprecated datetime properties in cryptography library can be ignored