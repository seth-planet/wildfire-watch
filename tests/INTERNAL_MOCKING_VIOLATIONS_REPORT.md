# Internal Module Mocking Violations Report

## Overview
This report identifies test files that violate the integration testing philosophy by mocking internal wildfire-watch modules instead of using real implementations.

According to CLAUDE.md:
> **Never mock internal modules** - Don't mock wildfire-watch modules like `consensus`, `trigger`, `detect`, etc.
> **Only mock external dependencies**: `RPi.GPIO`, `docker`, `requests`, File I/O, network calls, hardware interfaces, `time.sleep`
> **DO NOT Mock: `paho.mqtt.client`** - Use real MQTT broker for testing

## Violations Found

### 1. **test_new_features.py**
- **Line 24**: `with patch('consensus.mqtt.Client'):`
- **Line 68**: `with patch('consensus.mqtt.Client'):`
- **Line 105**: `with patch('consensus.mqtt.Client'):`
- **Line 147**: `with patch('consensus.mqtt.Client'):`
- **Line 186**: `with patch('consensus.mqtt.Client'):`
- **Issue**: Mocking MQTT client instead of using real MQTT broker
- **Fix**: Use `test_mqtt_broker` fixture to provide actual MQTT server

### 2. **test_core_logic.py**
- **Line 41**: `with patch('consensus.mqtt.Client'):`
- **Line 64**: `with patch('consensus.mqtt.Client'):`
- **Issue**: Mocking MQTT client instead of using real MQTT broker
- **Fix**: Use real MQTT broker for proper integration testing

### 3. **test_simplified_integration.py**
- **Line 149**: `with patch('consensus.threading.Timer'):`
- **Line 321**: `with patch('consensus.threading.Timer'):`
- **Line 252**: `with patch('trigger.threading.Timer'):`
- **Line 441**: `with patch('trigger.threading.Timer'):`
- **Issue**: Mocking threading.Timer within internal modules
- **Fix**: These are acceptable as they mock Python standard library, not internal modules

### 4. **test_camera_detector.py**
- **Line 59**: `with patch.object(detect.CameraDetector, '_start_background_tasks'):`
- **Line 399**: `with patch.object(detect.CameraDetector, '_start_background_tasks'):`
- **Line 189**: `@patch('detect.srp')`
- **Issue**: 
  - `_start_background_tasks` patch is acceptable (prevents background tasks during testing)
  - `detect.srp` is external (scapy library) - acceptable
- **Status**: Mostly compliant

### 5. **test_new_features.py** (Additional Issues)
- **Line 224**: `@patch('trigger.GPIO', create=True)`
- **Line 253**: `@patch('trigger.GPIO', create=True)`
- **Status**: These are acceptable - mocking external hardware interface

## Summary of Violations

### Critical Violations (Must Fix):
1. **MQTT Client Mocking** - 7 instances across 3 files
   - test_new_features.py (5 instances)
   - test_core_logic.py (2 instances)
   - These tests should use real MQTT broker via `test_mqtt_broker` fixture

### Acceptable Mocking (No Action Needed):
1. **Hardware Interfaces** - GPIO mocking is acceptable
2. **External Libraries** - scapy (srp) mocking is acceptable
3. **Background Tasks** - Preventing background tasks during testing is acceptable
4. **Standard Library** - threading.Timer mocking is acceptable

## Recommended Actions

1. **Immediate Priority**: Replace all `patch('consensus.mqtt.Client')` with real MQTT broker usage
2. **Update Tests**: Modify test_new_features.py and test_core_logic.py to use `test_mqtt_broker` fixture
3. **Guidelines**: Ensure all new tests follow the integration testing philosophy

## Example Fix

Instead of:
```python
with patch('consensus.mqtt.Client'):
    from consensus import FireConsensus
    consensus = FireConsensus.__new__(FireConsensus)
```

Use:
```python
def test_feature(test_mqtt_broker):
    from consensus import FireConsensus
    # Set environment variables for MQTT connection
    os.environ['MQTT_BROKER'] = test_mqtt_broker.host
    os.environ['MQTT_PORT'] = str(test_mqtt_broker.port)
    
    # Create real instance with real MQTT
    consensus = FireConsensus()
    # Test real functionality...
```