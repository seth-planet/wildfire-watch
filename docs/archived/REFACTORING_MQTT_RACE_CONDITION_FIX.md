# MQTT Race Condition Fix

## Issue Identified
The services inheriting from MQTTService were experiencing a race condition during initialization. The MQTT client's background thread was started (via `loop_start()`) before the service object was fully constructed, allowing callbacks to execute on a partially-initialized object.

## Root Cause
In `MQTTService.setup_mqtt()`, the method calls `_connect_with_retry()` which starts the MQTT network loop immediately. This creates a race condition where MQTT callbacks can fire while the subclass is still in its `__init__` method.

## Fix Applied
1. **Removed automatic connection** from `setup_mqtt()` in `mqtt_service.py`
2. **Added `connect()` method** to MQTTService for explicit connection control
3. **Updated services** to call `connect()` at the end of initialization

## Services That Need Updating
- [x] FireConsensus (fire_consensus/consensus.py) - DONE
- [ ] CameraDetector (camera_detector/detect.py) - TODO
- [ ] CameraTelemetry (cam_telemetry/telemetry.py) - TODO

## Code Changes Required

### For each service inheriting from MQTTService:

Add at the end of `__init__` method:
```python
# Connect to MQTT after everything is initialized
# This prevents race conditions during startup
self.connect()
self.logger.info(f"[ServiceName] fully initialized and connected")
```

## Testing
After updating all services, run the E2E tests to ensure the race condition is resolved:
```bash
CAMERA_CREDENTIALS=admin:S3thrule python3.12 -m pytest tests/test_integration_e2e_improved.py -xvs
```