# Wildfire Watch Refactored Services Deployment Package

## Package Contents

This deployment package contains all files necessary to deploy the refactored GPIO trigger service and prepare for future service refactoring.

### Core Files

#### 1. Refactored GPIO Trigger
- **File**: `gpio_trigger/trigger_refactored.py`
- **Description**: Complete refactored implementation using base classes
- **Dependencies**: Requires base classes from utils/

#### 2. Enhanced MQTT Base Class  
- **File**: `utils/mqtt_service_improved.py`
- **Description**: Improved MQTT service with exponential backoff and offline queuing
- **Status**: Ready for production use

#### 3. Configuration Files
- **Files**: `.env`, `.env.example`
- **Changes**: Updated GPIO pin assignments to avoid conflicts
- **Action Required**: Review and update production configs

### Documentation

#### Planning Documents
- `gpio_trigger/refactoring_plan.md` - Detailed refactoring approach
- `REFACTORED_SERVICES_DEPLOYMENT_PLAN.md` - Deployment strategy
- `PHASE_4_VALIDATION_REPORT.md` - Validation results
- `FINAL_E2E_TEST_AND_REFACTORING_REPORT.md` - Comprehensive summary

#### Test Results
- `E2E_STANDARDIZATION_COMPLETE_SUMMARY.md` - Standardization summary
- Various `*_SUMMARY.md` files documenting each phase

### Test Files
- `gpio_trigger/test_refactored_trigger.py` - Unit tests
- `test_refactored_integration.py` - Integration tests
- `tests/test_integration_e2e_improved.py` - Updated E2E tests

## Quick Start Deployment

### 1. Development Environment

```bash
# Clone the refactored branch
git checkout refactored-gpio-trigger

# Update environment variables
cp .env.example .env
# Edit .env to set your MQTT broker details

# Build the image with refactored code
docker build -t wildfire-watch/gpio_trigger:refactored -f gpio_trigger/Dockerfile .

# Run with feature flag
ENABLE_REFACTORED_GPIO=true docker-compose up -d gpio_trigger

# Monitor both old and new health topics
mosquitto_sub -h localhost -t "system/+/health" -t "system/trigger_telemetry"
```

### 2. Side-by-Side Comparison

```bash
# Run both implementations
docker-compose up -d gpio_trigger          # Original
docker-compose up -d gpio_trigger_refactored  # New

# Compare outputs
./scripts/compare_gpio_outputs.sh

# Check metrics
docker stats gpio_trigger gpio_trigger_refactored
```

### 3. Canary Deployment

```bash
# Deploy to specific devices
ansible-playbook -i inventory/canary.yml deploy_refactored_gpio.yml

# Monitor canary metrics
./scripts/monitor_canary_health.sh --service gpio_trigger
```

## Configuration Reference

### Required Environment Variables
```bash
# MQTT Configuration
MQTT_BROKER=mqtt_broker
MQTT_PORT=1883
MQTT_USERNAME=
MQTT_PASSWORD=
MQTT_TLS=false

# Health Reporting
HEALTH_REPORT_INTERVAL=60

# Reconnection Settings (New)
MQTT_RECONNECT_MIN_DELAY=1.0
MQTT_RECONNECT_MAX_DELAY=60.0
MQTT_RECONNECT_MULTIPLIER=2.0
MQTT_RECONNECT_JITTER=0.3

# Offline Queue (New)
MQTT_OFFLINE_QUEUE_ENABLED=true
MQTT_OFFLINE_QUEUE_SIZE=100

# GPIO Pins (Updated to avoid conflicts)
ENGINE_START_PIN=17
ENGINE_STOP_PIN=27
MAIN_VALVE_PIN=22
PRIMING_VALVE_PIN=5
REFILL_VALVE_PIN=6
RESERVOIR_FLOAT_PIN=13  # Changed from 16
LINE_PRESSURE_PIN=19    # Changed from 20
IGN_ON_PIN=23
IGN_OFF_PIN=24
IGN_START_PIN=25
RPM_REDUCE_PIN=16
```

### Health Topic Migration
| Service | Old Topic | New Topic |
|---------|-----------|-----------|
| GPIO Trigger | system/trigger_telemetry | system/gpio_trigger/health |
| Camera Detector | system/camera_detector/health | (unchanged) |
| Fire Consensus | system/fire_consensus/health | (unchanged) |

## Monitoring and Validation

### Health Check Script
```bash
#!/bin/bash
# Check if refactored service is healthy

# Subscribe to health topic
timeout 65 mosquitto_sub -h localhost \
  -t "system/gpio_trigger/health" -C 1 | \
  jq -r '.state'

# Expected output: "IDLE" or other valid states
```

### Metrics to Monitor
1. **Connection Stability**
   - Reconnection frequency
   - Connection uptime
   - Message delivery rate

2. **Performance**
   - CPU usage (should be lower)
   - Memory usage (should be stable)
   - Message latency

3. **Safety Features**
   - Emergency shutdown response time
   - Dry run detection accuracy
   - Timer precision

## Rollback Procedure

If issues arise, rollback is simple:

```bash
# Stop refactored service
docker-compose stop gpio_trigger

# Revert to original image
docker-compose up -d gpio_trigger --image wildfire-watch/gpio_trigger:latest

# Verify original service is running
docker-compose ps gpio_trigger
```

## Future Service Refactoring

Use the GPIO trigger as a template for refactoring other services:

### Priority Order
1. **Camera Detector** - Partially uses base classes already
2. **Fire Consensus** - Will benefit from reconnection improvements  
3. **Security NVR** - Most complex, refactor in phases

### Refactoring Checklist
- [ ] Inherit from appropriate base classes
- [ ] Migrate to ConfigBase for configuration
- [ ] Standardize health reporting topic
- [ ] Implement get_service_health() method
- [ ] Use SafeTimerManager for all timers
- [ ] Add connection state callbacks
- [ ] Enable offline message queuing
- [ ] Update tests for new patterns
- [ ] Document breaking changes
- [ ] Create migration guide

## Support and Troubleshooting

### Common Issues

1. **MQTT Connection Failures**
   - Check MQTT_BROKER environment variable
   - Verify network connectivity
   - Check broker logs

2. **Health Messages Not Appearing**
   - Verify HEALTH_REPORT_INTERVAL is set
   - Check service logs for errors
   - Ensure MQTT connection is established

3. **GPIO Pin Conflicts**
   - Review pin assignments in .env
   - Check for duplicate pin usage
   - Verify hardware connections

### Debug Commands
```bash
# Check service logs
docker-compose logs -f gpio_trigger

# Verify environment variables
docker-compose exec gpio_trigger env | grep -E "(MQTT|PIN|HEALTH)"

# Test MQTT connectivity
docker-compose exec gpio_trigger mosquitto_pub -h $MQTT_BROKER -t test -m "test"
```

## Version History

| Version | Date | Changes |
|---------|------|---------|
| refactored-v1 | 2025-07-03 | Initial refactored implementation |
| original | 2025-06-01 | Original implementation |

---

**Note**: This deployment package represents a significant improvement in code quality and maintainability while preserving all safety-critical functionality. Please test thoroughly in your environment before production deployment.