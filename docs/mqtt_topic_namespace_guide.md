# MQTT Topic Namespace Guide

## Overview

The Wildfire Watch system supports MQTT topic namespacing through the `MQTT_TOPIC_PREFIX` environment variable. This feature enables:

- **Parallel test execution** without topic conflicts
- **Multi-site deployments** on shared MQTT infrastructure
- **Development/staging/production** isolation
- **Multi-tenant** configurations

## Configuration

### Environment Variable

```bash
# Production (single-site, default)
MQTT_TOPIC_PREFIX=

# Multi-site deployment
MQTT_TOPIC_PREFIX=site1
MQTT_TOPIC_PREFIX=site2

# Testing (parallel workers)
MQTT_TOPIC_PREFIX=test/worker1
MQTT_TOPIC_PREFIX=test/worker2

# Development environments
MQTT_TOPIC_PREFIX=dev/john
MQTT_TOPIC_PREFIX=staging
```

### Docker Deployment

```yaml
# docker-compose.yml
services:
  camera_detector:
    environment:
      MQTT_TOPIC_PREFIX: ${MQTT_TOPIC_PREFIX}
  
  fire_consensus:
    environment:
      MQTT_TOPIC_PREFIX: ${MQTT_TOPIC_PREFIX}
  
  gpio_trigger:
    environment:
      MQTT_TOPIC_PREFIX: ${MQTT_TOPIC_PREFIX}
```

## Topic Structure

### Without Namespace (Production Default)
```
cameras/discovered
fire/detection
fire/trigger
system/consensus_telemetry
system/camera_detector_health
system/trigger_telemetry
```

### With Namespace
```
site1/cameras/discovered
site1/fire/detection
site1/fire/trigger
site1/system/consensus_telemetry
site1/system/camera_detector_health
site1/system/trigger_telemetry
```

## Use Cases

### 1. Production Single-Site (Default)
```bash
# .env
MQTT_TOPIC_PREFIX=
```
- No namespace prefix
- Standard topic names
- Simplest configuration

### 2. Multi-Site Deployment
```bash
# Site 1
MQTT_TOPIC_PREFIX=warehouse_a

# Site 2
MQTT_TOPIC_PREFIX=warehouse_b
```
- Each site operates independently
- Shared MQTT broker infrastructure
- Central monitoring possible

### 3. Parallel Testing
```bash
# pytest-xdist workers
MQTT_TOPIC_PREFIX=test/worker0
MQTT_TOPIC_PREFIX=test/worker1
MQTT_TOPIC_PREFIX=test/worker2
```
- Enables concurrent test execution
- No topic conflicts between workers
- Automatic cleanup per namespace

### 4. Development Isolation
```bash
# Developer environments
MQTT_TOPIC_PREFIX=dev/alice
MQTT_TOPIC_PREFIX=dev/bob

# Staging
MQTT_TOPIC_PREFIX=staging

# Production
MQTT_TOPIC_PREFIX=
```

## Service Implementation

All services automatically handle namespacing:

### Camera Detector
```python
# Publishes to:
{prefix}/cameras/discovered
{prefix}/system/camera_detector_health
```

### Fire Consensus
```python
# Subscribes to:
{prefix}/fire/detection
{prefix}/frigate/events
{prefix}/system/camera_telemetry

# Publishes to:
{prefix}/fire/trigger
{prefix}/system/consensus_telemetry
```

### GPIO Trigger
```python
# Subscribes to:
{prefix}/fire/trigger
{prefix}/fire/emergency

# Publishes to:
{prefix}/system/trigger_telemetry
```

## Best Practices

1. **Production**: Leave `MQTT_TOPIC_PREFIX` empty for single-site deployments
2. **Multi-site**: Use descriptive site names (e.g., `building_a`, `zone_1`)
3. **Testing**: Use hierarchical namespaces (e.g., `test/integration/worker1`)
4. **Development**: Include developer name (e.g., `dev/john`)
5. **Monitoring**: Subscribe to `+/system/#` to monitor all sites

## Migration Guide

### From Non-Namespaced to Namespaced

1. **Gradual Migration**:
   ```bash
   # Phase 1: Add namespace to new services
   MQTT_TOPIC_PREFIX=site1
   
   # Phase 2: Update monitoring/dashboards
   # Subscribe to both old and new topics
   
   # Phase 3: Migrate remaining services
   ```

2. **Bridge Old Topics** (temporary):
   ```python
   # MQTT bridge configuration
   topic cameras/# out 0 "" site1/cameras/
   topic fire/# out 0 "" site1/fire/
   ```

### Testing Namespace Support

```bash
# Test with namespace
MQTT_TOPIC_PREFIX=test/validation docker-compose up

# Monitor topics
mosquitto_sub -h localhost -t "test/validation/#" -v
```

## Troubleshooting

### Common Issues

1. **Services not communicating**
   - Ensure all services have same `MQTT_TOPIC_PREFIX`
   - Check Docker environment propagation
   - Verify with `mosquitto_sub -t '#' -v`

2. **Partial namespace adoption**
   - Some services namespaced, others not
   - Solution: Update all services simultaneously

3. **Monitoring breaks after namespacing**
   - Update dashboard subscriptions to `+/topic` pattern
   - Or use explicit namespace in subscriptions

### Debug Commands

```bash
# List all topics
mosquitto_sub -h localhost -t '#' -v | grep -E "^[^/]+/"

# Monitor specific namespace
mosquitto_sub -h localhost -t 'site1/#' -v

# Check service health across namespaces
mosquitto_sub -h localhost -t '+/system/+_telemetry' -v
```