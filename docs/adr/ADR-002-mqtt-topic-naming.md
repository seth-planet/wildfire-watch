# ADR-002: MQTT Topic Naming Convention

## Status
Accepted

## Context
The Wildfire Watch system uses MQTT as its primary communication protocol. Topic naming was inconsistent:
- `system/trigger_telemetry` (non-standard)
- `frigate/+/fire` (external system)
- `cameras/discovered` (resource-based)
- Various ad-hoc patterns

This inconsistency made it difficult to:
- Apply security policies
- Route messages efficiently  
- Monitor system health
- Understand message flow

## Decision
Adopt a hierarchical topic naming convention:

```
{prefix}/{domain}/{resource}/{action}
```

### Topic Structure
- **prefix**: Optional test/environment prefix (e.g., `test/worker-1`)
- **domain**: System domain (`system`, `fire`, `cameras`, `gpio`)
- **resource**: Specific resource or service name
- **action**: Operation or message type

### Standard Patterns

#### System Health
```
system/{service_name}/health
system/{service_name}/lwt
system/{service_name}/metrics
```

#### Fire Detection
```
fire/detection/{camera_id}
fire/consensus/result
fire/trigger
fire/emergency
```

#### Camera Management
```
cameras/discovered
cameras/{camera_id}/status
cameras/{camera_id}/config
```

#### GPIO Control
```
gpio/pump/command
gpio/pump/status
gpio/valve/{valve_id}/state
```

## Consequences

### Positive
- **Clarity**: Clear hierarchy shows relationships
- **Security**: Easy to apply ACLs by domain
- **Filtering**: Efficient subscription patterns
- **Monitoring**: Consistent health topic location
- **Extensibility**: New domains easily added

### Negative  
- **Migration Required**: Existing topics must change
- **Client Updates**: All MQTT clients need updating
- **Documentation**: Requires comprehensive docs

### Neutral
- **Verbosity**: Longer topic names
- **Learning Curve**: Team must learn convention

## Implementation

### Migration Strategy
1. **Phase 1**: Add new topics alongside legacy (dual publish)
2. **Phase 2**: Update all consumers to use new topics
3. **Phase 3**: Deprecation warnings on legacy topics
4. **Phase 4**: Remove legacy topic support (30 days)

### Topic Prefix for Testing
```python
prefix = os.getenv('MQTT_TOPIC_PREFIX', '')
topic = f"{prefix}/system/camera_detector/health" if prefix else "system/camera_detector/health"
```

### Wildcard Subscriptions
- `system/+/health` - All service health
- `fire/#` - All fire-related messages
- `+/+/status` - All status messages

## Alternatives Considered

### 1. Flat Topics
Keep simple topic names like `camera-health`, `fire-detected`
- ✅ Simple
- ❌ No hierarchy
- ❌ Hard to filter
- ❌ Namespace collisions

### 2. Service-Centric
Organize by service: `camera-detector/health`, `gpio-trigger/status`
- ✅ Service ownership clear
- ❌ Cross-service topics unclear
- ❌ Duplicated concepts

### 3. Event-Sourcing Style
Use event types: `events/fire-detected`, `commands/activate-pump`
- ✅ Clear intent
- ❌ Not intuitive for IoT
- ❌ Verbose for status updates

## Examples

### Before
```
system/trigger_telemetry
frigate/camera1/fire  
mqtt/status/camera_detector
pump_activated
```

### After
```
system/gpio_trigger/health
fire/detection/camera1
system/camera_detector/health
gpio/pump/status
```

## Security Considerations

### ACL Configuration
```conf
# Service-specific write access
user gpio_trigger
topic write system/gpio_trigger/health
topic write gpio/pump/status
topic read fire/trigger

# Read-only monitoring
user monitoring
topic read system/+/health
topic read +/+/status
```

## References
- [MQTT Topic Best Practices](https://www.hivemq.com/blog/mqtt-essentials-part-5-mqtt-topics-best-practices/)
- [AWS IoT Topic Design](https://docs.aws.amazon.com/iot/latest/developerguide/topics.html)
- [Eclipse Mosquitto ACL](https://mosquitto.org/documentation/authentication-methods/)

---

**Date**: 2025-07-04  
**Deciders**: Development Team  
**Related**: ADR-001-service-refactoring