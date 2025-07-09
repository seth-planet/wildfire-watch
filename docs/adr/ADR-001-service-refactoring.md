# ADR-001: Service Refactoring with Base Classes

## Status
Accepted

## Context
The Wildfire Watch system had grown organically with each service implementing its own MQTT connection handling, health reporting, and configuration management. This led to:
- Code duplication across services (~40-45% duplicate code)
- Inconsistent health reporting formats
- Varying reconnection strategies
- Difficult maintenance and debugging

## Decision
We decided to refactor all services to use a common set of base classes:
- `MQTTService` - Standardized MQTT connection management
- `HealthReporter` - Consistent health topic publishing
- `ThreadSafeService` - Thread management and shutdown coordination
- `ConfigBase` - Configuration validation and loading

## Consequences

### Positive
- **Code Reduction**: 40-45% less code to maintain
- **Consistency**: All services behave predictably
- **Reliability**: Improved error handling and reconnection logic
- **Monitoring**: Standardized health topics enable unified dashboards
- **Development Speed**: New services can extend base classes

### Negative
- **Migration Effort**: Requires updating all services
- **Training**: Team needs to learn new patterns
- **Compatibility**: Must maintain legacy topics during transition

### Neutral
- **Complexity**: Base classes add abstraction layer
- **Testing**: Need tests for both base classes and implementations

## Implementation Details

### Base Class Architecture
```python
class ServiceImplementation(MQTTService, HealthReporter, ThreadSafeService):
    def __init__(self):
        # Initialize all base classes
        MQTTService.__init__(self, "service_name", config)
        HealthReporter.__init__(self, health_interval)
        ThreadSafeService.__init__(self, logger)
```

### Health Topic Standard
- Old: Various formats (`system/trigger_telemetry`, custom topics)
- New: `system/{service_name}/health`
- Transition: Dual publishing for 30 days

### Reconnection Strategy
- Exponential backoff: 1s, 2s, 4s, 8s... up to 60s
- Jitter: ±30% to prevent thundering herd
- Offline queue: 100 messages max

## Alternatives Considered

### 1. Minimal Refactoring
Only fix immediate issues without base classes
- ✅ Less work
- ❌ Doesn't solve root cause
- ❌ Technical debt remains

### 2. Complete Rewrite
Start fresh with new architecture
- ✅ Clean slate
- ❌ High risk
- ❌ Long timeline
- ❌ No incremental value

### 3. Service Mesh
Use Istio/Linkerd for communication
- ✅ Industry standard
- ❌ Overhead for embedded systems
- ❌ Complex for small team

## Lessons Learned

1. **Gradual Migration Works**: Service-by-service approach reduced risk
2. **Backward Compatibility Critical**: Dual topic support prevented outages
3. **Base Classes Simplify**: 75% less code in refactored services
4. **Discovery Surprises**: Found 2 services already refactored

## References
- [Base Class Design Pattern](https://refactoring.guru/design-patterns/template-method)
- [MQTT Best Practices](http://www.steves-internet-guide.com/mqtt-clean-sessions-example/)
- [Exponential Backoff](https://cloud.google.com/iot/docs/how-tos/exponential-backoff)

---

**Date**: 2025-07-04  
**Deciders**: Development Team  
**Related**: ADR-002-mqtt-topic-naming