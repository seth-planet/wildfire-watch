# GPIO Configuration Migration Guide

## Overview

This document describes the migration from the legacy global `CONFIG` dictionary to the modern `PumpControllerConfig` class in the GPIO trigger service. This migration was completed to improve code maintainability, enable proper test isolation, and align with the configuration patterns used by other services.

## Migration Summary

### What Changed

1. **Removed Legacy System**:
   - Deleted `LazyConfigDict` class from `gpio_trigger/trigger.py`
   - Removed global `CONFIG` instance
   - Eliminated dual configuration system

2. **Updated PumpController**:
   - Now uses only `self.config` (PumpControllerConfig instance)
   - No longer creates `self.cfg` dictionary
   - All configuration access is type-safe and validated

3. **Test Infrastructure**:
   - Removed `update_gpio_config_for_tests()` function
   - Removed `configure_pump_controller_for_test()` function
   - Added new `pump_controller_factory` fixture

## Benefits

1. **Type Safety**: Configuration values are validated at instantiation
2. **Test Isolation**: Each test gets its own configuration instance
3. **Consistency**: Aligns with FireConsensus and CameraDetector patterns
4. **Maintainability**: Single source of truth for configuration

## Testing Patterns

### Old Pattern (Deprecated)
```python
def test_something(test_mqtt_broker, mqtt_topic_factory, monkeypatch):
    from tests.test_utils.helpers import update_gpio_config_for_tests
    from gpio_trigger.trigger import CONFIG
    
    # Lots of environment setup
    monkeypatch.setenv('MQTT_BROKER', broker_host)
    monkeypatch.setenv('PRIMING_DURATION', '0.2')
    # ... many more ...
    
    # Update global CONFIG
    update_gpio_config_for_tests(test_env, conn_params, topic_prefix)
    
    # Create controller
    controller = PumpController(auto_connect=False)
    controller.connect()
```

### New Pattern (Current)
```python
def test_something(test_mqtt_broker, mqtt_topic_factory, pump_controller_factory):
    # Get connection params
    conn_params = test_mqtt_broker.get_connection_params()
    full_topic = mqtt_topic_factory("dummy")
    topic_prefix = full_topic.rsplit('/', 1)[0]
    
    # Create controller with configuration
    controller = pump_controller_factory(
        mqtt_broker=conn_params['host'],
        mqtt_port=conn_params['port'],
        topic_prefix=topic_prefix,
        priming_duration=0.2,
        # any other config overrides
    )
    controller.connect()
```

## Production Usage

For production code, create a PumpController with explicit configuration:

```python
from gpio_trigger.trigger import PumpController, PumpControllerConfig

# Configuration loads from environment
config = PumpControllerConfig()

# Or override specific values
config = PumpControllerConfig()
config.priming_duration = 20.0
config.max_engine_runtime = 3600.0

# Create controller
controller = PumpController(config=config)
```

## Environment Variables

All environment variables are still supported and read by PumpControllerConfig:

- `MQTT_BROKER` - MQTT broker hostname
- `MQTT_PORT` - MQTT broker port
- `MQTT_TLS` - Enable TLS (true/false)
- `TOPIC_PREFIX` - Topic namespace prefix
- `PRIMING_DURATION` - Pump priming time in seconds
- `MAX_ENGINE_RUNTIME` - Maximum runtime before safety shutdown
- (and all other GPIO-related variables)

## Migration Checklist

If you have custom code using the GPIO trigger:

1. ✅ Remove any imports of `CONFIG` from `gpio_trigger.trigger`
2. ✅ Replace `CONFIG['KEY']` with `controller.config.key`
3. ✅ Update tests to use `pump_controller_factory`
4. ✅ Remove calls to `update_gpio_config_for_tests()`
5. ✅ Use PumpControllerConfig for any custom configuration needs

## Safety Note

The GPIO trigger service intentionally does NOT inherit from `MQTTService` base class for safety-critical reasons. This design decision ensures:
- Minimal dependencies for hardware control
- Direct MQTT connection without abstraction layers
- Lower latency for emergency response
- Simpler code path for critical operations

This architectural exception is documented in CLAUDE.md and should be preserved.