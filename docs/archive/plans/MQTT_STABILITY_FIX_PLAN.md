# MQTT Stability Fix Plan for Camera Detector

## Problem Analysis

The camera_detector service experiences frequent MQTT disconnections with the following root causes:

1. **Clean Session = False**: Causes message buildup on the broker, leading to memory issues and disconnections
2. **MQTT Loop in Main Thread**: Network I/O operations block the MQTT client's ability to send keepalive pings
3. **Thread Safety Issues**: Multiple threads publishing to MQTT without proper coordination
4. **Long-Running Operations**: Discovery operations (WS-Discovery, port scanning, RTSP validation) can take 30+ seconds, exceeding MQTT keepalive timeout

## Solution Overview

Implement a dedicated MQTT handler that:
- Runs MQTT client loop in a separate thread
- Uses a thread-safe queue for all publish operations
- Changes to `clean_session=True` to prevent message buildup
- Reduces keepalive interval for faster disconnection detection
- Provides non-blocking publish operations

## Implementation Plan

### Phase 1: Create Stable MQTT Handler
- [x] Create `mqtt_stability_fix.py` with `StableMQTTHandler` class
- [x] Implement dedicated threads for MQTT loop and publishing
- [x] Use message queue for thread-safe publishing
- [x] Implement proper reconnection logic with exponential backoff

### Phase 2: Integrate with Camera Detector
- [ ] Modify `detect.py` to use the new MQTT handler
- [ ] Replace all `mqtt_client.publish()` calls with `mqtt_publish()`
- [ ] Update connection/disconnection callbacks
- [ ] Ensure proper cleanup on shutdown

### Phase 3: Additional Improvements
- [ ] Add connection state monitoring
- [ ] Implement message priority for critical messages
- [ ] Add metrics for queue depth and publish success rate
- [ ] Consider implementing message batching for efficiency

## Specific Code Changes

### 1. Update MQTT Setup in detect.py

Replace the `_setup_mqtt` method:

```python
def _setup_mqtt(self):
    """Setup stable MQTT handler"""
    from mqtt_stability_fix import StableMQTTHandler
    
    # Create handler with reduced keepalive
    self.mqtt_handler = StableMQTTHandler(
        broker=self.config.MQTT_BROKER,
        port=self.config.MQTT_PORT,
        client_id=self.config.SERVICE_ID,
        keepalive=30,  # Reduced from 60
        tls_enabled=self.config.MQTT_TLS,
        ca_cert_path=self.config.TLS_CA_PATH if self.config.MQTT_TLS else None
    )
    
    # Set callbacks
    self.mqtt_handler.on_connect_callback = lambda: self._on_mqtt_connect(None, None, None, 0)
    self.mqtt_handler.on_disconnect_callback = lambda: self._on_mqtt_disconnect(None, None, 1)
    
    # Set LWT
    lwt_topic = f"{self.config.TOPIC_HEALTH}/{self.config.NODE_ID}/lwt"
    lwt_payload = json.dumps({
        'node_id': self.config.NODE_ID,
        'service': 'camera_detector',
        'status': 'offline',
        'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    })
    self.mqtt_handler.set_will(lwt_topic, lwt_payload, qos=1, retain=True)
    
    # Start handler
    self.mqtt_handler.start()
    
    # Wait for initial connection
    if not self.mqtt_handler.wait_for_connection(timeout=10.0):
        safe_log("Initial MQTT connection timeout - running in degraded mode", logging.WARNING)
```

### 2. Create Publish Wrapper

Add a new method for all MQTT publishing:

```python
def mqtt_publish(self, topic: str, payload: Any, qos: int = 0, retain: bool = False) -> bool:
    """Publish to MQTT using stable handler"""
    if not hasattr(self, 'mqtt_handler'):
        safe_log("MQTT handler not initialized", logging.ERROR)
        return False
    
    # Convert payload to JSON if needed
    if not isinstance(payload, str):
        payload = json.dumps(payload)
    
    return self.mqtt_handler.publish(topic, payload, qos, retain)
```

### 3. Update All Publish Calls

Replace all instances of:
```python
self.mqtt_client.publish(topic, payload, qos=qos, retain=retain)
```

With:
```python
self.mqtt_publish(topic, payload, qos=qos, retain=retain)
```

### 4. Update Cleanup Method

Modify the cleanup method to properly shutdown the MQTT handler:

```python
# In cleanup method, replace MQTT cleanup section with:
try:
    if hasattr(self, 'mqtt_handler'):
        # Publish offline status
        lwt_payload = json.dumps({
            'node_id': self.config.NODE_ID,
            'service': 'camera_detector',
            'status': 'offline',
            'timestamp': datetime.now(timezone.utc).isoformat() + 'Z'
        })
        self.mqtt_publish(
            f"{self.config.TOPIC_HEALTH}/{self.config.NODE_ID}/lwt",
            lwt_payload,
            qos=1,
            retain=True
        )
        
        # Give time for final message
        time.sleep(0.5)
        
        # Stop handler
        self.mqtt_handler.stop()
except Exception as e:
    safe_log(f"Error during MQTT cleanup: {e}")
```

## Testing Plan

1. **Unit Tests**:
   - Test message queue overflow handling
   - Test reconnection logic
   - Test concurrent publishing

2. **Integration Tests**:
   - Run with simulated network delays
   - Test with broker restarts
   - Monitor for disconnections during heavy discovery

3. **Load Tests**:
   - Publish 1000 messages rapidly
   - Run discovery with 100+ cameras
   - Monitor memory usage and connection stability

## Success Metrics

- No MQTT disconnections during normal operation
- Message delivery success rate > 99.9%
- Reconnection time < 5 seconds after network interruption
- Zero message loss for QoS > 0 messages

## Rollback Plan

If issues arise:
1. Revert to original MQTT implementation
2. Increase keepalive timeout to 120 seconds
3. Reduce discovery parallelism
4. Implement discovery throttling

## Alternative Approaches Considered

1. **Using asyncio**: Would require major refactoring of entire codebase
2. **External MQTT proxy**: Adds complexity and another failure point
3. **Reducing discovery frequency**: Doesn't solve the root cause
4. **Using MQTT5**: Not widely supported by all brokers yet

## Next Steps

1. Implement the changes in detect.py
2. Test thoroughly in development environment
3. Monitor for 24 hours before production deployment
4. Create monitoring dashboard for MQTT health metrics