# 📊 Camera Telemetry Service

## What Does This Do?

The Camera Telemetry Service monitors the health and status of each camera node in the wildfire detection system. It publishes regular heartbeat messages containing:

- **System health**: CPU, memory, disk usage
- **Camera status**: Online/offline state
- **Configuration snapshot**: Current RTSP URL, model path, detector backend
- **Uptime tracking**: How long the camera has been running

## Why This Matters

Without telemetry, you won't know if:
- A camera has gone offline
- A node is running out of disk space
- CPU/memory usage is too high
- Configuration has drifted from expected values

The telemetry service ensures you can proactively monitor and maintain your wildfire detection network.

## Quick Start

### Default Behavior

The service publishes telemetry every 60 seconds to the MQTT topic `system/telemetry` with:
- Camera ID (defaults to hostname)
- System metrics (if psutil is available)
- Configuration snapshot
- Timestamp

### What You'll See

Normal telemetry message:
```json
{
  "camera_id": "cam-north-01",
  "timestamp": "2024-01-15T10:30:00Z",
  "status": "online",
  "backend": "coral",
  "config": {
    "rtsp_url": "rtsp://192.168.1.100:554/stream",
    "model_path": "/models/wildfire_detector.tflite"
  },
  "free_disk_mb": 15234.5,
  "total_disk_mb": 32768.0,
  "memory_percent": 45.2,
  "cpu_percent": 12.5,
  "uptime_seconds": 86400
}
```

## Configuration

### Environment Variables

```bash
# MQTT Connection
MQTT_BROKER=mqtt_broker       # MQTT broker hostname

# Identity
CAMERA_ID=cam-north-01        # Unique camera identifier (default: hostname)

# Telemetry Settings  
TELEMETRY_INTERVAL=60         # Seconds between telemetry reports
TELEMETRY_TOPIC=system/telemetry  # Base topic for telemetry

# Last Will Testament
LWT_TOPIC=system/telemetry/{camera_id}/lwt  # Offline notification topic

# Configuration Snapshot
RTSP_STREAM_URL=rtsp://...   # Current RTSP URL
MODEL_PATH=/models/fire.tflite # Current model path
DETECTOR=coral                 # Detector backend (coral/cpu/gpu)
```

## Features

### Last Will Testament (LWT)

The service configures MQTT Last Will Testament to automatically publish an offline message if the connection drops:

```json
{
  "camera_id": "cam-north-01",
  "status": "offline",
  "timestamp": "2024-01-15T10:31:00Z"
}
```

This ensures immediate notification when a camera goes offline unexpectedly.

### System Metrics Collection

If `psutil` is installed, the service collects:
- **Disk usage**: Free and total space in MB
- **Memory usage**: Percentage of RAM used
- **CPU usage**: Current CPU percentage
- **Uptime**: Seconds since system boot

If `psutil` is not available, metrics are omitted but telemetry continues.

### Automatic Reconnection

The service automatically reconnects to MQTT if the connection is lost, ensuring continuous monitoring even with network issues.

### Lightweight Design

- Minimal CPU/memory footprint
- Non-blocking metric collection
- Graceful degradation without psutil
- Thread-based scheduling

## Integration

### Monitoring Dashboard

Subscribe to telemetry topics to build a monitoring dashboard:

```python
import paho.mqtt.client as mqtt
import json

def on_message(client, userdata, msg):
    if msg.topic.endswith("/lwt"):
        # Handle offline notification
        data = json.loads(msg.payload)
        alert(f"Camera {data['camera_id']} is offline!")
    else:
        # Process telemetry
        data = json.loads(msg.payload)
        update_dashboard(data)

client = mqtt.Client()
client.on_message = on_message
client.connect("mqtt_broker", 1883)
client.subscribe("system/telemetry/+")
client.subscribe("system/telemetry/+/lwt")
client.loop_forever()
```

### Health Checks

Use telemetry for automated health checks:

```python
# Check disk space
if data["free_disk_mb"] < 1000:
    alert("Low disk space on " + data["camera_id"])

# Check CPU usage
if data["cpu_percent"] > 80:
    alert("High CPU usage on " + data["camera_id"])

# Check uptime (detect recent reboots)
if data["uptime_seconds"] < 300:
    alert("Recent reboot detected on " + data["camera_id"])
```

### Grafana Integration

Create Grafana dashboards using MQTT datasource:

1. Add MQTT datasource pointing to your broker
2. Create queries for each camera:
   - Topic: `system/telemetry`
   - JSON path: `$.camera_id`, `$.cpu_percent`, etc.
3. Build dashboards with:
   - CPU/Memory gauges
   - Disk space alerts
   - Uptime tracking
   - Offline notifications

## Troubleshooting

### No Telemetry Messages

**Check MQTT connection:**
```bash
mosquitto_sub -h mqtt_broker -t "system/telemetry/#" -v
```

**Verify service is running:**
```bash
docker logs cam_telemetry
```

### Missing System Metrics

**Install psutil:**
```bash
pip install psutil
```

Or in Docker:
```dockerfile
RUN pip install psutil
```

### Camera Shows Offline

**Check network connectivity:**
```bash
ping mqtt_broker
```

**Verify MQTT broker is accessible:**
```bash
telnet mqtt_broker 1883
```

## Advanced Usage

### Custom Metrics

Extend telemetry with custom metrics:

```python
# In telemetry.py
def get_custom_metrics():
    return {
        "temperature_c": read_cpu_temp(),
        "network_latency_ms": ping_broker(),
        "active_detections": count_detections()
    }

# In publish_telemetry()
payload.update(get_custom_metrics())
```

### Multi-Camera Nodes

For nodes with multiple cameras, use unique IDs:

```bash
CAMERA_ID=cam-north-01-stream1
CAMERA_ID=cam-north-01-stream2
```

### High-Frequency Monitoring

For critical systems, reduce interval:

```bash
TELEMETRY_INTERVAL=10  # Every 10 seconds
```

Note: Balance between monitoring needs and network traffic.

## Performance Impact

The telemetry service is designed for minimal impact:
- CPU: < 1% average usage
- Memory: < 20MB RAM
- Network: ~1KB per telemetry message
- Disk: No persistent storage required

## Security Considerations

- Telemetry contains configuration details (RTSP URLs)
- Use MQTT TLS for encrypted transport
- Consider filtering sensitive data in production
- Monitor for unusual telemetry patterns (potential attacks)

## Related Services

- [**Camera Detector**](../camera_detector/README.md): Discovers and configures cameras
- [**Fire Consensus**](../fire_consensus/README.md): Uses telemetry for camera health checks
- [**MQTT Broker**](../mqtt_broker/README.md): Routes telemetry messages
- [**GPIO Trigger**](../gpio_trigger/README.md): Also publishes telemetry to `system/trigger_telemetry`

## See Also

- [System Architecture](../README.md#system-architecture) - Overview of all services
- [MQTT Topics](../mqtt_broker/README.md#common-mqtt-topics-in-wildfire-watch) - Complete topic reference
- [Multi-Node Setup](../docs/multi-node.md) - Telemetry in distributed deployments