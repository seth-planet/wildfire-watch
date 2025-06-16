# 📡 MQTT Broker Service

## What Does This Do?

The MQTT Broker is the "communication hub" for your wildfire detection system. Think of it as a smart switchboard that:

- 📬 **Routes messages** between cameras, detectors, and sprinklers
- 🔒 **Secures communications** with encryption (optional)
- 💾 **Stores important messages** so they're not lost
- 🌐 **Enables remote monitoring** via web browsers
- 🔄 **Handles network issues** automatically

## Why This Matters

Without the MQTT broker, your system components can't talk to each other. The broker ensures:
- Fire detections reach the consensus system
- Trigger commands activate sprinklers
- Camera status updates are distributed
- Everything keeps working even with network hiccups

## Quick Start

### Default Setup

The broker starts automatically with:
- **Port 1883**: Standard MQTT (unencrypted)
- **Port 8883**: Secure MQTT (encrypted with default certificates)
- **Port 9001**: WebSocket access (for web dashboards)

**⚠️ Security Note**: The system includes default certificates for easy testing. These are INSECURE and must be replaced for production use (see Security section below).

### What You'll See

Normal startup logs:
```
Starting Avahi mDNS daemon...
Publishing MQTT service via mDNS...
Warning: TLS certificates found at /mnt/data/certs/
Only plain MQTT on port 1883 will be available
==================================================
Wildfire Watch MQTT Broker Starting
Plain MQTT: Port 1883
Secure MQTT: Port 8883 (if certs available)
WebSocket: Port 9001
Hostname: mqtt_broker
==================================================
```

## Configuration Files

### Main Configuration (`mosquitto.conf`)

The main settings that control the broker:

```conf
# Message persistence - saves messages to disk
persistence true
persistence_location /mosquitto/data/
autosave_interval 30        # Save every 30 seconds

# Connection settings
max_connections -1          # Unlimited connections
keepalive_interval 60       # Check connection every 60s
max_keepalive 120          # Allow up to 120s keepalive

# Message settings
max_inflight_messages 40    # Messages "in flight" per client
max_queued_messages 10000   # Queue size when client offline
retain_available true       # Allow retained messages
```

### TLS/Security Configuration (`conf.d/tls.conf`)

For encrypted communications:

```conf
# Secure listener on port 8883
listener 8883

# Certificate files
cafile /mnt/data/certs/ca.crt      # Certificate Authority
certfile /mnt/data/certs/server.crt # Server certificate
keyfile /mnt/data/certs/server.key  # Server private key

# Security settings
require_certificate false    # Don't require client certs
allow_anonymous true        # Allow any client to connect
```

**💡 To enable encryption**: Place certificate files in `/mnt/data/certs/`

### WebSocket Configuration (`conf.d/websockets.conf`)

For web browser access:

```conf
# WebSocket listener on port 9001
listener 9001
protocol websockets

# Optional: Secure WebSocket (uncomment to enable)
# listener 9443
# protocol websockets
# cafile /mnt/data/certs/ca.crt
# certfile /mnt/data/certs/server.crt
# keyfile /mnt/data/certs/server.key
```

### Bridge Configuration (`conf.d/bridge.conf`)

For cloud connectivity (optional):

```conf
# Uncomment to enable cloud relay
# connection wildfire_cloud_bridge
# address your.cloud.broker:8883
# topic fire/# out 2          # Send fire events to cloud
# topic system/# out 1        # Send system status to cloud
# bridge_insecure false       # Require TLS
# cleansession false          # Preserve messages
```

## Common Tasks

### Enabling Encryption

1. Generate certificates (see scripts/generate_certs.sh)
2. Copy certificates to `/mnt/data/certs/`
3. Restart the broker
4. Clients will automatically use port 8883

### Connecting from Web Browser

Use MQTT.js or similar library:
```javascript
const client = mqtt.connect('ws://your-device:9001')
client.subscribe('fire/#')
client.on('message', (topic, message) => {
  console.log(`${topic}: ${message}`)
})
```

### Monitoring Broker Status

View active connections:
```bash
docker exec mqtt_broker mosquitto_sub -t '$SYS/broker/clients/connected' -C 1
```

View message statistics:
```bash
docker exec mqtt_broker mosquitto_sub -t '$SYS/broker/messages/#' -v
```

## Troubleshooting

### Problem: Clients Can't Connect

**Symptoms**: "Connection refused" errors

**Solutions**:
1. **Check broker is running**: 
   ```bash
   docker ps | grep mqtt_broker
   ```
2. **Verify network**: Ensure client and broker on same network
3. **Check firewall**: Ports 1883, 8883, 9001 must be open
4. **Test locally**: 
   ```bash
   docker exec mqtt_broker mosquitto_pub -t test -m "hello"
   ```

### Problem: Messages Being Lost

**Symptoms**: Detections not reaching consensus service

**Solutions**:
1. **Check persistence**: Ensure `/mosquitto/data` is writable
2. **Increase queue size**:
   ```conf
   max_queued_messages 50000
   ```
3. **Check client subscriptions**: Verify topics match

### Problem: High Memory Usage

**Symptoms**: Broker using excessive RAM

**Solutions**:
1. **Limit message size**:
   ```conf
   message_size_limit 1048576  # 1MB max
   ```
2. **Reduce queue sizes**:
   ```conf
   max_queued_messages 1000
   ```
3. **Enable memory limit**:
   ```conf
   memory_limit 134217728  # 128MB
   ```

## Security Best Practices

### ⚠️ Default Certificates Warning

**The system includes default TLS certificates for instant deployment. These are INSECURE!**

See [Security Best Practices](../docs/security.md) for complete hardening guide.

## Performance Tuning

### For Small Networks (< 10 devices)
```conf
# Default settings work well
max_connections 100
```

### For Medium Networks (10-50 devices)
```conf
max_connections 500
max_inflight_messages 100
max_queued_messages 50000
```

### For Large Networks (50+ devices)
```conf
max_connections 2000
max_inflight_messages 200
max_queued_messages 100000
# Consider multiple brokers with bridging
```

## Monitoring and Debugging

### View All Messages
```bash
docker exec mqtt_broker mosquitto_sub -t '#' -v
```

### View Specific Topics
```bash
# Fire detections only
docker exec mqtt_broker mosquitto_sub -t 'fire/#' -v

# System health only
docker exec mqtt_broker mosquitto_sub -t 'system/#' -v
```

### Enable Debug Logging
Add to mosquitto.conf:
```conf
log_type all
log_dest file /mosquitto/log/debug.log
```

### Common MQTT Topics in Wildfire Watch

- `fire/detection/{camera_id}` - Camera fire detections
- `fire/trigger` - Sprinkler activation commands
- `fire/consensus` - Consensus decisions
- `camera/discovery/{camera_id}` - New cameras found
- `camera/status/{camera_id}` - Camera online/offline
- `system/+/health` - Component health reports
- `system/telemetry` - Camera node telemetry (see [Camera Telemetry](../cam_telemetry/README.md))
- `system/telemetry/{camera_id}/lwt` - Camera offline notifications
- `frigate/events` - Frigate NVR events

## Advanced Features

### Enabling Bridge Mode

To relay messages to cloud:

1. Uncomment bridge configuration in `conf.d/bridge.conf`
2. Replace `your.cloud.broker` with actual address
3. Add cloud certificates if using TLS
4. Restart broker

### Custom Plugins

Mosquitto supports custom auth/ACL plugins:
```conf
auth_plugin /mosquitto/plugins/my-plugin.so
auth_opt_db_host localhost
auth_opt_db_port 5432
```

### High Availability Setup

For mission-critical deployments:
1. Run multiple brokers
2. Configure bridging between them
3. Use keepalived for failover
4. Consider MQTT cluster solutions

## Integration Examples

### Python Client
```python
import paho.mqtt.client as mqtt

client = mqtt.Client()
client.connect("mqtt_broker", 1883)
client.subscribe("fire/#")
client.on_message = lambda c,u,m: print(f"{m.topic}: {m.payload}")
client.loop_forever()
```

### Node.js Client
```javascript
const mqtt = require('mqtt')
const client = mqtt.connect('mqtt://mqtt_broker:1883')

client.subscribe('fire/#')
client.on('message', (topic, message) => {
  console.log(`${topic}: ${message.toString()}`)
})
```

### Home Assistant Integration
```yaml
mqtt:
  broker: mqtt_broker
  port: 1883
  discovery: true
  discovery_prefix: homeassistant
```

## Learn More

### MQTT Resources
- [MQTT.org](https://mqtt.org/) - Protocol specification
- [Mosquitto Documentation](https://mosquitto.org/documentation/)
- [MQTT Best Practices](https://www.hivemq.com/mqtt-essentials/)

### Related Tools
- [MQTT Explorer](http://mqtt-explorer.com/) - GUI client
- [Node-RED](https://nodered.org/) - Visual automation
- [Grafana](https://grafana.com/) - Metrics visualization

## Getting Help

If the broker isn't working:
1. Check container logs: `docker logs mqtt_broker`
2. Verify network connectivity
3. Test with mosquitto_sub/pub
4. Check disk space for persistence
5. Review configuration syntax

Remember: The MQTT broker is the backbone of your system. Keep it running smoothly, and everything else will follow!
