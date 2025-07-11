# Wildfire Watch Web Interface

## Overview

The `web_interface` service provides a real-time web-based dashboard for monitoring and debugging the Wildfire Watch system. It displays system health, GPIO states, fire detection status, camera information, and event logs through a lightweight, secure interface.

## Features

### Core Functionality
- **Real-time Status Monitoring**: View live system status across all services
- **Service Health Dashboard**: Monitor health of all microservices (gpio_trigger, fire_consensus, camera_detector, etc.)
- **GPIO State Visualization**: See current pin states for pumps, valves, and ignition
- **Event Timeline**: Scrollable log of recent fire detections, triggers, and system events
- **Camera Grid**: Display discovered cameras and their detection counts
- **MQTT Diagnostics**: View connection status and recent messages

### Security Features
- **LAN-Only Access**: Restricted to local network by default
- **Read-Only Interface**: No control capabilities in standard mode
- **Optional Debug Mode**: Protected control features with token authentication
- **Input Sanitization**: All displayed data is properly escaped
- **Rate Limiting**: API endpoints protected against abuse

### Performance
- **Lightweight Design**: Optimized for Raspberry Pi deployment
- **Server-Side Rendering**: Minimal JavaScript for better performance
- **Circular Buffer**: Memory-efficient event storage (configurable size)
- **HTMX Updates**: Efficient partial page updates without full reloads

## Configuration

The service is configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `STATUS_PANEL_PORT` | `8080` | HTTP server port |
| `STATUS_PANEL_DEBUG` | `false` | Enable debug mode |
| `STATUS_PANEL_TOKEN` | `""` | Debug mode access token |
| `STATUS_PANEL_BUFFER` | `1000` | Max MQTT events to store |
| `STATUS_PANEL_REFRESH` | `15` | UI refresh interval (seconds) |
| `MQTT_BROKER` | `mqtt_broker` | MQTT broker hostname |
| `MQTT_PORT` | `8883` | MQTT broker port |
| `MQTT_TLS` | `true` | Enable TLS for MQTT |
| `LOG_LEVEL` | `info` | Logging verbosity |

## Architecture

### Technology Stack
- **Backend**: FastAPI (Python 3.12)
- **Frontend**: Jinja2 templates + HTMX + Alpine.js
- **Styling**: Tailwind CSS (compiled)
- **MQTT Client**: paho-mqtt with TLS support
- **Container**: Multi-platform Docker support (amd64/arm64)

### Service Integration
The web interface subscribes to MQTT topics from all services:
```
system/+/health          # Service health reports
system/trigger_telemetry/+   # GPIO trigger telemetry
fire/trigger             # Fire detection events
fire/consensus_state     # Consensus state updates
camera/discovery/+       # Camera discovery events
frigate/events          # Frigate detection events
gpio/status             # GPIO state changes
```

### Directory Structure
```
web_interface/
├── app.py              # FastAPI application
├── config.py           # Configuration management
├── mqtt_handler.py     # MQTT subscription handler
├── models.py           # Pydantic data models
├── security.py         # Security middleware
├── templates/          # Jinja2 templates
│   ├── base.html
│   ├── index.html
│   ├── debug.html
│   └── components/
├── static/             # CSS and JavaScript
│   ├── style.css
│   └── app.js
├── Dockerfile
├── requirements.txt
├── entrypoint.sh
└── nsswitch.conf
```

## Usage

### Accessing the Interface
Once deployed, access the web interface at:
- **Production**: `http://localhost:8080/` (bound to localhost only by default)
- **LAN Access**: Set `STATUS_PANEL_PORT=0.0.0.0:8080` in docker-compose to allow LAN access
- **Development**: `http://localhost:8080/`

The interface is restricted to localhost only by default for maximum security. To allow LAN access, you must explicitly configure it.

### Debug Mode
To enable debug mode for manual triggers and advanced diagnostics:

1. Set environment variables:
   ```bash
   STATUS_PANEL_DEBUG=true
   STATUS_PANEL_TOKEN=your-secure-token
   ```

2. Access debug panel at:
   ```
   http://<device-ip>:8080/debug?token=your-secure-token
   ```

### API Endpoints
- `GET /` - Main dashboard
- `GET /api/status` - Current system status (JSON)
- `GET /api/events` - Recent events (JSON)
- `GET /api/health` - Service health check
- `GET /debug` - Debug panel (requires token)
- `POST /api/emergency/stop` - Emergency pump stop (debug mode only)

## Development

### Running Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export MQTT_BROKER=localhost
export MQTT_TLS=false
export STATUS_PANEL_DEBUG=true

# Run the application
python app.py
```

### Testing
```bash
# Install test dependencies
pip install -r requirements_test.txt

# Run tests
pytest tests/
```

### Building the Container
```bash
# Build for current platform
docker build -t wildfire-watch/web_interface .

# Build for multiple platforms
docker buildx build --platform linux/amd64,linux/arm64 -t wildfire-watch/web_interface .
```

## Security Considerations

### LAN-Only Access
The service implements IP filtering to restrict access to RFC1918 addresses:
- 10.0.0.0/8
- 172.16.0.0/12
- 192.168.0.0/16

### Authentication
- No authentication required for read-only monitoring
- Debug mode requires token authentication
- Consider adding Basic Auth for production deployments

### Data Safety
- All MQTT payloads are sanitized before display
- Templates use auto-escaping to prevent XSS
- No direct GPIO control from web interface
- All control actions go through MQTT with validation

## Troubleshooting

### Common Issues

1. **Cannot access the interface**
   - Verify you're on the same LAN as the device
   - Check firewall settings
   - Ensure the service is running: `docker logs web-interface`

2. **No data displayed**
   - Check MQTT connection: Look for "MQTT Connected" indicator
   - Verify other services are running and publishing
   - Check browser console for JavaScript errors

3. **Debug mode not working**
   - Ensure `STATUS_PANEL_DEBUG=true` is set
   - Verify token matches `STATUS_PANEL_TOKEN`
   - Check logs for authentication errors

### Logging
View service logs:
```bash
docker logs -f web-interface
```

Enable debug logging:
```bash
export LOG_LEVEL=debug
```

## Performance Tuning

### For Raspberry Pi 3/4
- Reduce buffer size: `STATUS_PANEL_BUFFER=500`
- Increase refresh interval: `STATUS_PANEL_REFRESH=30`
- Disable debug mode in production

### For Raspberry Pi 5 / x86
- Can handle default settings
- Consider increasing buffer for longer history
- Enable gzip compression (default)

## Contributing

When adding new features:
1. Follow the existing FastAPI patterns
2. Use server-side rendering where possible
3. Maintain LAN-only security model
4. Add appropriate tests
5. Update this documentation

## License

Part of the Wildfire Watch system. See main project LICENSE.