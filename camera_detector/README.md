# 📷 Camera Detector Service

## What Does This Do?

The Camera Detector automatically finds and configures security cameras on your network for wildfire detection. Think of it as a "smart camera finder" that:

- 🔍 **Automatically discovers** cameras on your network (no manual IP entry needed!)
- 🏷️ **Tracks cameras by their unique ID** even when their network address changes
- 🔧 **Configures cameras** for optimal fire detection settings
- 📊 **Monitors camera health** and alerts you if one goes offline
- 🤝 **Integrates with Frigate NVR** for AI-powered fire detection

## Why This Matters

Traditional camera systems require manual configuration of each camera's IP address. When your router assigns a new address (which happens often), the system breaks. Our detector solves this by tracking cameras by their hardware ID (MAC address), so they never get "lost" on the network.

## Quick Start

### For Most Users

The camera detector starts automatically when you deploy the Wildfire Watch system. It will:

1. Scan your network every 5 minutes for cameras
2. Try common usernames/passwords to connect
3. Configure found cameras for fire detection
4. Keep track of cameras even if their IP addresses change

### What You'll See

In the logs, you'll see messages like:
```
[INFO] Starting camera discovery...
[INFO] ONVIF camera found: Hikvision DS-2CD2042WD at 192.168.1.100 (MAC: AA:BB:CC:DD:EE:FF)
[INFO] RTSP stream found at 192.168.1.101 (MAC: 11:22:33:44:55:66)
[INFO] Updated Frigate config with 2 cameras
```

## Resource-Efficient Operation

### Smart Discovery Mode (Default: Enabled)

The camera detector uses intelligent discovery to minimize resource usage:

1. **Startup Phase** (First 3 scans)
   - Aggressive full network scanning every 5 minutes
   - Discovers all cameras on the network
   - Builds initial camera inventory

2. **Stabilization Phase**
   - Monitors for camera count changes
   - Enters steady-state after 3 stable scans

3. **Steady-State Mode**
   - Quick health checks every 60 seconds (only known cameras)
   - Full network scan only every 30 minutes
   - Skips already-discovered IP addresses
   - 90%+ reduction in network traffic and CPU usage

4. **Resource Savings**
   - Health checks: 10 parallel connections vs 200 for full scan
   - No UDP broadcasts in steady state
   - No ARP scanning between full discoveries
   - Typical steady-state load: <1% CPU

## Configuration Options

### Basic Settings (Most Important)

```bash
# Camera login credentials - comma-separated pairs
CAMERA_USERNAME=admin        # Default username to try
CAMERA_PASSWORD=             # Default password (empty for no password)
CAMERA_CREDENTIALS=admin:,admin:admin,admin:12345
```

**💡 Tip**: 
- Add your camera's username:password to `CAMERA_CREDENTIALS` if it's not in the default list
- **Performance**: If you set both `CAMERA_USERNAME` and `CAMERA_PASSWORD`, the detector will use ONLY those credentials, significantly speeding up discovery

### Discovery Settings

```bash
# How often to scan for new cameras (seconds)
DISCOVERY_INTERVAL=300       # 5 minutes default (minimum 30 seconds)

# Enable tracking cameras by MAC address
MAC_TRACKING_ENABLED=true    # Keeps track of cameras when IPs change

# Automatically update Frigate configuration
FRIGATE_UPDATE_ENABLED=true  # Set to false to disable auto-config

# Smart Discovery Settings (reduces resource usage)
SMART_DISCOVERY_ENABLED=true      # Enable intelligent discovery mode
INITIAL_DISCOVERY_COUNT=3         # Number of aggressive startup scans
STEADY_STATE_INTERVAL=1800        # Full scan interval in steady state (30 min)
QUICK_CHECK_INTERVAL=60           # Health check interval (1 min)
```

### Advanced Network Settings

```bash
# Timeouts for camera connections (automatically validated)
RTSP_TIMEOUT=10             # Seconds to wait for video stream (1-60 seconds)
ONVIF_TIMEOUT=5             # Seconds to wait for camera info (1-30 seconds)

# Camera network ports (automatically validated)
RTSP_PORT=554               # Video streaming port (1-65535)
ONVIF_PORT=80               # Camera control port  
HTTP_PORT=80                # Web interface port
MQTT_PORT=1883              # MQTT broker port (1-65535)
```

### Health Monitoring

```bash
# How often to check if cameras are working (automatically validated)
HEALTH_CHECK_INTERVAL=60    # Every minute (minimum 10 seconds)
OFFLINE_THRESHOLD=180       # Mark offline after 3 minutes (minimum 60 seconds)
```

## Common Issues and Solutions

### Problem: Cameras Not Found

**Symptoms**: No cameras appear in logs

**Solutions**:
1. **Check network connection**: Ensure cameras and detector are on same network
2. **Verify camera is ONVIF compatible**: Most modern IP cameras support ONVIF
3. **Add credentials**: Your camera might use non-standard login:
   ```bash
   CAMERA_CREDENTIALS=admin:,admin:admin,root:pass,user:user
   ```
4. **Check firewall**: Ensure ports 80 and 554 aren't blocked

### Problem: Camera Goes Offline Frequently

**Symptoms**: "Camera offline" messages in logs

**Solutions**:
1. **Increase timeout**: Some cameras respond slowly:
   ```bash
   RTSP_TIMEOUT=20
   OFFLINE_THRESHOLD=300
   ```
2. **Check network stability**: Use wired connection if possible
3. **Reduce camera resolution**: High-res streams can overwhelm network

### Problem: Wrong Password Errors

**Symptoms**: "Auth failed" or "401 Unauthorized" in logs

**Solutions**:
1. Find your camera's password (check camera label or manual)
2. Add to credentials list:
   ```bash
   CAMERA_CREDENTIALS=admin:your_password,admin:admin
   ```
3. **Note**: Malformed credentials are automatically handled with fallback defaults

## How It Works (Technical Details)

### Discovery Methods

1. **ONVIF Discovery** (Primary)
   - Uses WS-Discovery protocol to find ONVIF-compatible cameras
   - Gets camera details: manufacturer, model, capabilities
   - Retrieves RTSP URLs for video streaming

2. **mDNS/Avahi Discovery** (Secondary)
   - Finds cameras advertising via Bonjour/Avahi
   - Common with consumer cameras

3. **RTSP Port Scanning** (Fallback)
   - Scans network for open RTSP ports (554)
   - Tries common RTSP paths
   - Last resort for non-ONVIF cameras
   - Uses parallel socket scanning for performance
   - Automatically skips very large networks (>5000 hosts)
   - Prioritizes common camera IP ranges (100-254) on large networks
   - In steady-state, only scans for new IP addresses

**Performance Features**:
- All discovery methods run in parallel using ThreadPoolExecutor
- Adaptive worker counts based on CPU cores (up to 500 workers on high-core systems)
- Automatic scanning of adjacent subnets on large networks
- Concurrent credential and RTSP path testing
- Reduced socket timeout (0.2s) for faster port scanning
- Smart credential handling - uses only provided credentials when available
- Manufacturer-aware RTSP path prioritization
- Immediate rediscovery when cameras go offline (DHCP address change detection)
- Targeted IP range scanning for known camera locations

### MAC Address Tracking

The detector maintains a MAC-to-IP mapping database:
- Uses ARP (Address Resolution Protocol) to get MAC addresses
- Gracefully handles permission issues (ARP scan requires root)
- Falls back to system ARP table and ping methods
- Updates mappings every minute
- Tracks IP history for each camera
- Publishes "ip_changed" events when cameras move
- Thread-safe operations for concurrent access

### Frigate Integration

The detector automatically:
1. Generates optimal Frigate configuration for each camera
2. Sets up dual-stream when available (high-res for recording, low-res for detection)
3. Configures fire/smoke detection parameters
4. Triggers Frigate to reload configuration via MQTT

### MQTT Events Published

- `camera/discovery/{camera_id}` - New camera found
- `camera/status/{camera_id}` - Status changes (online/offline/ip_changed)
- `system/camera_detector_health` - Service health status
- `frigate/config/cameras` - Frigate configuration update

## Reliability and Error Handling

The camera detector includes robust error handling for production environments:

### **Automatic Recovery**
- **MQTT Connection**: Exponential backoff retry with connection limits (max 10 attempts)
- **Resource Cleanup**: Proper cleanup of network connections and file handles
- **Thread Safety**: Thread-safe camera dictionary access with proper locking
- **Memory Management**: Efficient handling of large camera counts and MAC tracking

### **Input Validation**
- **Configuration**: Automatic validation and correction of invalid timeout/port values
- **Network Safety**: Prevention of command injection via malicious IP addresses
- **Credential Parsing**: Robust parsing with fallback to defaults for malformed input
- **RTSP Validation**: Graceful handling of malformed URLs and network timeouts

### **Performance Optimizations**
- **Timeout Enforcement**: Prevents hanging on slow network operations
- **Resource Limits**: Prevents infinite recursion in MAC address lookup
- **Concurrent Safety**: Thread-safe operations for multi-camera environments

## Security Considerations

⚠️ **Important Security Notes**:

1. **Credentials are stored in plain text** in environment variables
2. **Camera streams are not encrypted** unless camera supports RTPS
3. **Use strong passwords** on your cameras
4. **Isolate cameras** on separate network/VLAN if possible
5. **Input Validation**: Service validates all network inputs to prevent injection attacks

## Monitoring and Debugging

### View Logs
```bash
docker logs wildfire-watch_camera_detector_1
```

### Check Camera Status
Look for health reports in MQTT:
```
Topic: system/camera_detector_health
```

### Debug Discovery Issues
Set log level to debug:
```bash
LOG_LEVEL=DEBUG
```

## Advanced Customization

### Adding Custom Discovery Logic

Edit `detect.py` to add camera-specific discovery:

```python
# Add to rtsp_paths list in _check_rtsp_stream()
rtsp_paths = [
    '/cam/realmonitor?channel=1&subtype=0',  # Amcrest main stream
    '/cam/realmonitor?channel=1&subtype=1',  # Amcrest sub stream
    '/your/camera/path',
    # ... existing paths
]
```

### Custom Frigate Configuration

Modify `to_frigate_config()` method in `detect.py`:

```python
# Adjust detection parameters
'detect': {
    'enabled': True,
    'width': 1280,
    'height': 720,
    'fps': 5,  # Adjust FPS
}
```

### Network Optimization

For large networks, adjust scanning:
```bash
# Scan less frequently
DISCOVERY_INTERVAL=600  # 10 minutes

# Increase timeouts for slow networks
RTSP_TIMEOUT=30
ONVIF_TIMEOUT=10
```

## Dependencies

The camera detector requires the following Python packages:
- `paho-mqtt`: MQTT client for communication
- `python-onvif-zeep`: ONVIF camera protocol support
- `ws4py` and `wsdiscovery`: WS-Discovery for camera finding
- `netifaces`: Network interface detection
- `opencv-python` (cv2): RTSP stream validation
- `scapy`: ARP scanning for MAC address tracking
- `python-dotenv`: Environment variable management
- `PyYAML`: Configuration file handling

Note: `concurrent.futures` is used for parallel processing but is part of Python's standard library.

## Learn More

### Camera Protocols
- **ONVIF**: [Official ONVIF Site](https://www.onvif.org/)
- **RTSP**: [RTSP Protocol Guide](https://www.ietf.org/rfc/rfc2326.txt)
- **mDNS**: [Multicast DNS Explanation](https://en.wikipedia.org/wiki/Multicast_DNS)

### Related Documentation
- [Frigate NVR Documentation](https://docs.frigate.video/)
- [MQTT Protocol](https://mqtt.org/)
- [Docker Networking](https://docs.docker.com/network/)

### Troubleshooting Resources
- Camera manufacturer's ONVIF guide
- [Wireshark](https://www.wireshark.org/) for network debugging
- [ONVIF Device Manager](https://sourceforge.net/projects/onvifdm/) for testing

## Testing

The camera detector includes comprehensive tests for all functionality:

```bash
# Run all camera detector tests
python3.12 -m pytest tests/test_detect.py -v

# Run camera discovery tests (includes smart discovery)
python3.12 -m pytest tests/test_camera_discovery.py -v

# Run with test camera credentials
CAMERA_USER=admin CAMERA_PASS=yourpass python3.12 -m pytest tests/test_camera_discovery.py -v
```

**Test Configuration**: Tests use environment variables for camera credentials to avoid hardcoding sensitive information. See `tests/.env.example` for configuration options.

**Important**: The integration tests (`test_camera_discovery_integration` and `test_live_stream_validation`) REQUIRE actual cameras on your network to pass. They use auto-discovery to find cameras, just like production.

Test coverage includes:
- Basic functionality and configuration
- MAC address tracking
- Network detection and RTSP scanning
- RTSP stream validation
- Camera health monitoring
- Smart discovery and steady-state optimization
- Network change detection
- Resource usage optimization

## Getting Help

If cameras aren't detected:
1. Check logs for error messages
2. Verify camera is on same network
3. Try adding camera credentials manually
4. Enable debug logging
5. Check our [troubleshooting guide](../docs/troubleshooting.md)

Remember: The detector is designed to "just work" with most cameras. If you're having issues, it's usually a network or credential problem rather than a detector issue.
