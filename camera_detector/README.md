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

## Configuration Options

### Basic Settings (Most Important)

```bash
# Camera login credentials - comma-separated pairs
CAMERA_USERNAME=admin        # Default username to try
CAMERA_PASSWORD=             # Default password (empty for no password)
CAMERA_CREDENTIALS=admin:,admin:admin,admin:12345,admin:password
```

**💡 Tip**: Add your camera's username:password to `CAMERA_CREDENTIALS` if it's not in the default list.

### Discovery Settings

```bash
# How often to scan for new cameras (seconds)
DISCOVERY_INTERVAL=300       # 5 minutes default

# Enable tracking cameras by MAC address
MAC_TRACKING_ENABLED=true    # Keeps track of cameras when IPs change

# Automatically update Frigate configuration
FRIGATE_UPDATE_ENABLED=true  # Set to false to disable auto-config
```

### Advanced Network Settings

```bash
# Timeouts for camera connections
RTSP_TIMEOUT=10             # Seconds to wait for video stream
ONVIF_TIMEOUT=5             # Seconds to wait for camera info

# Camera network ports
RTSP_PORT=554               # Video streaming port
ONVIF_PORT=80               # Camera control port
HTTP_PORT=80                # Web interface port
```

### Health Monitoring

```bash
# How often to check if cameras are working
HEALTH_CHECK_INTERVAL=60    # Every minute
OFFLINE_THRESHOLD=180       # Mark offline after 3 minutes
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

### MAC Address Tracking

The detector maintains a MAC-to-IP mapping database:
- Uses ARP (Address Resolution Protocol) to get MAC addresses
- Updates mappings every minute
- Tracks IP history for each camera
- Publishes "ip_changed" events when cameras move

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

## Security Considerations

⚠️ **Important Security Notes**:

1. **Credentials are stored in plain text** in environment variables
2. **Camera streams are not encrypted** unless camera supports RTPS
3. **Use strong passwords** on your cameras
4. **Isolate cameras** on separate network/VLAN if possible

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

## Getting Help

If cameras aren't detected:
1. Check logs for error messages
2. Verify camera is on same network
3. Try adding camera credentials manually
4. Enable debug logging
5. Check our [troubleshooting guide](../docs/troubleshooting.md)

Remember: The detector is designed to "just work" with most cameras. If you're having issues, it's usually a network or credential problem rather than a detector issue.
