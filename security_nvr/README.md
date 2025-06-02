# 🎥 Security NVR Service (Frigate-based)

## What Does This Do?

The Security NVR Service is your intelligent video surveillance system that:

- 📹 **Records all cameras** continuously to external USB storage
- 🔥 **Detects wildfires** using custom AI models
- 🚀 **Auto-configures hardware** acceleration (Raspberry Pi, Intel, AMD)
- 💾 **Stores 6+ months** of security footage efficiently
- 🔋 **Optimizes power usage** for off-grid operation
- 📊 **Provides searchable history** of all detections
- 🔄 **Integrates with camera_detector** for automatic camera discovery

## Why This Matters

Traditional NVR systems:
- Waste power on continuous high-quality recording
- Miss important events in hours of footage
- Require manual configuration of hardware acceleration
- Don't understand wildfire-specific threats

Our system:
- Records efficiently using motion detection
- Highlights fire/smoke events instantly
- Auto-detects and uses available hardware
- Optimized for wildfire detection

## Quick Start

### Automatic Deployment

The system automatically:
1. Detects available hardware (Coral, Hailo, GPU, etc.)
2. Discovers cameras via camera_detector service
3. Configures optimal settings for your hardware
4. Begins recording to USB storage

### Manual Camera Configuration (Optional)

Add cameras manually in `/config/frigate/custom_cameras.yml`:
```yaml
cameras:
  front_yard:
    ffmpeg:
      inputs:
        - path: rtsp://admin:password@192.168.1.100:554/stream1
          roles:
            - detect
            - record
```

## Hardware Acceleration Support

### Automatic Detection

The system automatically detects and configures:

| Hardware | Detection | Notes |
|----------|-----------|-------|
| **Raspberry Pi 5** | ✅ | Hardware H.265 decode, VideoCore VII |
| **Coral USB** | ✅ | 4 TOPS, USB 3.0 required |
| **Coral PCIe** | ✅ | M.2 or PCIe versions |
| **Hailo-8** | ✅ | 26 TOPS, PCIe |
| **Hailo-8L** | ✅ | 13 TOPS, lower power |
| **Intel QuickSync** | ✅ | Hardware decode/encode |
| **AMD VCE** | ✅ | Hardware decode/encode |
| **NVIDIA GPU** | ✅ | CUDA acceleration |

### Power Efficiency Features

- **Adaptive Quality**: Reduces stream quality when on battery
- **Sub-stream Detection**: Uses low-res streams for AI
- **Hardware Decode**: Offloads CPU for video decoding
- **Smart Recording**: Only records motion + events

## Storage Configuration

### USB Drive Auto-Detection

The system automatically:
1. Detects USB drives at `/dev/sd*`
2. Mounts largest partition to `/media/frigate`
3. Creates folder structure
4. Begins recording

### Manual USB Configuration

```bash
# List USB drives
lsblk | grep sd

# Format drive (WARNING: Erases data!)
sudo mkfs.ext4 -L frigate-storage /dev/sda1

# Mount will happen automatically on restart
```

### Storage Structure

```
/media/frigate/
├── recordings/          # Continuous recordings
│   └── 2024-01-15/     # By date
│       └── front_yard/  # By camera
├── clips/              # Event clips
│   └── 2024-01-15/     
└── exports/            # Manual exports
```

## Wildfire Detection Models

### Pre-trained Models Included

| Model | Hardware | Accuracy | Speed | Power |
|-------|----------|----------|-------|-------|
| `wildfire_coral_lite` | Coral USB/PCIe | Good | Fast | Low |
| `wildfire_hailo8` | Hailo-8 | Excellent | Very Fast | Medium |
| `wildfire_hailo8l` | Hailo-8L | Good | Fast | Low |
| `wildfire_tensorrt` | NVIDIA GPU | Excellent | Fast | High |
| `wildfire_openvino` | Intel CPU/GPU | Good | Medium | Medium |
| `wildfire_cpu` | Any CPU | Fair | Slow | Low |

### Model Selection

The system automatically selects the best model for your hardware. Override with:

```bash
FRIGATE_MODEL=wildfire_hailo8
```

### Custom Model Training

See [model_training/README.md](model_training/README.md) for training your own models.

## Configuration

### Environment Variables

```bash
# Storage
USB_MOUNT_PATH=/media/frigate     # Where to mount USB drive
RECORD_RETAIN_DAYS=180            # How long to keep recordings

# Detection
FRIGATE_MODEL=auto                # auto|wildfire_coral_lite|wildfire_hailo8|etc
DETECTION_THRESHOLD=0.7           # Confidence threshold (0-1)
DETECTION_FPS=5                   # Detections per second

# Power Management
POWER_MODE=balanced               # performance|balanced|powersave
BATTERY_THRESHOLD=20              # Switch to powersave below this %

# Camera Settings
CAMERA_DETECT_WIDTH=1280          # Detection resolution
CAMERA_DETECT_HEIGHT=720
CAMERA_RECORD_QUALITY=70          # Recording quality (0-100)

# Hardware
FRIGATE_HARDWARE=auto             # auto|coral|hailo8|hailo8l|gpu|cpu
HARDWARE_ACCEL=auto               # auto|vaapi|qsv|nvdec|v4l2
```

### Power Profiles

**Performance Mode** (Grid Power):
- Full resolution recording
- Maximum detection FPS
- All cameras active

**Balanced Mode** (Default):
- Adaptive quality based on motion
- 5 FPS detection
- Smart recording

**Power Save Mode** (Low Battery):
- Reduced recording quality
- 2 FPS detection
- Critical cameras only

## Searching Historical Data

### Web Interface

Access at `http://device-ip:5000`:
- Timeline view of all events
- Filter by object type (fire, smoke, person)
- Search by date/time
- Export clips

### Command Line Search

```bash
# Find all fire detections from last week
docker exec frigate-nvr find-events \
  --type fire \
  --days 7 \
  --min-score 0.8

# Export specific timeframe
docker exec frigate-nvr export-recording \
  --camera front_yard \
  --start "2024-01-15 14:00" \
  --end "2024-01-15 15:00" \
  --output /media/exports/
```

### API Access

```python
import requests

# Get recent events
events = requests.get('http://device-ip:5000/api/events', params={
    'label': 'fire',
    'after': 1704067200,  # Unix timestamp
    'limit': 100
}).json()

# Download clip
clip_url = f"http://device-ip:5000/api/events/{event_id}/clip"
```

## Multi-Node Deployment

### Distributed Camera Processing

Deploy multiple NVR nodes with assigned cameras:

**Node 1** (Raspberry Pi 5 + Coral):
```yaml
assigned_cameras:
  - north_perimeter
  - northwest_field
```

**Node 2** (x86 + Hailo-8):
```yaml
assigned_cameras:
  - south_building
  - parking_lot
```

### Failover Configuration

```yaml
failover:
  enabled: true
  peer_nodes:
    - nvr-node-2.local
    - nvr-node-3.local
  takeover_timeout: 30  # seconds
```

## Network Resilience

### Handling Network Issues

The system automatically:
- Buffers recordings during network outages
- Retries failed camera connections
- Stores events locally until MQTT available
- Continues detection without network

### Camera Connection Resilience

```yaml
cameras:
  front_yard:
    ffmpeg:
      retry_interval: 10    # Retry every 10 seconds
      max_retry: -1         # Infinite retries
      input_args:
        - -timeout
        - '5000000'         # 5 second timeout
        - -reconnect
        - '1'
```

## Debugging and Logs

### Log Levels

```bash
# Set debug logging
LOG_LEVEL=debug

# Component-specific debugging
FRIGATE_LOGGER_LEVEL=detector.coral=debug
```

### Key Log Locations

- **Frigate logs**: `docker logs security-nvr`
- **Detection logs**: `/media/frigate/logs/detection.log`
- **Hardware logs**: `/media/frigate/logs/hardware.log`

### Common Debug Commands

```bash
# Check hardware detection
docker exec security-nvr check-hardware

# View detection statistics
docker exec security-nvr stats

# Test camera connection
docker exec security-nvr test-camera front_yard

# Benchmark detection speed
docker exec security-nvr benchmark-detector
```

## Performance Optimization

### Raspberry Pi 5 Specific

```yaml
# Enable hardware acceleration
ffmpeg:
  hwaccel_args:
    - -c:v
    - h264_v4l2m2m
  output_args:
    record:
      - -c:v
      - h265_v4l2m2m  # Hardware H.265 encoding
```

### Intel/AMD Specific

```yaml
# Intel QuickSync
ffmpeg:
  hwaccel_args:
    - -hwaccel
    - qsv
    - -hwaccel_device
    - /dev/dri/renderD128

# AMD
ffmpeg:
  hwaccel_args:
    - -hwaccel
    - vaapi
    - -hwaccel_device
    - /dev/dri/renderD128
```

## Troubleshooting

### Problem: No Hardware Acceleration

**Symptoms**: High CPU usage, laggy video

**Solutions**:
1. Check hardware detection:
   ```bash
   docker exec security-nvr check-hardware
   ```
2. Verify device permissions:
   ```bash
   ls -la /dev/dri/  # For GPU
   ls -la /dev/bus/usb/  # For Coral
   ```
3. Enable debug logging:
   ```bash
   FRIGATE_LOGGER_LEVEL=frigate.detectors=debug
   ```

### Problem: USB Drive Not Detected

**Symptoms**: Recordings not saving

**Solutions**:
1. Check USB detection:
   ```bash
   lsblk | grep sd
   dmesg | grep -i usb
   ```
2. Manual mount:
   ```bash
   sudo mount /dev/sda1 /media/frigate
   ```
3. Check permissions:
   ```bash
   ls -la /media/frigate
   ```

### Problem: Detection Not Working

**Symptoms**: No fire/smoke alerts

**Solutions**:
1. Check model loading:
   ```bash
   docker exec security-nvr test-model
   ```
2. Verify camera feed:
   ```bash
   docker exec security-nvr test-camera front_yard
   ```
3. Lower detection threshold:
   ```bash
   DETECTION_THRESHOLD=0.5
   ```

## Advanced Features

### Custom Detection Zones

```yaml
cameras:
  front_yard:
    motion:
      mask:
        - 0,0,300,0,300,300,0,300  # Ignore this area
    zones:
      high_risk:
        coordinates: 400,0,800,0,800,600,400,600
        fire_threshold: 0.6  # More sensitive in high risk areas
```

### Integration with Home Assistant

```yaml
# configuration.yaml
frigate:
  host: security-nvr.local
  port: 5000
  client_id: frigate
  stats: true
```

### Alerts and Notifications

```yaml
notifications:
  mqtt:
    enabled: true
    topic: frigate/events
  webhook:
    enabled: true
    url: https://your-webhook.com/frigate
    events:
      - fire
      - smoke
```

## Best Practices

1. **Storage Management**
   - Use USB 3.0+ drives for best performance
   - Format as ext4 for Linux compatibility
   - Monitor disk space with alerts

2. **Camera Configuration**
   - Use sub-streams for detection
   - Main stream for recording only
   - Match FPS to your needs (5-10 typical)

3. **Power Management**
   - Enable hardware acceleration
   - Use motion detection
   - Reduce quality during low battery

4. **Network Setup**
   - Isolate cameras on VLAN
   - Use wired connections when possible
   - Configure retry settings

## Learn More

- [Frigate Documentation](https://docs.frigate.video/)
- [Hardware Acceleration Guide](https://docs.frigate.video/configuration/hardware_acceleration)
- [Custom Model Training](model_training/README.md)
- [Multi-Node Setup](docs/multi-node-nvr.md)

## Getting Help

If detection isn't working:
1. Check hardware is detected
2. Verify cameras are accessible
3. Review detection logs
4. Test with lower thresholds
5. Try a different model

Remember: The system is designed to work with minimal configuration while allowing deep customization for advanced users.
