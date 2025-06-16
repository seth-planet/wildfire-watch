# 🎥 Security NVR Service (Frigate-based)

## What Does This Do?

The Security NVR Service is your intelligent video surveillance system that:

- 📹 **Records all cameras** with motion-based retention
- 🔥 **Detects wildfires** using AI models optimized for edge devices
- 🚀 **Auto-detects hardware** acceleration (Coral, Hailo, GPU)
- 💾 **Manages storage** efficiently with event-based recording
- 🔋 **Optimizes performance** based on available hardware
- 📊 **Provides real-time alerts** via MQTT
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
1. Detects available hardware via `hardware_detector.py`
2. Discovers cameras from the camera_detector service
3. Manages USB storage with `usb_manager.py`
4. Configures Frigate using `camera_manager.py`

### Key Components

- **hardware_detector.py** - Identifies Coral, Hailo, GPU acceleration
- **camera_manager.py** - Integrates discovered cameras into Frigate
- **usb_manager.py** - Auto-mounts and manages USB storage
- **nvr_base_config.yml** - Base Frigate configuration template
- **entrypoint.sh** - Startup orchestration

### Manual Camera Configuration (Optional)

Add cameras manually in `/config/custom_cameras.yml`:
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

The hardware detector (`hardware_detector.py`) automatically identifies and configures:

| Hardware | Status | Notes |
|----------|--------|-------|
| **Raspberry Pi 5** | ✅ | Hardware H.264/H.265 decode via V4L2 |
| **Coral USB** | ✅ | Edge TPU support, requires USB 3.0 |
| **Coral PCIe** | ✅ | M.2 or PCIe versions |
| **Hailo-8** | ✅ | 26 TOPS, requires PCIe slot |
| **Intel QuickSync** | ✅ | Hardware decode/encode via VAAPI |
| **NVIDIA GPU** | ✅ | CUDA/NVDEC acceleration |
| **CPU Fallback** | ✅ | Works on any system without acceleration |

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

### Model Integration

The system uses models from the [Model Converter](../converted_models/README.md) service:

| Model Format | Hardware | Notes |
|--------------|----------|-------|
| `.tflite` | Coral TPU | INT8 quantized models |
| `.hef` | Hailo-8/8L | Compiled for Hailo accelerator |
| `.onnx` | TensorRT/OpenVINO | Optimized for GPU/CPU |
| `.pt` | CPU | PyTorch format (fallback) |

### Model Configuration

Models are specified in the Frigate configuration:

```yaml
detectors:
  coral:
    type: edgetpu
    device: usb
    model:
      path: /models/yolo8l_rev5_640x640.tflite
      width: 640
      height: 640
      # Note: Use 320x320 models for Raspberry Pi or low-power devices
```

### Model Labels

Wildfire-specific labels:
- `fire` - Active flames
- `smoke` - Smoke plumes
- `person` - Human presence (safety)
- `vehicle` - Cars/trucks (context)

## Configuration

### Environment Variables

```bash
# MQTT Connection
FRIGATE_MQTT_HOST=mqtt_broker     # MQTT broker hostname
FRIGATE_MQTT_PORT=8883            # MQTT port (8883 for TLS)
FRIGATE_MQTT_TLS=true             # Enable TLS encryption
FRIGATE_CLIENT_ID=frigate-nvr     # MQTT client ID

# Storage
USB_MOUNT_PATH=/media/frigate     # USB storage mount point
RECORD_RETAIN_DAYS=180            # Recording retention period

# Detection
FRIGATE_DETECTOR=auto             # auto|coral|hailo|gpu|cpu
DETECTION_THRESHOLD=0.7           # AI confidence threshold
DETECTION_FPS=5                   # Detections per second
MIN_CONFIDENCE=0.7                # Minimum confidence for fire alerts

# Camera Settings
CAMERA_DETECT_WIDTH=640           # Detection resolution (640 optimal, 320 for limited hardware)
CAMERA_DETECT_HEIGHT=640
CAMERA_RECORD_QUALITY=70          # Recording quality percentage

# Hardware Acceleration
HARDWARE_ACCEL=auto               # auto|v4l2|vaapi|qsv|nvdec
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

## Integration with Wildfire Watch

### MQTT Events

The NVR publishes fire detection events to MQTT:

```json
{
  "type": "new",
  "before": {
    "id": "1234567890.123456-abc123",
    "camera": "front_yard",
    "frame_time": 1234567890.123456,
    "label": "fire",
    "score": 0.85,
    "box": [320, 180, 480, 360]
  }
}
```

Published to: `frigate/events`

### Camera Integration

The service integrates with [Camera Detector](../camera_detector/README.md):
1. Monitors `cameras/discovered` topic for new cameras
2. Automatically adds discovered cameras to Frigate config
3. Applies optimal settings based on camera capabilities
4. Handles camera disconnections gracefully

### Fire Consensus Integration

Works with [Fire Consensus](../fire_consensus/README.md):
- Publishes detection events to `frigate/events`
- Includes confidence scores and bounding boxes
- Provides camera location for multi-camera validation

## Web Interface

Access Frigate UI at `http://device-ip:5000`:
- Live view of all cameras
- Timeline of detection events
- Review recordings
- Export clips
- System statistics

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

## Monitoring and Debugging

### View Logs

```bash
# Frigate logs
docker logs security_nvr -f

# Hardware detection logs
docker exec security_nvr cat /tmp/hardware_detection.log

# Check detector status
docker exec security_nvr python3 /opt/frigate/frigate/detectors/detector_status.py
```

### Common Issues

**No detections:**
- Check model is loaded: Look for "detector started" in logs
- Verify camera streams are working in UI
- Lower detection threshold: `MIN_CONFIDENCE=0.5`

**High CPU usage:**
- Ensure hardware acceleration is detected
- Reduce detection FPS: `DETECTION_FPS=3`
- Use substreams for detection

**Storage issues:**
- Check USB mount: `docker exec security_nvr df -h`
- Verify write permissions: `docker exec security_nvr touch /media/frigate/test`
- Monitor disk usage in Frigate UI

## Performance Optimization

### Performance Metrics

Expected performance with proper hardware acceleration:

| Hardware | Inference Speed | Max FPS | CPU Usage | Bandwidth |
|----------|----------------|---------|-----------|-----------|
| **Coral USB** | 15-20ms | 10 FPS | 5-15% | 2-4 Mbps/camera |
| **Hailo-8** | 10-25ms | 15 FPS | 5-10% | 2-4 Mbps/camera |
| **GPU (RTX)** | 8-12ms | 20 FPS | 10-20% | 2-4 Mbps/camera |
| **CPU Only** | 100-200ms | 5 FPS | 60-80% | 2-4 Mbps/camera |

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

## Important Notes

### Model Compatibility

- **Coral TPU**: Requires INT8 quantized models from [Model Converter](../converted_models/README.md)
  - **Important**: Requires Python 3.8 for `tflite_runtime` compatibility
  - See [Coral Python 3.8 Requirements](../docs/coral_python38_requirements.md) for details
- **Hailo**: Uses compiled `.hef` format models
- **GPU**: Best with ONNX or TensorRT models
- **CPU**: Can use any format but slower

### Storage Requirements

- **Minimum**: 64GB USB drive for 30 days retention
- **Recommended**: 256GB+ for 180 days
- **Format**: ext4 recommended for Linux
- **Speed**: USB 3.0 or better

### Network Bandwidth

- Each camera: ~2-4 Mbps for main stream
- Detection stream: ~0.5-1 Mbps
- Plan network capacity accordingly

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

## Related Documentation

- [Camera Detector](../camera_detector/README.md) - Automatic camera discovery
- [Model Converter](../converted_models/README.md) - Convert models for your hardware
- [Fire Consensus](../fire_consensus/README.md) - Multi-camera fire validation
- [Hardware Guide](../docs/hardware.md) - Recommended hardware configurations
- [Multi-Node Setup](../docs/multi-node.md) - Scaling to multiple locations

## External Resources

- [Frigate Documentation](https://docs.frigate.video/) - Official Frigate docs
- [Hardware Acceleration](https://docs.frigate.video/configuration/hardware_acceleration) - Frigate hardware guide

## Troubleshooting Summary

1. **No cameras showing**: Check camera_detector service is running
2. **No detections**: Verify model is loaded and lower threshold
3. **High CPU**: Enable hardware acceleration
4. **Storage full**: Reduce retention days or add larger drive
5. **Can't access UI**: Check port 5000 is not blocked

The system prioritizes reliability and automatic configuration while supporting advanced customization when needed.
