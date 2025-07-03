# Hailo Quick Start Guide for Wildfire Watch

## 5-Minute Setup

### Prerequisites Check
```bash
# Check Hailo device
ls /dev/hailo0 && echo "✓ Hailo device found" || echo "✗ Hailo device not found"

# Check Docker
docker --version && echo "✓ Docker installed" || echo "✗ Docker not installed"

# Check Python
python3.10 --version && echo "✓ Python 3.10 installed" || echo "✗ Python 3.10 not installed"
```

### Quick Deploy

1. **Clone and Configure**
```bash
# Clone repository
git clone https://github.com/your-org/wildfire-watch.git
cd wildfire-watch

# Download pre-converted Hailo model
mkdir -p converted_models/hailo_qat_output
wget https://your-storage/yolo8l_fire_640x640_hailo8l_qat.hef \
  -O converted_models/hailo_qat_output/yolo8l_fire_640x640_hailo8l_qat.hef

# Create minimal config
cat > .env << EOF
FRIGATE_DETECTOR=hailo8l
CAMERA_CREDENTIALS=username:password
CONSENSUS_THRESHOLD=2
EOF
```

2. **Start Services**
```bash
# Start with Hailo support
docker-compose up -d

# Verify services
docker-compose ps
```

3. **Test Fire Detection**
```bash
# Watch for detections
docker exec -it mqtt-broker mosquitto_sub -t "frigate/+/fire" -v
```

## Verify Installation

### Quick Test Script
```python
#!/usr/bin/env python3.10
# save as: quick_test.py

import os
import sys
from pathlib import Path

print("=== Hailo Quick Test ===\n")

# 1. Check device
if Path("/dev/hailo0").exists():
    print("✓ Hailo device found")
else:
    print("✗ Hailo device not found")
    sys.exit(1)

# 2. Check model
hef_path = "converted_models/hailo_qat_output/yolo8l_fire_640x640_hailo8l_qat.hef"
if Path(hef_path).exists():
    print("✓ HEF model found")
else:
    print("✗ HEF model not found")
    sys.exit(1)

# 3. Check Python package
try:
    import hailo_platform
    print("✓ Hailo Python package installed")
except ImportError:
    print("✗ Hailo Python package not installed")
    print("  Run: pip install hailort")
    sys.exit(1)

# 4. Basic inference test
try:
    from hailo_platform import VDevice
    device = VDevice()
    print("✓ Hailo device initialized")
    print("\n✅ All checks passed! Ready for fire detection.")
except Exception as e:
    print(f"✗ Initialization failed: {e}")
    sys.exit(1)
```

Run test:
```bash
python3.10 quick_test.py
```

## Common Operations

### Add IP Camera
```bash
# Add to environment
echo "CAMERA_IP_1=192.168.1.100" >> .env
echo "CAMERA_CREDENTIALS_1=admin:camera123" >> .env

# Restart camera detector
docker-compose restart camera-detector
```

### View Live Detections
```bash
# Terminal 1: Watch fire detections
docker exec -it mqtt-broker mosquitto_sub -t "frigate/+/fire" -v

# Terminal 2: Watch consensus decisions
docker exec -it mqtt-broker mosquitto_sub -t "trigger/fire_detected" -v

# Terminal 3: Monitor performance
docker exec -it mqtt-broker mosquitto_sub -t "telemetry/inference_metrics" -v
```

### Check System Health
```bash
# One-line health check
docker exec security-nvr curl -s localhost:5000/api/stats | \
  jq '.detectors.hailo8l | {fps, inference_speed, detection_fps}'
```

## Minimal Python Example

```python
#!/usr/bin/env python3.10
# fire_detect_demo.py

import cv2
import paho.mqtt.client as mqtt
from pathlib import Path

# MQTT callback
def on_fire_detected(client, userdata, message):
    print(f"🔥 FIRE DETECTED: {message.payload.decode()}")

# Setup MQTT
client = mqtt.Client()
client.on_message = on_fire_detected
client.connect("localhost", 1883, 60)
client.subscribe("frigate/+/fire")

# Start monitoring
print("Monitoring for fire detections... (Ctrl+C to stop)")
client.loop_forever()
```

## Performance Tuning (Optional)

### For Maximum Speed
```yaml
# Add to docker-compose.yml
environment:
  - HAILO_BATCH_SIZE=8
  - FRIGATE_DETECT_WIDTH=416
  - FRIGATE_DETECT_HEIGHT=416
```

### For Maximum Accuracy
```yaml
# Add to docker-compose.yml
environment:
  - HAILO_BATCH_SIZE=1
  - FRIGATE_DETECT_WIDTH=640
  - FRIGATE_DETECT_HEIGHT=640
  - MIN_CONFIDENCE=0.4
```

## Next Steps

1. **Add More Cameras**: See [Deployment Guide](hailo_deployment_guide.md)
2. **Tune Performance**: See [Validation Guide](hailo_model_validation_guide.md)
3. **Troubleshoot Issues**: See [Troubleshooting Guide](hailo_troubleshooting_guide.md)

## Quick Commands Reference

```bash
# Service Management
docker-compose up -d          # Start all services
docker-compose down           # Stop all services
docker-compose logs -f        # View all logs
docker-compose ps             # Check status

# Debugging
docker logs security-nvr      # Frigate logs
docker exec -it security-nvr bash  # Shell access
journalctl -u docker -f       # System logs

# Performance
htop                          # CPU/Memory usage
nvtop                         # GPU usage (if available)
iotop                         # Disk I/O
```

## FAQ

**Q: How do I know it's working?**
A: Check for MQTT messages on `frigate/+/fire` topic

**Q: What's the minimum hardware?**
A: Hailo-8L M.2, 8GB RAM, Ubuntu 20.04+

**Q: Can I use USB cameras?**
A: Yes, mount them in docker-compose.yml

**Q: How many cameras supported?**
A: 8-16 depending on resolution and FPS

**Q: Where are recordings stored?**
A: `/media/frigate/recordings/`