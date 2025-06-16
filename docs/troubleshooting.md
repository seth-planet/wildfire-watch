# Troubleshooting Guide

Common issues and solutions for Wildfire Watch deployments.

## Quick Diagnostics

Run the diagnostic script first:
```bash
./scripts/diagnose.sh

# Or manually check:
docker-compose ps
docker logs mqtt-broker --tail 50
mosquitto_sub -h localhost -t '#' -v
```

## Service Issues

### MQTT Broker

**Container won't start:**
```bash
# Check ports
sudo netstat -tlnp | grep 1883
# Kill conflicting process or change port in .env

# Check permissions
ls -la /var/lib/docker/volumes/wildfire_mqtt_data
# Fix: docker run --rm -v wildfire_mqtt_data:/data alpine chown -R 1883:1883 /data
```

**Connection refused:**
```bash
# Test local connection
mosquitto_pub -h localhost -t test -m "hello"

# Check firewall
sudo ufw status
sudo ufw allow 1883/tcp

# Verify broker config
docker exec mqtt-broker cat /mosquitto/config/mosquitto.conf
```

**Authentication failures:**
```bash
# Reset password
docker exec mqtt-broker mosquitto_passwd -c /mosquitto/config/passwd admin

# Check ACL
docker exec mqtt-broker cat /mosquitto/config/acl.conf

# Test with credentials
mosquitto_sub -h localhost -u admin -P password -t '#' -v
```

### Camera Detector

**No cameras discovered:**
```bash
# Check network connectivity
docker exec camera-detector ping -c 3 192.168.1.100

# Verify credentials
docker exec camera-detector env | grep CAMERA_CREDENTIALS

# Test ONVIF manually
docker exec camera-detector python -c "
from onvif import ONVIFCamera
cam = ONVIFCamera('192.168.1.100', 80, 'admin', 'password')
print(cam.devicemgmt.GetDeviceInformation())
"

# Force specific subnet scan
docker exec camera-detector python camera_detector.py --subnet 192.168.1.0/24
```

**Wrong camera URLs:**
```bash
# Check discovered cameras
mosquitto_sub -t 'cameras/discovered' -C 1 | jq

# Test RTSP URL
ffprobe rtsp://admin:password@192.168.1.100:554/stream

# Override stream path
CAMERA_STREAM_PATHS="camera1=/h264/ch1/main/av_stream,camera2=/live/0/MAIN"
```

### Frigate NVR

**High CPU usage:**
```bash
# Check detector type
docker exec security-nvr cat /config/config.yml | grep detector

# Verify hardware acceleration
docker exec security-nvr ls -la /dev/dri  # For Intel GPU
docker exec security-nvr lsusb | grep Google  # For Coral

# Reduce motion detection
# In frigate.yml:
# motion:
#   threshold: 30  # Increase from 25
#   contour_area: 200  # Increase from 100
```

**No recordings:**
```bash
# Check disk space
df -h /media/frigate

# Verify recording config
docker exec security-nvr cat /config/config.yml | grep -A5 record

# Check camera streams
docker exec security-nvr ffprobe rtsp://camera-url

# Test recording manually
docker exec security-nvr ffmpeg -i rtsp://camera-url -t 10 test.mp4
```

**Coral not detected:**
```bash
# Check USB connection
lsusb | grep 1a6e  # Should show Google Inc.

# Verify permissions
ls -la /dev/bus/usb/*/*

# Test Coral directly (requires Python 3.8)
docker run --rm --device /dev/bus/usb:/dev/bus/usb \
  --device /dev/dri:/dev/dri \
  python:3.8-slim bash -c "
pip install tflite-runtime
python -c 'import tflite_runtime.interpreter as tflite; print(\"Coral TPU support loaded\")'
"

# Important: Coral TPU requires Python 3.8 for tflite_runtime
# See docs/coral_python38_requirements.md for details
```

**Hailo not working:**
```bash
# Check device
ls -la /dev/hailo*

# Verify driver
dmesg | grep hailo

# Test Hailo
docker exec security-nvr hailortcli fw-control identify
```

### Fire Consensus

**Not triggering pumps:**
```bash
# Monitor fire events
mosquitto_sub -t 'frigate/fire/+' -v

# Check consensus state
mosquitto_sub -t 'consensus/status' -C 1 | jq

# Lower threshold temporarily
docker exec fire-consensus sh -c "echo 'CONSENSUS_THRESHOLD=1' >> /app/.env"
docker restart fire-consensus

# Verify camera overlap
docker exec fire-consensus python -c "
import json
with open('/app/camera_map.json') as f:
    print(json.dumps(json.load(f), indent=2))
"
```

**False alarms:**
```bash
# Increase threshold
CONSENSUS_THRESHOLD=3

# Require larger fires
MIN_FIRE_SIZE=20000

# Add delay
ACTIVATION_DELAY=10

# Check recent events
mosquitto_sub -t 'consensus/events' -C 1 | jq '.[-10:]'
```

### GPIO Trigger

**Pump not activating:**
```bash
# Check GPIO permissions
ls -la /dev/gpiomem

# Test GPIO manually
docker exec gpio-trigger gpio readall

# Test specific pin
docker exec gpio-trigger gpio write 17 1
sleep 2
docker exec gpio-trigger gpio write 17 0

# Monitor MQTT commands
mosquitto_sub -t 'gpio/pump/+' -v

# Check wiring
# Relay should click when activated
```

**Pump won't stop:**
```bash
# Emergency stop
mosquitto_pub -t 'gpio/emergency/stop' -m '1'

# Force GPIO low
docker exec gpio-trigger gpio write 17 0

# Check max runtime
docker exec gpio-trigger env | grep MAX_ENGINE_RUNTIME

# Hardware override
# Disconnect relay power
```

## Network Issues

### Camera Connection Problems

**RTSP timeouts:**
```bash
# Test connectivity
ping -c 3 camera-ip

# Check RTSP port
nmap -p 554 camera-ip

# Try alternate ports
ffprobe rtsp://user:pass@camera-ip:8554/stream

# Reduce resolution
# Camera web UI > Video > Substream > Lower resolution
```

**Network congestion:**
```bash
# Monitor bandwidth
iftop -i eth0

# Check dropped packets
netstat -i

# Implement QoS
tc qdisc add dev eth0 root handle 1: htb default 30
tc class add dev eth0 parent 1: classid 1:1 htb rate 100mbit
tc class add dev eth0 parent 1:1 classid 1:10 htb rate 50mbit ceil 80mbit
```

### VLAN Issues

**Cameras on different VLAN:**
```bash
# Add VLAN interface
sudo ip link add link eth0 name eth0.10 type vlan id 10
sudo ip addr add 192.168.10.1/24 dev eth0.10
sudo ip link set dev eth0.10 up

# Configure Docker
# In docker-compose.yml:
# networks:
#   camera_vlan:
#     driver: macvlan
#     driver_opts:
#       parent: eth0.10
```

## Hardware Issues

### AI Accelerator Problems

**Coral overheating:**
```bash
# Check temperature
cat /sys/class/thermal/thermal_zone*/temp

# Add cooling
# Physical heatsink or fan

# Reduce inference rate
# In frigate.yml:
# detect:
#   fps: 3  # Reduce from 5
```

**Hailo initialization errors:**
```bash
# Update firmware
sudo hailortcli fw-update --scan

# Reset device
sudo modprobe -r hailo
sudo modprobe hailo

# Check power
# Hailo needs stable 3.3V supply
```

### Raspberry Pi Issues

**Under-voltage warnings:**
```bash
# Check throttling
vcgencmd get_throttled
# 0x50005 = throttled due to under-voltage

# Solutions:
# - Use official 27W PSU
# - Reduce USB devices
# - Disable unnecessary services
```

**SD card corruption:**
```bash
# Check filesystem
sudo fsck -f /dev/mmcblk0p2

# Prevent corruption:
# - Use A2-rated cards
# - Enable overlayfs for /var/log
# - Move Docker to external SSD
```

## Performance Optimization

### Reduce CPU Usage

```bash
# Profile services
docker stats --no-stream

# Optimize detection
# Reduce camera resolution
# Lower FPS
# Use hardware acceleration

# Disable unnecessary features
RECORDING_ENABLED=false  # If only need detection
SNAPSHOT_ENABLED=false
```

### Memory Issues

```bash
# Check memory usage
free -h
docker system df

# Clean up
docker system prune -a
docker volume prune

# Limit container memory
# In docker-compose.yml:
# mem_limit: 2g
```

## Common Error Messages

### "No EdgeTPU detected"
- Check USB connection
- Verify udev rules: `/etc/udev/rules.d/99-coral.rules`
- Try different USB port
- Check power supply
- **Ensure Python 3.8 is used** - Coral TPU requires Python 3.8 for tflite_runtime
  - See [Coral Python 3.8 Requirements](coral_python38_requirements.md)

### "Failed to connect to MQTT broker"
- Verify broker is running
- Check network connectivity
- Confirm credentials
- Review firewall rules

### "Camera stream timeout"
- Reduce camera resolution
- Check network bandwidth
- Verify RTSP URL
- Update camera firmware

### "Consensus timeout"
- Increase `CONSENSUS_TIMEOUT`
- Check time sync between nodes
- Verify MQTT bridging
- Review camera overlap

## Diagnostic Commands

### System Health
```bash
# Overall status
docker-compose ps
docker-compose logs --tail 100

# Service health
curl http://localhost:5000/api/stats  # Frigate
mosquitto_sub -t '$SYS/#' -v  # MQTT

# Resource usage
htop
iotop
nethogs
```

### Data Flow Test
```bash
# Simulate fire detection
mosquitto_pub -t 'frigate/fire/test_camera' -m '{
  "camera": "test_camera",
  "label": "fire",
  "score": 0.95,
  "box": [100, 100, 200, 200]
}'

# Watch consensus
mosquitto_sub -t 'consensus/+' -v

# Monitor pump commands
mosquitto_sub -t 'gpio/pump/+' -v
```

## Getting Help

### Collect Debug Info
```bash
./scripts/collect_debug.sh

# Creates debug bundle with:
# - Service logs
# - Configuration files
# - System information
# - Network status
```

### Debug Mode
```bash
# Enable debug logging
LOG_LEVEL=DEBUG docker-compose up

# Service-specific debug
docker exec camera-detector python -m pdb detect.py
docker exec fire-consensus python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
```

### Community Support
- GitHub Issues: Include debug bundle
- Discord: Real-time help
- Forum: Detailed discussions

## Recovery Procedures

### Full System Reset
```bash
# Stop all services
docker-compose down

# Clear data (preserves recordings)
docker volume rm wildfire_mqtt_data wildfire_consensus_data

# Regenerate certificates
./scripts/generate_certs.sh custom

# Restart
docker-compose up -d
```

### Restore from Backup
```bash
# Stop services
docker-compose down

# Restore volumes
docker run --rm -v wildfire_mqtt_data:/data -v /backup:/backup alpine tar xzf /backup/mqtt_data.tar.gz

# Restart
docker-compose up -d
```

### Emergency Bypass
```bash
# Direct pump control (bypasses consensus)
docker exec gpio-trigger python -c "
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.OUT)
GPIO.output(17, GPIO.HIGH)
"
```
