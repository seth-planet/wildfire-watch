# 🍓 Raspberry Pi 5 Quick Start Guide

## Overview

This guide will help you deploy Wildfire Watch on a Raspberry Pi 5 with Hailo-8L AI accelerator for efficient wildfire detection.

## Hardware Requirements

- **Raspberry Pi 5** (4GB or 8GB RAM)
- **Hailo-8L M.2 AI Kit** or Coral USB Accelerator
- **MicroSD Card** (32GB minimum, 64GB recommended)
- **USB Storage** (1TB+ for 6 months of recordings)
- **8-Channel Relay Board** for pump control
- **IP Cameras** (ONVIF compatible)
- **Power Supply** (27W USB-C PD for Pi 5)

## Software Setup

### 4. Install Coral Runtime (if using Coral USB)

```bash
# Add Coral repository
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

# Install Edge TPU runtime
sudo apt update
sudo apt install -y libedgetpu1-std

# Verify installation
ls /dev/bus/usb/
# Should show Coral device when plugged in
```

## Deployment

### 1. Clone Wildfire Watch

```bash
# Clone repository
git clone https://github.com/your-org/wildfire-watch.git
cd wildfire-watch

# Create .env file
cp .env.example .env
```

### 2. Configure Environment

Edit `.env` file:

```bash
# Platform
PLATFORM=linux/arm64

# MQTT Security (set to true for production)
MQTT_TLS=false

# Camera settings
CAMERA_CREDENTIALS=admin:yourpassword,username:password
DISCOVERY_INTERVAL=300

# Hardware acceleration
FRIGATE_DETECTOR=hailo  # or 'coral' if using Coral
FRIGATE_HARDWARE=hailo  # or 'coral'

# Storage
USB_MOUNT_PATH=/media/frigate
RECORD_RETAIN_DAYS=180

# GPIO Pins for pump control
MAIN_VALVE_PIN=18
IGNITION_START_PIN=23
IGNITION_ON_PIN=24
IGNITION_OFF_PIN=25
REFILL_VALVE_PIN=22
PRIMING_VALVE_PIN=26
RPM_REDUCE_PIN=27

# Optional safety sensors
RESERVOIR_FLOAT_PIN=16
LINE_PRESSURE_PIN=20

# Pump timing (adjust for your setup!)
MAX_ENGINE_RUNTIME=1800  # 30 minutes - MUST be less than tank capacity/flow rate
REFILL_MULTIPLIER=40
```

### 3. Prepare USB Storage

```bash
# List USB drives
lsblk

# Format USB drive (WARNING: erases all data!)
sudo mkfs.ext4 -L frigate-storage /dev/sda1

# Create mount point
sudo mkdir -p /media/frigate

# Mount drive
sudo mount /dev/sda1 /media/frigate

# Make persistent
echo "/dev/sda1 /media/frigate ext4 defaults 0 0" | sudo tee -a /etc/fstab

# Set permissions
sudo chown -R $USER:docker /media/frigate
```

### 4. Wire GPIO Connections

```
Raspberry Pi 5          8-Channel Relay Board
--------------          --------------------
GPIO 18 (Pin 12) ----> IN1 (Main Valve)
GPIO 23 (Pin 16) ----> IN2 (Ignition Start)
GPIO 24 (Pin 18) ----> IN3 (Ignition On)
GPIO 25 (Pin 22) ----> IN4 (Ignition Off)
GPIO 22 (Pin 15) ----> IN5 (Refill Valve)
GPIO 26 (Pin 37) ----> IN6 (Priming Valve)
GPIO 27 (Pin 13) ----> IN7 (RPM Reduce)
GND (Pin 6)      ----> GND

Optional Sensors:
GPIO 16 (Pin 36) <---- Reservoir Float Switch
GPIO 20 (Pin 38) <---- Line Pressure Switch
```

### 5. Security Setup

#### For Development/Testing
```bash
# Default certificates are included - INSECURE!
# Skip to deployment step
```

#### For Production
```bash
# Generate custom certificates
./scripts/generate_certs.sh custom
# Follow prompts for CA password and certificate details

# Enable TLS in .env
sed -i 's/MQTT_TLS=false/MQTT_TLS=true/' .env
```

### 6. Deploy Services

```bash
# Deploy with Docker Compose
docker-compose up -d

# Verify all services are running
docker ps
# Should show: mqtt_broker, camera-detector, security_nvr, fire_consensus, gpio_trigger

# View logs
docker-compose logs -f

# Get Frigate credentials
docker logs security_nvr | grep "Password:"
```

## Verification

### 1. Check Hardware Detection

```bash
# Verify Hailo detection
docker exec frigate cat /proc/device-tree/model
docker exec frigate ls /dev/hailo*

# Or verify Coral detection
docker exec frigate ls /dev/bus/usb/
```

### 2. Monitor Camera Discovery

```bash
# Watch camera detector logs
docker logs camera_detector -f

# Should see:
# [INFO] ONVIF camera found: Hikvision DS-2CD2042WD at 192.168.1.100
# [INFO] Updated Frigate config with 2 cameras
```

### 3. Test Fire Detection

```bash
# Simulate fire detection
mosquitto_pub -h localhost -t 'fire/detection' -m '{
  "camera_id": "test_cam",
  "confidence": 0.8,
  "bounding_box": [0.1, 0.1, 0.05, 0.05]
}'

# Monitor consensus
docker logs fire_consensus -f
```

### 4. Test GPIO (Safely!)

```bash
# Test with engine disconnected!
docker exec gpio_trigger python -c "
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)
GPIO.output(18, GPIO.HIGH)
print('Main valve opened')
import time; time.sleep(2)
GPIO.output(18, GPIO.LOW)
print('Main valve closed')
GPIO.cleanup()
"
```

## Performance Optimization

### 1. GPU Memory Split

```bash
# Reduce GPU memory (we're using Hailo/Coral)
sudo raspi-config
# Advanced Options > Memory Split > 16
```

### 2. Overclock (Optional)

```bash
# Edit config
sudo nano /boot/firmware/config.txt

# Add (for active cooling only):
over_voltage=4
arm_freq=2800
gpu_freq=800
```

### 3. Optimize Docker

```bash
# Limit logging
cat > /etc/docker/daemon.json <<EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
EOF

sudo systemctl restart docker
```

## Troubleshooting

### Problem: Hailo Not Detected

```bash
# Check PCIe connection
lspci | grep Hailo

# Check kernel module
lsmod | grep hailo

# Reinstall driver
sudo apt reinstall hailo-rt
```

### Problem: Camera Detection Fails

```bash
# Check network
ip addr show

# Test camera directly
curl -u username:password http://camera-ip/onvif/device_service

# Check firewall
sudo iptables -L
```

### Problem: GPIO Permission Denied

```bash
# Add user to gpio group
sudo usermod -aG gpio $USER

# Logout and login again
```

### Problem: Low Performance

```bash
# Check temperature
vcgencmd measure_temp

# Check throttling
vcgencmd get_throttled

# Add cooling if needed
```

## Maintenance

### Weekly Tasks

1. Check storage space: `df -h /media/frigate`
2. Verify all cameras online: http://pi-ip:5000
3. Test pump sequence (dry run)
4. Check system logs: `sudo journalctl -u docker`

### Monthly Tasks

1. Update system: `sudo apt update && sudo apt upgrade`
2. Clean old recordings if needed
3. Test full system with water
4. Backup configuration

### Annual Tasks

1. Replace pump engine oil
2. Test all valves and sensors
3. Update Wildfire Watch: `git pull && docker-compose up -d --build`
4. Review and rotate certificates

## Advanced Configuration

### Multi-Camera Optimization

```yaml
# In frigate config
cameras:
  front_yard:
    detect:
      width: 640
      height: 480
      fps: 5  # Lower FPS for Pi 5
    ffmpeg:
      hwaccel_args:
        - -c:v
        - h264_v4l2m2m  # Hardware decode
```

### Power Management

```bash
# For battery/solar systems
# Create power profile
cat > power_save.sh <<EOF
#!/bin/bash
# Reduce CPU frequency
echo powersave | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
# Reduce detection FPS
docker exec frigate sed -i 's/fps: 5/fps: 2/' /config/config.yml
docker restart frigate
EOF
```

### Remote Access

```bash
# Install Tailscale for secure remote access
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up

# Access from anywhere
http://pi-hostname:5000  # Frigate UI
ssh pi@pi-hostname       # SSH access
```

## Security Notes

⚠️ **Remember to**:
1. Replace default certificates before production
2. Change default camera passwords
3. Secure physical access to Pi
4. Enable firewall for internet-exposed systems
5. Regularly update all software

## Performance Expectations

With Raspberry Pi 5 + Hailo-8L:
- **Inference Speed**: 20-30ms per frame
- **Cameras Supported**: 4-6 simultaneous
- **Recording**: 1080p continuous
- **Power Usage**: ~10W average
- **Temperature**: 50-60°C with cooling

## Next Steps

1. **Train Custom Model**: See [Model Converter](../converted_models/README.md)
2. **Multi-Node Setup**: Add more Pi units for larger properties
3. **Cloud Integration**: Enable MQTT bridge for remote monitoring
4. **Home Assistant**: Integrate for automation

## Getting Help

- Check service logs: `docker logs [service_name]`
- Enable debug mode: `LOG_LEVEL=DEBUG`
- GPIO testing: Disconnect pump first!
- Join community forums for Pi-specific help

Remember: Test thoroughly with pump disconnected before connecting to actual fire suppression system!1. Install Raspberry Pi OS

```bash
# Download Raspberry Pi Imager
# Select: Raspberry Pi OS (64-bit) 
# Configure: hostname, WiFi, SSH
```

### 2. Initial System Configuration

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo apt install -y docker-compose

# Enable required interfaces
sudo raspi-config
# Enable: I2C, SPI, Camera, GPIO
```

### 3. Install Hailo Runtime (if using Hailo-8L)

```bash
# Add Hailo repository
wget -qO - https://hailo.ai/keys/hailo-rpi5-public.key | sudo apt-key add -
echo "deb https://hailo.ai/raspberrypi/dists/rpios-bookworm/main binary/" | sudo tee /etc/apt/sources.list.d/hailo.list

# Install Hailo RT
sudo apt update
sudo apt install -y hailo-rt

# Verify installation
hailortcli scan
# Should show: Hailo-8L device
```

###
