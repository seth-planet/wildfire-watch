# 🖥️ Linux PC Quick Start Guide

## Overview

This guide will help you deploy Wildfire Watch on a Linux PC (Ubuntu/Debian) with support for various AI accelerators (Coral, Hailo, NVIDIA GPU).

## Hardware Requirements

- **x86_64 PC** with Ubuntu 20.04+ or Debian 11+
- **8GB+ RAM** (16GB recommended)
- **AI Accelerator** (one or more):
  - Google Coral M.2/PCIe/USB
  - Hailo-8 PCIe
  - NVIDIA GPU (GTX 1050+)
  - Intel with QuickSync (for decode)
- **Storage**: 1TB+ for recordings
- **Network**: Gigabit Ethernet recommended
- **USB-to-GPIO adapter** (for pump control)

## Software Setup

### 1. Install Base System

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y \
    curl \
    git \
    build-essential \
    python3-pip \
    net-tools \
    v4l-utils \
    vainfo

# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo apt install -y docker-compose

# Logout and login for group changes
```

### 2. Hardware-Specific Setup

#### For Coral M.2/PCIe

```bash
# Add Coral repository
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

# Install PCIe driver and runtime
sudo apt update
sudo apt install -y gasket-dkms libedgetpu1-std

# Add udev rules
sudo sh -c "echo 'SUBSYSTEM==\"apex\", MODE=\"0660\", GROUP=\"docker\"' > /etc/udev/rules.d/65-apex.rules"
sudo udevadm control --reload-rules && sudo udevadm trigger

# Verify installation
ls /dev/apex_0
lspci | grep Coral
```

#### For Coral USB

```bash
# Install runtime
sudo apt install -y libedgetpu1-std

# Add udev rules for USB
sudo sh -c "echo 'SUBSYSTEM==\"usb\", ATTRS{idVendor}==\"1a6e\", GROUP=\"docker\"' > /etc/udev/rules.d/65-coral-usb.rules"
sudo sh -c "echo 'SUBSYSTEM==\"usb\", ATTRS{idVendor}==\"18d1\", GROUP=\"docker\"' >> /etc/udev/rules.d/65-coral-usb.rules"
sudo udevadm control --reload-rules && sudo udevadm trigger

# Verify (plug in Coral USB)
lsusb | grep -E "(1a6e|18d1)"
```

#### For Hailo-8

```bash
# Download Hailo driver from https://hailo.ai/developer-zone/
# Install HailoRT
sudo dpkg -i hailort_*_amd64.deb

# Load driver
sudo modprobe hailo_pci

# Verify installation
hailortcli scan
```

#### For NVIDIA GPU

```bash
# Install NVIDIA drivers
sudo apt install -y nvidia-driver-525

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update
sudo apt install -y nvidia-container-toolkit

# Configure Docker for NVIDIA
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```

#### For Intel QuickSync

```bash
# Install Intel Media Driver
sudo apt install -y intel-media-va-driver-non-free

# Verify
vainfo
# Should show VAProfileH264Main, VAProfileHEVCMain, etc.
```

### 3. USB-to-GPIO Setup (for pump control)

```bash
# Option 1: USB-to-GPIO adapter (FT232H)
sudo apt install -y python3-libftdi1
sudo pip3 install pyftdi

# Add udev rules
sudo sh -c "echo 'SUBSYSTEM==\"usb\", ATTR{idVendor}==\"0403\", ATTR{idProduct}==\"6014\", GROUP=\"gpio\", MODE=\"0660\"' > /etc/udev/rules.d/99-ftdi.rules"
sudo udevadm control --reload-rules && sudo udevadm trigger

# Option 2: Arduino as GPIO
sudo apt install -y arduino
sudo usermod -aG dialout $USER
# Upload Firmata sketch to Arduino
```

## Deployment

### 1. Clone Wildfire Watch

```bash
# Clone repository
git clone https://github.com/seth-planet/wildfire-watch.git
cd wildfire-watch

# Create .env file
cp .env.example .env
```

### 2. Configure Environment

Edit `.env` file:

```bash
# Platform
PLATFORM=linux/amd64

# Network
NETWORK_SUBNET=192.168.100.0/24
MQTT_BROKER_STATIC_IP=192.168.100.10

# Camera settings
CAMERA_CREDENTIALS=admin:yourpassword,admin:camera123
DISCOVERY_INTERVAL=300

# Hardware acceleration (auto-detect or specify)
FRIGATE_DETECTOR=auto  # auto|coral|hailo|gpu|cpu
FRIGATE_HARDWARE=auto  # auto|coral|hailo|nvidia|intel

# Storage
USB_MOUNT_PATH=/media/frigate
RECORD_RETAIN_DAYS=180

# GPIO adapter type
GPIO_TYPE=ft232h  # ft232h|arduino|none

# GPIO Pins (adjust for your adapter)
MAIN_VALVE_PIN=0
IGNITION_START_PIN=1
IGNITION_ON_PIN=2
IGNITION_OFF_PIN=3
REFILL_VALVE_PIN=4
PRIMING_VALVE_PIN=5
RPM_REDUCE_PIN=6

# Optional safety sensors
RESERVOIR_FLOAT_PIN=7
LINE_PRESSURE_PIN=8

# Pump timing (adjust for your setup!)
MAX_ENGINE_RUNTIME=1800  # 30 minutes - MUST be less than tank capacity/flow rate
REFILL_MULTIPLIER=40
```

### 3. Prepare Storage

```bash
# Create mount point
sudo mkdir -p /media/frigate

# Option 1: Use separate drive
# List drives
lsblk
# Mount drive
sudo mount /dev/sdb1 /media/frigate
# Add to fstab for persistence
echo "UUID=$(sudo blkid -s UUID -o value /dev/sdb1) /media/frigate ext4 defaults 0 0" | sudo tee -a /etc/fstab

# Option 2: Use directory on main drive
sudo mkdir -p /media/frigate
sudo chown $USER:docker /media/frigate
```

### 4. Deploy with Docker Compose

```bash
# Use local compose file for development
docker-compose -f docker-compose.local.yml up -d

# View logs
docker-compose logs -f

# Check service status
docker ps
```

### 5. Generate Secure Certificates

```bash
# Generate custom certificates
./scripts/generate_certs.sh custom

# Deploy certificates
sudo mkdir -p /mnt/data/certs
sudo cp certs/* /mnt/data/certs/

# Restart services
docker-compose restart
```

## Verification

### 1. Check Hardware Detection

```bash
# Check detected hardware
docker exec security-nvr check-hardware

# Should show:
# Detected Hardware:
# - Coral PCIe Accelerator
# - NVIDIA GeForce RTX 3060
# - Intel QuickSync
```

### 2. Monitor System

```bash
# View Frigate UI
firefox http://localhost:5000

# Monitor MQTT traffic
docker exec mqtt-broker mosquitto_sub -t '#' -v

# Check camera discovery
docker logs camera-detector -f
```

### 3. Test Detection

```bash
# Run detection benchmark
docker exec security-nvr benchmark-detector

# Should show:
# Coral: 15ms average
# GPU: 12ms average
# CPU: 150ms average
```

### 4. Test GPIO Adapter

```bash
# For FT232H
python3 -c "
from pyftdi.gpio import GpioController
gpio = GpioController()
gpio.open_from_url('ftdi://ftdi:232h/1')
gpio.set_direction(0xFF, 0xFF)  # All outputs
gpio.write(0x01)  # Turn on pin 0
import time; time.sleep(1)
gpio.write(0x00)  # Turn off
"

# For Arduino
python3 -c "
import serial
import time
ser = serial.Serial('/dev/ttyUSB0', 9600)
ser.write(b'1')  # Turn on
time.sleep(1)
ser.write(b'0')  # Turn off
"
```

## Performance Optimization

### 1. CPU Governor

```bash
# Set performance mode
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Make persistent
sudo apt install -y cpufrequtils
echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils
sudo systemctl restart cpufrequtils
```

### 2. Docker Optimization

```bash
# Configure Docker daemon
sudo tee /etc/docker/daemon.json <<EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
EOF

sudo systemctl restart docker
```

### 3. Network Optimization

```bash
# Increase network buffers
sudo sysctl -w net.core.rmem_max=26214400
sudo sysctl -w net.core.rmem_default=26214400
sudo sysctl -w net.core.wmem_max=26214400
sudo sysctl -w net.core.wmem_default=26214400

# Make persistent
echo "net.core.rmem_max=26214400" | sudo tee -a /etc/sysctl.conf
echo "net.core.rmem_default=26214400" | sudo tee -a /etc/sysctl.conf
```

## Multi-Accelerator Configuration

### Using Multiple AI Accelerators

```yaml
# In frigate config
detectors:
  coral:
    type: edgetpu
    device: pci
  gpu:
    type: tensorrt
    device: 0
  
cameras:
  high_risk_area:
    detect:
      enabled: true
    ffmpeg:
      hwaccel_args:
        - -hwaccel
        - cuda
        - -hwaccel_output_format
        - cuda
  
  general_area:
    detect:
      enabled: true
    ffmpeg:
      hwaccel_args:
        - -hwaccel
        - vaapi
        - -hwaccel_device
        - /dev/dri/renderD128
```

## Troubleshooting

### Problem: Coral Not Detected

```bash
# Check driver
lsmod | grep apex

# Reinstall driver
sudo apt reinstall gasket-dkms

# Check permissions
ls -la /dev/apex_0
```

### Problem: NVIDIA GPU Not Available

```bash
# Check driver
nvidia-smi

# Check Docker runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi

# Reinstall container toolkit
sudo apt reinstall nvidia-container-toolkit
```

### Problem: High CPU Usage

```bash
# Check hardware acceleration
docker exec frigate ffmpeg -hwaccels

# Monitor with htop
htop

# Check inference device
docker logs frigate | grep "Inference"
```

## Advanced Configuration

### Remote Management

```bash
# Install Portainer
docker volume create portainer_data
docker run -d -p 8000:8000 -p 9443:9443 --name portainer \
  --restart=always \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v portainer_data:/data \
  portainer/portainer-ce:latest

# Access at https://localhost:9443
```

### Monitoring Stack

```yaml
# Add to docker-compose.yml
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
  
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

### High Availability

```bash
# For mission-critical deployments
# Use Docker Swarm or Kubernetes
docker swarm init
docker stack deploy -c docker-compose.yml wildfire-watch
```

## Performance Expectations

With modern x86 PC + Coral + GPU:
- **Cameras**: 16+ simultaneous streams
- **Inference**: 10-20ms per frame
- **Recording**: 4K resolution capable
- **Power**: 50-150W depending on load
- **Storage**: 1TB handles ~30 days of 16 cameras

## Security Hardening

```bash
# Enable firewall
sudo ufw enable
sudo ufw allow 22/tcp     # SSH
sudo ufw allow 5000/tcp   # Frigate UI (local only)
sudo ufw allow 1883/tcp   # MQTT (local only)

# Restrict Docker exposed ports
# Edit docker-compose.yml:
# Change "5000:5000" to "127.0.0.1:5000:5000"

# Enable AppArmor
sudo apt install -y apparmor apparmor-utils
sudo aa-enforce /etc/apparmor.d/docker
```

## Next Steps

1. **Multi-Node Setup**: Scale to multiple PCs
2. **Cloud Backup**: Enable recording sync to cloud
3. **Custom Models**: Train for specific threats
4. **Integration**: Connect to existing security systems

## Getting Help

- Hardware detection: `docker exec security-nvr check-hardware`
- Performance issues: Check `htop` and `nvidia-smi`
- Network problems: `docker network ls`
- Storage issues: `df -h /media/frigate`

Remember: Test thoroughly in a safe environment before deploying in production!
