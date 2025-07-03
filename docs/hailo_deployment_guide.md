# Hailo-8L Deployment Guide for Wildfire Watch

## Prerequisites

### Hardware Requirements
- Hailo-8L M.2 AI accelerator (13 TOPS)
- M.2 Key M slot (PCIe interface)
- Ubuntu 20.04+ or compatible Linux distribution
- Minimum 8GB RAM
- 50GB free disk space

### Software Requirements
- Docker and Docker Compose
- Python 3.10
- HailoRT 4.21.0 drivers
- MQTT broker (included in docker-compose)

## Installation Steps

### 1. Install Hailo Drivers

```bash
# Download HailoRT
wget https://hailo.ai/downloads/hailort-4.21.0-linux-x86_64.tar.gz
tar -xzf hailort-4.21.0-linux-x86_64.tar.gz

# Install PCIe driver
cd hailort-4.21.0/drivers
sudo ./install_pcie_driver.sh

# Verify installation
lspci | grep Hailo
# Should show: Hailo Technologies Ltd. Hailo-8 AI Processor

# Check device
ls -la /dev/hailo*
# Should show: /dev/hailo0
```

### 2. Install HailoRT Python Package

```bash
# Create Python 3.10 environment
python3.10 -m venv hailo_env
source hailo_env/bin/activate

# Install HailoRT Python package
pip install hailort-4.21.0-cp310-cp310-linux_x86_64.whl
```

### 3. Deploy Wildfire Watch with Hailo

```bash
# Clone repository
git clone https://github.com/your-org/wildfire-watch.git
cd wildfire-watch

# Copy Hailo model
mkdir -p converted_models/hailo_qat_output
# Copy your yolo8l_fire_640x640_hailo8l_qat.hef file here

# Set environment variables
cat > .env << EOF
# Hailo Configuration
FRIGATE_DETECTOR=hailo8l
HAILO_DEVICE=/dev/hailo0

# Camera Configuration
CAMERA_CREDENTIALS=username:password
CONSENSUS_THRESHOLD=2
MIN_CONFIDENCE=0.5

# MQTT Configuration
MQTT_BROKER=mqtt-broker
MQTT_PORT=1883
EOF

# Start services
docker-compose up -d
```

## Configuration

### Frigate Configuration
The system automatically configures Frigate to use Hailo. Key settings:

```yaml
detectors:
  hailo8l:
    type: hailo8l
    device: PCIe
    
cameras:
  camera_name:
    detect:
      width: 640
      height: 640
      fps: 10
    objects:
      track:
        - fire
        - smoke
      filters:
        fire:
          min_score: 0.5
          threshold: 0.6
```

### Performance Tuning

```bash
# Monitor Hailo temperature
watch -n 1 'cat /sys/class/hwmon/hwmon*/temp1_input'

# Check Hailo utilization
hailortcli monitor

# Adjust batch size for throughput
# In docker-compose.yml:
HAILO_BATCH_SIZE=8  # Default: 1
```

## Monitoring and Troubleshooting

### Check Service Status

```bash
# View all services
docker-compose ps

# Check Frigate logs
docker-compose logs -f security-nvr

# Monitor MQTT messages
docker exec -it mqtt-broker mosquitto_sub -t "#" -v
```

### Verify Fire Detection

```bash
# Watch for fire detections
docker exec -it mqtt-broker mosquitto_sub -t "frigate/+/fire" -v

# Check consensus decisions
docker exec -it mqtt-broker mosquitto_sub -t "trigger/fire_detected" -v
```

### Common Issues

1. **Device Not Found**
   ```bash
   # Check permissions
   sudo chmod 666 /dev/hailo0
   
   # Add user to video group
   sudo usermod -a -G video $USER
   ```

2. **Low FPS**
   ```bash
   # Reduce camera resolution
   # In camera config: width: 416, height: 416
   
   # Increase batch size
   HAILO_BATCH_SIZE=16
   ```

3. **High Temperature**
   ```bash
   # Add cooling or reduce FPS
   # Target temperature: <85°C
   ```

## Performance Metrics

### Expected Performance
- Inference latency: 10-25ms
- Throughput: 40-100 FPS (depending on batch size)
- Power consumption: ~5W
- Temperature: 65-80°C under load

### Monitoring Commands
```bash
# Real-time performance
docker exec -it security-nvr python3 -c "
import frigate
frigate.stats.print_detector_metrics('hailo8l')
"

# Historical metrics
docker exec -it mqtt-broker mosquitto_sub -t "telemetry/inference_metrics" -v
```

## Backup and Recovery

### Backup Configuration
```bash
# Backup Hailo model
tar -czf hailo_model_backup.tar.gz converted_models/hailo_qat_output/

# Backup Frigate config
docker cp security-nvr:/config ./frigate_config_backup

# Backup environment
cp .env .env.backup
```

### Restore Process
```bash
# Stop services
docker-compose down

# Restore files
tar -xzf hailo_model_backup.tar.gz
cp .env.backup .env

# Restart services
docker-compose up -d
```

## Security Considerations

1. **Network Isolation**: Hailo device only accessible within Docker network
2. **MQTT Authentication**: Enable in production
   ```bash
   MQTT_USERNAME=wildfire
   MQTT_PASSWORD=secure_password
   MQTT_TLS=true
   ```
3. **Camera Credentials**: Use strong passwords and rotate regularly

## Maintenance

### Regular Tasks
- Monitor temperature daily
- Check inference latency weekly
- Update Hailo drivers quarterly
- Backup configuration monthly

### Log Rotation
```bash
# Add to crontab
0 2 * * * docker exec security-nvr find /media/frigate/logs -mtime +7 -delete
```

## Support Resources

- Hailo Developer Zone: https://hailo.ai/developer-zone/
- HailoRT Documentation: https://hailo.ai/developer-zone/documentation/
- Wildfire Watch Issues: https://github.com/your-org/wildfire-watch/issues
- Community Forum: https://community.hailo.ai/