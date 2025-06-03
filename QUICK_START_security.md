# 🚀 Quick Start Guide - Wildfire Watch

## ⚡ 5-Minute Deployment

### 1. Clone and Deploy

```bash
# Clone the repository
git clone https://github.com/your-org/wildfire-watch.git
cd wildfire-watch

# Deploy with Balena
balena push my-wildfire-app

# Or use Docker Compose
docker-compose up -d
```

### 2. System Starts Automatically

The system will:
- ✅ Start MQTT broker with default certificates
- ✅ Begin scanning for cameras
- ✅ Configure Frigate NVR for fire detection
- ✅ Monitor for consensus events
- ✅ Control GPIO pins when fire detected
- ✅ Record security footage to USB storage

### 3. Access the System

- **Frigate Web UI**: Browse to `http://device-ip:5000`
- **MQTT Explorer**: Connect to `device-ip:1883` or `device-ip:8883`
- **Logs**: `balena logs` or `docker-compose logs`

## 📋 System Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  IP Cameras │────▶│   Camera    │────▶│  Security   │
│   (RTSP)    │     │  Detector   │     │ NVR Service │
└─────────────┘     └─────────────┘     │  (Frigate)  │
                            │           └─────────────┘
                            │                    │
                            ▼                    ▼
                    ┌─────────────┐     ┌─────────────┐
                    │    MQTT     │◀────│    Fire     │
                    │   Broker    │     │  Consensus  │
                    └─────────────┘     └─────────────┘
                            │                    │
                            ▼                    ▼
                    ┌─────────────┐     ┌─────────────┐
                    │    GPIO     │     │    Pump     │
                    │  Trigger    │────▶│   System    │
                    └─────────────┘     └─────────────┘
```

### Service Descriptions

1. **Camera Detector**: Automatically discovers IP cameras on your network
2. **Security NVR (Frigate)**: AI-powered object detection and recording
   - Automatically detects available hardware acceleration (Hailo, Coral, GPU)
   - Only records when objects are detected (fire, smoke, person, car, wildlife)
   - Integrates with camera_detector for dynamic configuration
3. **Fire Consensus**: Multi-camera validation to prevent false alarms
4. **GPIO Trigger**: Controls pump, valves, and monitors system health
5. **MQTT Broker**: Secure communication hub for all services

## ⚠️ CRITICAL SECURITY WARNING ⚠️

**This quick deployment uses INSECURE default certificates!**

```
🔓 The private keys are PUBLIC
🔓 Anyone can decrypt your traffic  
🔓 Anyone can impersonate your services
🔓 DO NOT USE IN PRODUCTION
```

## 🔒 Securing Your Deployment

### Before Going Live

**Time Required**: 15 minutes

1. **Generate Secure Certificates**
   ```bash
   ./scripts/generate_certs.sh custom
   
   # You'll be asked for:
   # - CA password (use 20+ characters)
   # - Organization name
   # - Server hostnames
   ```

2. **Deploy to All Devices**
   ```bash
   # Option A: Automatic
   ./scripts/provision_certs.sh auto device1.local device2.local
   
   # Option B: Manual
   scp -r certs/* root@device:/mnt/data/certs/
   ```

3. **Restart Services**
   ```bash
   # Balena
   balena restart --all
   
   # Docker Compose
   docker-compose restart
   ```

4. **Verify Security**
   ```bash
   # Check for warning in logs
   docker logs mqtt_broker | grep WARNING
   # Should NOT see: "DEFAULT INSECURE CERTIFICATES"
   ```

## 📋 Deployment Checklist

### For Testing/Development ✅
- [x] Default certificates are fine
- [x] No internet exposure
- [x] Local network only
- [x] Short-term use

### For Production 🚨
- [ ] Generate custom certificates
- [ ] Deploy custom certificates
- [ ] Enable MQTT authentication
- [ ] Configure firewall rules
- [ ] Test failover scenarios
- [ ] Document your setup
- [ ] Plan certificate rotation

## 🎯 Configuration Overview

### Essential Settings

```yaml
# Camera Detection
CAMERA_CREDENTIALS: "admin:,admin:admin,admin:12345"  # Add your camera passwords

# Fire Detection  
CONSENSUS_THRESHOLD: "2"      # How many cameras must agree
MIN_CONFIDENCE: "0.7"         # AI confidence threshold

# Pump Control
MAX_ENGINE_RUNTIME: "1800"    # 30-minute safety limit
FIRE_OFF_DELAY: "1800"        # Run 30 min after fire gone
REFILL_MULTIPLIER: "40"       # Refill time = runtime × 40

# Optional Safety Sensors
RESERVOIR_FLOAT_PIN: "16"     # GPIO pin for tank level sensor
LINE_PRESSURE_PIN: "20"       # GPIO pin for pressure switch

# Security NVR
RECORD_RETAIN_DAYS: "180"     # Keep 6 months of recordings
USB_MOUNT_PATH: "/media/frigate"  # USB storage location
FRIGATE_HARDWARE: "auto"      # auto-detects Hailo, Coral, or GPU
```

### Network Architecture

```
[IP Cameras] ←→ [Camera Detector] ←→ [Security NVR/Frigate]
                                            ↓
                                    [MQTT Broker] ←→ [Fire Consensus]
                                            ↓
                                    [GPIO Trigger/Pump]
```

## 🔍 Verifying Your System

### 1. Check Service Health

```bash
# View all services
docker ps

# Check MQTT broker
mosquitto_sub -h localhost -t 'system/+/health' -v

# Monitor fire detections
mosquitto_sub -h localhost -t 'fire/#' -v
```

### 2. Camera Discovery

Watch cameras being found:
```bash
docker logs camera_detector -f

# You should see:
# [INFO] ONVIF camera found: Hikvision DS-2CD2042WD at 192.168.1.100
# [INFO] Updated Frigate config with 2 cameras
```

### 3. Frigate NVR Status

Access Frigate UI at `http://device-ip:5000` to:
- View live camera feeds
- Check object detection
- Review recorded events
- Monitor storage usage

### 4. Test Fire Detection

```bash
# Simulate a fire detection
mosquitto_pub -t 'fire/detection' -m '{
  "camera_id": "test_cam1",
  "confidence": 0.8,
  "bounding_box": [0.1, 0.1, 0.05, 0.05]
}'

# Watch consensus decision
docker logs fire_consensus -f
```

## 🚫 Common Mistakes to Avoid

### 1. Using Default Certs in Production
**Why it's bad**: Anyone can decrypt your traffic  
**Fix**: Generate custom certificates before going live

### 2. Not Testing Pump Hardware
**Why it's bad**: Pump might not start when needed  
**Fix**: Test GPIO connections with engine disconnected first

### 3. Wrong Camera Credentials  
**Why it's bad**: Cameras won't be discovered  
**Fix**: Add your camera's username:password to CAMERA_CREDENTIALS

### 4. Insufficient Storage
**Why it's bad**: Recordings will fail  
**Fix**: Use large USB drive (1TB+ recommended for 6 months)

### 5. Incorrect Runtime Settings
**Why it's bad**: Pump may run dry or overflow tank  
**Fix**: Calculate based on your tank size and flow rate

## 📚 Next Steps

### Basic Customization
1. **Adjust Detection Sensitivity**
   - Edit `CONSENSUS_THRESHOLD` for your camera count
   - Modify `MIN_CONFIDENCE` based on false positive rate

2. **Configure Pump Timing**
   - Set `MAX_ENGINE_RUNTIME` for your pump capacity
   - Adjust `REFILL_MULTIPLIER` for reservoir size
   - Add safety sensors if available

3. **Add Camera Credentials**
   - Update `CAMERA_CREDENTIALS` with your cameras
   - Test with `docker logs camera_detector`

4. **Configure Storage**
   - Format USB drive as ext4
   - Set retention period with `RECORD_RETAIN_DAYS`

### Advanced Features
- [Enable MQTT Authentication](mqtt_broker/README.md#authentication-setup-optional)
- [Configure Multi-Node Deployment](docs/multi-node.md)
- [Integrate with Home Assistant](docs/integrations.md)
- [Set Up Cloud Monitoring](mqtt_broker/README.md#bridge-configuration)
- [Train Custom Models](security_nvr/README.md#custom-model-training)

## 🆘 Troubleshooting

### Nothing Happens When Fire Detected
1. Check consensus threshold matches camera count
2. Verify MQTT broker is running: `docker ps`
3. Check GPIO permissions: `groups` should include `gpio`
4. Review logs: `docker logs fire_consensus`

### Cameras Not Found
1. Ensure cameras and detector on same network
2. Add camera credentials to environment
3. Check firewall allows ports 80, 554
4. Enable debug logging: `LOG_LEVEL=DEBUG`

### Storage Not Working
1. Check USB drive detected: `lsblk`
2. Verify mount: `df -h | grep frigate`
3. Check permissions: `ls -la /media/frigate`
4. Review logs: `docker logs security-nvr`

### Pump Won't Start
1. Check refill status in telemetry
2. Verify reservoir sensor if configured
3. Test GPIO manually
4. Check MQTT connection

## 📞 Getting Help

1. **Check Service Logs**
   ```bash
   docker logs [service_name] --tail 50
   ```

2. **Enable Debug Mode**
   ```bash
   LOG_LEVEL=DEBUG docker-compose up
   ```

3. **Community Support**
   - GitHub Issues for bugs
   - Discussions for questions
   - Wiki for documentation

## 🎉 Success Indicators

You'll know your system is working when:
- ✅ Cameras appear in Frigate UI
- ✅ Health reports show all services "online"
- ✅ Object detections appear in timeline
- ✅ Test fire detection triggers consensus
- ✅ GPIO pins activate in correct sequence
- ✅ USB storage shows recordings

## ⏰ Time Estimates

- **Basic deployment**: 5 minutes
- **Secure deployment**: 20 minutes
- **Full configuration**: 1 hour
- **Multi-node setup**: 2 hours

## 🏁 Summary

1. **Deploy quickly** with default settings for testing
2. **Secure properly** before production use
3. **Test thoroughly** with disconnected pumps
4. **Monitor actively** via MQTT and logs
5. **Maintain regularly** with certificate rotation

Remember: The default configuration gets you started quickly, but security is YOUR responsibility for production use!
