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
- ✅ Configure Frigate for fire detection
- ✅ Monitor for consensus events
- ✅ Control GPIO pins when fire detected

### 3. Access the System

- **MQTT Explorer**: Connect to `device-ip:1883` or `device-ip:8883`
- **Frigate Web UI**: Browse to `http://device-ip:5000`
- **Logs**: `balena logs` or `docker-compose logs`

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
MAX_ENGINE_RUNTIME: "600"     # 10-minute safety limit
FIRE_OFF_DELAY: "1800"        # Run 30 min after fire gone
```

### Network Architecture

```
[IP Cameras] ←→ [Camera Detector] ←→ [MQTT Broker] ←→ [Fire Consensus]
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

### 3. Test Fire Detection

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

### 4. Consensus Threshold Too High
**Why it's bad**: System won't trigger with few cameras  
**Fix**: Set CONSENSUS_THRESHOLD to match your camera count

## 📚 Next Steps

### Basic Customization
1. **Adjust Detection Sensitivity**
   - Edit `CONSENSUS_THRESHOLD` for your camera count
   - Modify `MIN_CONFIDENCE` based on false positive rate

2. **Configure Pump Timing**
   - Set `MAX_ENGINE_RUNTIME` for your pump capacity
   - Adjust `REFILL_MULTIPLIER` for reservoir size

3. **Add Camera Credentials**
   - Update `CAMERA_CREDENTIALS` with your cameras
   - Test with `docker logs camera_detector`

### Advanced Features
- [Enable MQTT Authentication](mqtt_broker/README.md#authentication-setup-optional)
- [Configure Multi-Node Deployment](docs/multi-node.md)
- [Integrate with Home Assistant](docs/integrations.md)
- [Set Up Cloud Monitoring](mqtt_broker/README.md#bridge-configuration)

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

### Certificate Errors
1. Ensure `/mnt/data/certs` contains all files
2. Check file permissions: `ls -la /mnt/data/certs`
3. Verify certificate validity: `openssl x509 -in /mnt/data/certs/server.crt -dates`
4. Try with plain MQTT first: `MQTT_TLS=false`

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
- ✅ Cameras appear in logs within 5 minutes
- ✅ Health reports show all services "online"
- ✅ Frigate web UI shows camera feeds
- ✅ Test fire detection triggers consensus
- ✅ GPIO pins activate in correct sequence

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
