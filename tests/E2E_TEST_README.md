# E2E Test Configuration

## Running End-to-End Tests

The E2E tests discover and use real cameras on your network. To run them:

### 1. Set Camera Credentials

Export your camera credentials as an environment variable:

```bash
export CAMERA_CREDENTIALS="admin:your_password"
```

For multiple credential sets (the detector will try each one):
```bash
export CAMERA_CREDENTIALS="username:password1,username:password2,user:pass3"
```

### 2. Run the E2E Test

```bash
python3.12 -m pytest tests/test_e2e_fire_detection_full.py -v -s
```

### 3. What the Test Does

1. **Builds all Docker images** - MQTT broker, Camera Detector, Frigate, Fire Consensus, GPIO Trigger
2. **Discovers real cameras** - Scans your network for ONVIF/RTSP cameras
3. **Configures Frigate** - Automatically sets up Frigate with discovered cameras
4. **Simulates fire detection** - Injects fire detection events via MQTT
5. **Verifies consensus** - Ensures multi-camera consensus logic works
6. **Triggers GPIO** - Verifies pump activation (simulated in test mode)

### 4. Requirements

- Docker installed and running
- At least one IP camera on your network
- Camera credentials set via environment variable
- Network access to cameras from Docker containers

### 5. Troubleshooting

If cameras aren't discovered:
- Check that cameras are on the same network
- Verify credentials are correct
- Ensure Docker containers can access your LAN
- Check firewall rules for ONVIF (port 80) and RTSP (port 554)

### 6. Test Timeout

The full E2E test can take 5-10 minutes due to:
- Docker image building
- Camera discovery process
- Service startup time
- Simulated fire detection cycle

Use `--timeout=600` if needed:
```bash
python3.12 -m pytest tests/test_e2e_fire_detection_full.py -v -s --timeout=600
```