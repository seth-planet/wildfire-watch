# Configuration Reference

Complete configuration guide for Wildfire Watch. All settings use environment variables via `.env` file.

## Essential Configuration

**Minimum required settings:**

```bash
# Camera credentials (comma-separated user:pass pairs)
CAMERA_CREDENTIALS=username:password,username2:password2,root:rootpass

# Fire consensus (number of cameras that must agree)
CONSENSUS_THRESHOLD=2

# Pump safety (seconds before auto-shutoff)
MAX_ENGINE_RUNTIME=1800
```

## Service Configuration

### Camera Detector

```bash
# Network scanning
SCAN_INTERVAL=300              # Seconds between camera scans
NETWORK_SUBNET=192.168.1.0/24  # Subnet to scan (auto-detected if blank)
ONVIF_TIMEOUT=5                # ONVIF discovery timeout
PORT_SCAN_THREADS=50           # Parallel port scan threads

# Camera defaults
DEFAULT_RTSP_PORT=554          # RTSP port if not detected
DEFAULT_STREAM_PATH=/stream1   # Default stream path
MAX_CAMERAS=50                 # Maximum cameras to configure
```

### Frigate NVR

```bash
# AI Detection
FRIGATE_DETECTOR=auto          # auto|coral|hailo|gpu|cpu
FIRE_MODEL_PATH=/models/fire_edgetpu.tflite
MIN_CONFIDENCE=0.7             # Fire detection threshold (0-1)
DETECTION_FPS=5                # Detections per second

# Recording
RECORDING_ENABLED=true         # Enable 24/7 recording
RECORDING_RETAIN_DAYS=7        # Days to keep recordings
RECORDING_QUALITY=80           # JPEG quality (1-100)
SNAPSHOT_RETAIN_DAYS=14        # Days to keep snapshots

# Motion Detection
MOTION_THRESHOLD=25            # Motion sensitivity (0-255)
MOTION_CONTOUR_AREA=100        # Minimum motion area
MOTION_MASK_ZONES=true         # Enable motion masking
```

### Fire Consensus

```bash
# Consensus Logic
CONSENSUS_THRESHOLD=2          # Cameras required for agreement
CONSENSUS_TIMEOUT=30           # Seconds to collect detections
MIN_FIRE_SIZE=10000           # Minimum fire area (pixels²)
ACTIVATION_DELAY=5             # Delay before pump activation

# Advanced Filtering
SPATIAL_CORRELATION=true       # Require overlapping views
MAX_DETECTION_AGE=60          # Ignore old detections (seconds)
CONFIDENCE_WEIGHTING=true      # Weight by detection confidence
```

### GPIO Trigger

```bash
# GPIO Pins (BCM numbering)
PUMP_GPIO_PIN=17              # Pump relay control
VALVE_GPIO_PIN=27             # Valve control
FLOAT_SWITCH_PIN=22           # Tank level sensor
EMERGENCY_STOP_PIN=23         # Physical E-stop button

# Pump Control
MAX_ENGINE_RUNTIME=1800       # Maximum runtime (seconds)
REFILL_MULTIPLIER=40          # Tank refill time calculation
PUMP_TEST_DURATION=5          # Test mode duration
COOLDOWN_PERIOD=300           # Minimum time between activations

# Safety Features
ENABLE_WATCHDOG=true          # Hardware watchdog timer
FLOAT_SWITCH_REQUIRED=false   # Require tank sensor
TELEMETRY_INTERVAL=10         # Status update frequency
```

### MQTT Broker

```bash
# Network
MQTT_PORT=1883                # Standard MQTT port
MQTT_TLS_PORT=8883           # Secure MQTT port
MQTT_WEBSOCKET_PORT=9001     # WebSocket port
MQTT_BIND_ADDRESS=0.0.0.0    # Listen address

# Performance
MQTT_MAX_CLIENTS=1000        # Maximum connections
MQTT_MESSAGE_SIZE=262144     # Max message size (bytes)
MQTT_QOS_DEFAULT=1           # Default QoS level (0-2)
MQTT_KEEPALIVE=60            # Keepalive interval

# Security
MQTT_AUTH_REQUIRED=false     # Require authentication
MQTT_TLS_ONLY=false         # Force TLS connections
MQTT_PERSISTENCE=true        # Enable message persistence
```

## Network Configuration

```bash
# Docker Network
NETWORK_SUBNET=192.168.100.0/24
MQTT_STATIC_IP=192.168.100.10

# Camera Network
CAMERA_VLAN_ID=10             # Camera VLAN tag
MULTICAST_ENABLED=true        # Enable ONVIF multicast
DNS_SERVER=8.8.8.8           # DNS for camera resolution

# Multi-Node
ENABLE_BRIDGE_MODE=false      # Enable MQTT bridging
REMOTE_BROKER_HOST=           # Remote MQTT broker
REMOTE_BROKER_PORT=8883       # Remote broker port
```

## Hardware Acceleration

```bash
# Coral TPU
CORAL_DEVICE_PATH=/dev/bus/usb/002/004
CORAL_ORIENTATION=horizontal   # horizontal|vertical

# Hailo
HAILO_DEVICE_ID=0             # /dev/hailo0
HAILO_POWER_MODE=performance  # performance|balanced|efficiency

# NVIDIA GPU
NVIDIA_VISIBLE_DEVICES=all    # GPU selection
CUDA_COMPUTE_CAPABILITY=7.5   # Minimum compute capability
```

## Deployment Profiles

### Low Power Mode
```bash
SCAN_INTERVAL=600
DETECTION_FPS=3
RECORDING_QUALITY=70
MOTION_THRESHOLD=30
TELEMETRY_INTERVAL=30
```

### High Accuracy Mode
```bash
MIN_CONFIDENCE=0.85
CONSENSUS_THRESHOLD=3
SPATIAL_CORRELATION=true
CONFIDENCE_WEIGHTING=true
DETECTION_FPS=10
```

### Development Mode
```bash
MQTT_AUTH_REQUIRED=false
CONSENSUS_THRESHOLD=1
ACTIVATION_DELAY=30
PUMP_TEST_DURATION=2
DEBUG_LOGGING=true
```

## Environment File Example

Complete `.env` for production:

```bash
# Deployment
COMPOSE_PROJECT_NAME=wildfire-watch
COMPOSE_PROFILES=production,raspberry-pi

# Cameras
CAMERA_CREDENTIALS=admin:MySecurePass123,root:CameraAdmin456
NETWORK_SUBNET=192.168.10.0/24

# Detection
FRIGATE_DETECTOR=coral
MIN_CONFIDENCE=0.75
CONSENSUS_THRESHOLD=2

# Pump Safety
MAX_ENGINE_RUNTIME=1800
REFILL_MULTIPLIER=40
COOLDOWN_PERIOD=300

# Security
MQTT_AUTH_REQUIRED=true
MQTT_TLS_ONLY=true

# Performance
RECORDING_RETAIN_DAYS=7
DETECTION_FPS=5
```

## Configuration Validation

Test configuration:
```bash
docker-compose config
docker-compose run --rm fire_consensus python -m config_validator
```

## Dynamic Reconfiguration

Some settings can be changed at runtime via MQTT:

```bash
# Change consensus threshold
mosquitto_pub -t 'config/consensus/threshold' -m '3'

# Adjust detection sensitivity  
mosquitto_pub -t 'config/frigate/confidence' -m '0.8'

# Emergency pump shutoff time
mosquitto_pub -t 'config/gpio/max_runtime' -m '900'
```
