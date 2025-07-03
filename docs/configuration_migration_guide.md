# Configuration Migration Guide

This guide explains how to migrate services to use the centralized configuration system, which properly handles Docker vs bare metal deployments and decouples service dependencies.

## Overview

The new configuration system provides:

1. **Service Discovery** - Automatic hostname resolution based on deployment mode
2. **Path Resolution** - Proper paths for configs, certs, and data
3. **Topic Management** - Centralized MQTT topic definitions
4. **Environment Flexibility** - Works in Docker, Kubernetes, Balena, and bare metal

## Migration Steps

### 1. Replace Hardcoded Values

#### Before:
```python
MQTT_BROKER = os.getenv('MQTT_BROKER', 'localhost')
MQTT_PORT = int(os.getenv('MQTT_PORT', '1883'))
TOPIC_FIRE = 'detection/fire'
```

#### After:
```python
from config_manager import ConfigManager

config = ConfigManager('my_service')
mqtt_config = config.get_mqtt_config()

MQTT_BROKER = mqtt_config['broker']
MQTT_PORT = mqtt_config['port']
TOPIC_FIRE = config.topics.detection_fire
```

### 2. Use Service Discovery

#### Before:
```python
# Hardcoded service locations
frigate_url = "http://frigate:5000"
mqtt_host = "mqtt-broker"  # Only works in Docker
```

#### After:
```python
# Automatic resolution based on deployment
frigate_config = config.get_service_config('frigate')
frigate_url = f"http://{frigate_config['host']}:{frigate_config['port']}"

mqtt_host = config.discovery.get_service_host('mqtt-broker')
```

### 3. Handle File Paths

#### Before:
```python
# Breaks between Docker and bare metal
config_file = "/app/config/settings.yaml"  # Docker path
cert_path = "./certs/ca.crt"  # Bare metal path
```

#### After:
```python
# Works everywhere
config_file = config.paths.get_config_path('settings.yaml')
cert_path = config.paths.get_cert_path('ca')
model_dir = config.paths.get_data_path('models')
```

### 4. Service-Specific Examples

#### Camera Detector
```python
from config_manager import CameraDetectorConfig

class CameraDetector:
    def __init__(self):
        self.config = CameraDetectorConfig()
        
        # Use discovery for MQTT
        mqtt_config = self.config.get_mqtt_config()
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.connect(
            mqtt_config['broker'],
            mqtt_config['port']
        )
        
        # Use centralized topics
        self.mqtt_client.publish(
            self.config.topics.camera_discovery.replace('+', camera_id),
            camera_data
        )
        
        # Service-specific settings
        self.discovery_interval = self.config.discovery_interval
        self.rtsp_timeout = self.config.rtsp_timeout
```

#### Fire Consensus
```python
from config_manager import FireConsensusConfig

class FireConsensus:
    def __init__(self):
        self.config = FireConsensusConfig()
        
        # Subscribe to topics
        topics_to_subscribe = [
            self.config.topics.detection_fire,
            self.config.topics.frigate_events,
            self.config.topics.camera_discovery
        ]
        
        # Use consensus settings
        self.threshold = self.config.consensus_threshold
        self.min_confidence = self.config.min_confidence
```

#### GPIO Trigger
```python
from config_manager import GPIOTriggerConfig

class PumpController:
    def __init__(self):
        self.config = GPIOTriggerConfig()
        
        # Get pin configuration
        self.pins = self.config.get_pin_config()
        
        # Check simulation mode
        if self.config.gpio_simulation:
            logger.info("Running in GPIO simulation mode")
        
        # Engine settings
        self.max_runtime = self.config.max_runtime
```

## Environment Variables

The system supports environment variable overrides using this pattern:

```bash
# Format: SERVICE_NAME_CONFIG_KEY
CAMERA_DETECTOR_DISCOVERY_INTERVAL=600
FIRE_CONSENSUS_CONSENSUS_THRESHOLD=3
GPIO_TRIGGER_GPIO_SIMULATION=true

# Service hosts (override service discovery)
MQTT_BROKER_HOST=192.168.1.100
FRIGATE_HOST=nvr.local

# Paths
CERT_DIR=/custom/certs
MODELS_PATH=/mnt/models
```

## Configuration Files

Services can have YAML configuration files:

```yaml
# camera_detector.yaml
discovery:
  interval: 300
  smart_mode: true

rtsp:
  timeout: 10
  retry_attempts: 3

cameras:
  credentials:
    - ""
    - "user:12345"
```

## Docker Compose Integration

Update `docker-compose.yml` to use service names:

```yaml
services:
  camera-detector:
    environment:
      # No need to specify MQTT_BROKER=mqtt-broker
      # Service discovery handles it automatically
      - NODE_ID=detector-1
      
  fire-consensus:
    environment:
      # Override specific settings if needed
      - FIRE_CONSENSUS_CONSENSUS_THRESHOLD=2
```

## Testing

Test configuration in different modes:

```python
import pytest
from config_manager import ServiceDiscovery, DeploymentMode

def test_service_discovery_docker(monkeypatch):
    # Mock Docker environment
    monkeypatch.setattr('os.path.exists', lambda p: p == '/.dockerenv')
    
    discovery = ServiceDiscovery()
    assert discovery.mode == DeploymentMode.DOCKER
    assert discovery.get_service_host('mqtt-broker') == 'mqtt-broker'

def test_service_discovery_bare_metal():
    discovery = ServiceDiscovery()
    # On bare metal, services default to localhost
    assert discovery.get_service_host('mqtt-broker') == 'localhost'
```

## Benefits

1. **No more hardcoded hostnames** - Services work in any environment
2. **Consistent topic names** - No typos or mismatches
3. **Proper path handling** - No more Docker vs bare metal issues
4. **Easy testing** - Mock deployment modes easily
5. **Configuration validation** - Type checking and defaults
6. **Debug support** - Save runtime configs for troubleshooting

## Rollout Strategy

1. Start with one service (e.g., Camera Detector)
2. Test in both Docker and bare metal
3. Migrate other services incrementally
4. Update docker-compose.yml to remove redundant environment variables
5. Update documentation

## Troubleshooting

### Service Can't Connect
```python
# Debug service discovery
config = ConfigManager('my_service')
print(f"Deployment mode: {config.discovery.mode}")
print(f"MQTT host resolved to: {config.discovery.get_service_host('mqtt-broker')}")

# Save runtime config for debugging
config.save_runtime_config(config.get_mqtt_config())
```

### Wrong Paths
```python
# Check path resolution
print(f"App root: {config.paths.app_root}")
print(f"Config path: {config.paths.get_config_path('test.yaml')}")
```

### Topic Mismatches
```python
# Verify topic configuration
print(f"Fire topic: {config.topics.detection_fire}")
print(f"LWT topic: {config.topics.get_lwt_topic('my_service', 'node-1')}")
```