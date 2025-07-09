
# 🔥 Wildfire Watch

**Open-source wildfire detection and suppression system for edge deployment**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?logo=docker&logoColor=white)](docker-compose.yml)
[![Balena](https://img.shields.io/badge/balena-%23000000.svg?logo=balena&logoColor=white)](https://www.balena.io/)

## Overview

Wildfire Watch is an automated fire detection and suppression platform that runs entirely on your local network. It uses AI-powered camera monitoring to detect fires and automatically activates sprinkler systems when multiple cameras confirm a fire.

### Key Features
- 🎯 **Multi-camera consensus** - Prevents false alarms
- 🚀 **Edge AI acceleration** - Supports Coral, Hailo, NVIDIA GPUs
- 💧 **Automated pump control** - GPIO-based sprinkler activation
- 📹 **24/7 recording** - Frigate NVR with motion detection
- 🔒 **Secure by default** - TLS encryption (requires certificate setup)
- 🔄 **Self-healing** - Automatic camera discovery and failover

## Quick Start

### 1. Clone and Setup

```bash
# Clone repository
git clone https://github.com/your-org/wildfire-watch.git
cd wildfire-watch

# Copy environment template
cp .env.example .env
# Edit .env to configure your settings
```

### 2. Security Setup (Choose One)

#### Option A: Development/Testing (Insecure)
```bash
# Use default certificates - INSECURE, for testing only!
# No action needed, default certs are included
```

#### Option B: Production (Secure)
```bash
# Generate secure certificates and enable TLS
./scripts/configure_security.sh enable
# This will guide you through certificate generation if needed

# Or manually:
./scripts/generate_certs.sh custom
echo "MQTT_TLS=true" >> .env
```

### 3. Deploy Services

```bash
# Deploy with Docker Compose
docker-compose up -d

# Verify all services are running
docker ps

# Check service logs
docker-compose logs -f
```

### 4. Access Services

- **Frigate NVR**: http://your-device:5000
  - Default credentials are shown in logs: `docker logs security_nvr | grep Password`
- **MQTT Broker**: Port 1883 (insecure) or 8883 (TLS)

⚠️ **Security Warning**: Default certificates are INSECURE. Always generate custom certificates for production use. See [Security Guide](docs/security.md).

### 5. Verify Security Configuration

```bash
# Check current security status
./scripts/configure_security.sh status

# Example output:
# ✓ TLS is ENABLED
# ✓ Using CUSTOM certificates
# ✓ MQTT broker is using TLS port 8883
```

## Documentation

### Getting Started
- [**Raspberry Pi 5 Guide**](docs/QUICK_START_pi5.md) - Recommended for most users
- [**Linux PC Guide**](docs/QUICK_START_pc.md) - For x86 systems with GPU
- [**Hardware Requirements**](docs/hardware.md) - Complete component list

### Service Documentation
- [**Camera Detector**](camera_detector/README.md) - Automatic camera discovery
- [**Security NVR**](security_nvr/README.md) - Frigate configuration
- [**Fire Consensus**](fire_consensus/README.md) - Multi-camera validation
- [**GPIO Trigger**](gpio_trigger/README.md) - Pump control system
- [**MQTT Broker**](mqtt_broker/README.md) - Communication hub
- [**Camera Telemetry**](cam_telemetry/README.md) - Health monitoring and metrics

### Advanced Topics
- [**Model Converter**](converted_models/README.md) - Custom AI models
- [**Multi-Node Setup**](docs/multi-node.md) - Scale to large properties
- [**Troubleshooting**](docs/troubleshooting.md) - Common issues
- [**Coral TPU Python 3.8**](docs/coral_python38_requirements.md) - Important compatibility info

## System Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  IP Cameras │────▶│   Camera    │────▶│  Frigate    │
│   (RTSP)    │     │  Detector   │     │    NVR      │
└─────────────┘     └─────────────┘     └─────────────┘
                            │                    │
                            ▼                    ▼
                    ┌─────────────┐     ┌─────────────┐
                    │    MQTT     │◀────│    Fire     │
                    │   Broker    │     │  Consensus  │
                    └─────────────┘     └─────────────┘
                            │                    │
                            ▼                    ▼
                    ┌─────────────┐     ┌─────────────┐
                    │    GPIO     │────▶│    Pump     │
                    │  Trigger    │     │   System    │
                    └─────────────┘     └─────────────┘
```

### Service Architecture (Refactored)
All services now use standardized base classes for improved reliability:

- **MQTTService**: Automatic reconnection with exponential backoff
- **HealthReporter**: Standardized health monitoring and metrics
- **ThreadSafeService**: Safe thread management and shutdown
- **ConfigBase**: Centralized configuration management

## Configuration

### Essential Settings (.env)

```bash
# Camera credentials (comma-separated username:password pairs)
CAMERA_CREDENTIALS=username:password,username2:password2

# Fire detection
CONSENSUS_THRESHOLD=2      # Cameras required for consensus
MIN_CONFIDENCE=0.7         # AI confidence threshold

# Pump control
MAX_ENGINE_RUNTIME=1800    # 30 min (adjust for your water tank size!)
REFILL_MULTIPLIER=40       # Refill duration multiplier

# Hardware
FRIGATE_DETECTOR=auto      # auto|coral|hailo|gpu|cpu

# Service health monitoring
HEALTH_REPORT_INTERVAL=60  # Health report frequency (seconds)

# MQTT reconnection (all services)
MQTT_RECONNECT_MIN_DELAY=1.0   # Min reconnection delay
MQTT_RECONNECT_MAX_DELAY=60.0  # Max reconnection delay
```

See [Configuration Guide](docs/configuration.md) for all options.

## Deployment Options

### Docker Compose (Recommended)
```bash
docker-compose up -d
```

### Balena Cloud
```bash
balena push wildfire-watch
```

### Kubernetes
```bash
kubectl apply -k k8s/
```

## Hardware Support

| Accelerator | Performance | Power | Recommended For |
|------------|-------------|--------|-----------------|
| Coral TPU* | 15-20ms | 2W | Low power, always-on |
| Hailo-8L | 20-25ms | 2.5W | Raspberry Pi 5 |
| Hailo-8 | 10-15ms | 5W | High accuracy |
| NVIDIA GPU | 8-12ms | 15W+ | Multiple cameras |
| CPU Only | 200-300ms | 5W | Testing only |

*Note: Coral TPU requires Python 3.8 for tflite_runtime compatibility

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup.

## Support

- 📖 [Documentation](docs/)
- 💬 [Discussions](https://github.com/seth-planet/wildfire-watch/discussions)
- 🐛 [Issues](https://github.com/seth-planet/wildfire-watch/issues)

## License

MIT License - See [LICENSE](LICENSE) for details.

## Disclaimer

This system is provided as-is for educational and experimental use. It does not guarantee fire prevention or property protection. Always follow local fire safety regulations and consult professionals for critical safety systems.
