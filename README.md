
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

```bash
# Clone repository
git clone https://github.com/your-org/wildfire-watch.git
cd wildfire-watch

# Generate secure certificates (REQUIRED for production)
./scripts/generate_certs.sh custom

# Deploy with Docker Compose
docker-compose up -d

# Access Frigate UI
http://your-device:5000
```

⚠️ **Security Warning**: Default certificates are INSECURE. See [Security Setup](../docs/security.md).

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

### Advanced Topics
- [**Model Converter**](converted_models/README.md) - Custom AI models
- [**Multi-Node Setup**](docs/multi-node.md) - Scale to large properties
- [**Troubleshooting**](docs/troubleshooting.md) - Common issues

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

## Configuration

### Essential Settings (.env)

```bash
# Camera credentials
CAMERA_CREDENTIALS=admin:password,admin:12345

# Fire detection
CONSENSUS_THRESHOLD=2      # Cameras required for consensus
MIN_CONFIDENCE=0.7         # AI confidence threshold

# Pump control
MAX_ENGINE_RUNTIME=1800    # 30 min (adjust for your water tank size!)
REFILL_MULTIPLIER=40       # Refill duration multiplier

# Hardware
FRIGATE_DETECTOR=auto      # auto|coral|hailo|gpu|cpu
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
| Coral TPU | 15-20ms | 2W | Low power, always-on |
| Hailo-8L | 20-25ms | 2.5W | Raspberry Pi 5 |
| Hailo-8 | 10-15ms | 5W | High accuracy |
| NVIDIA GPU | 8-12ms | 15W+ | Multiple cameras |
| CPU Only | 200-300ms | 5W | Testing only |

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
