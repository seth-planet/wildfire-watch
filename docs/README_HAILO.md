# Hailo-8L Integration for Wildfire Watch

## Overview

This directory contains the Hailo-8L M.2 AI accelerator integration for the Wildfire Watch fire detection system. The integration enables high-performance, low-power inference using quantized INT8 models optimized for the Hailo-8L hardware.

## Quick Start

```bash
# 1. Validate installation
./scripts/validate_hailo_integration.sh

# 2. Configure environment
cp .env.hailo .env

# 3. Start services
docker-compose up -d

# 4. Monitor detections
./scripts/demo_hailo_fire_detection.py
```

## Directory Structure

```
wildfire-watch/
├── converted_models/
│   ├── hailo_qat_output/         # Converted HEF models
│   │   ├── yolo8l_fire_640x640_hailo8l_qat.hef  # Main model
│   │   └── yolo8l_fire_640x640_hailo8_qat.hef   # Alternative
│   ├── hailo_utils/              # Hailo-specific utilities
│   │   ├── inspect_hef.py        # Model inspection tool
│   │   └── validate_hef.py       # Validation tool
│   └── convert_hailo_qat.py      # QAT conversion script
├── tests/
│   ├── test_hailo_*.py           # Hailo-specific tests
│   └── hailo_test_utils.py       # Test utilities
├── docs/
│   ├── hailo_quick_start.md      # 5-minute guide
│   ├── hailo_deployment_guide.md # Full deployment
│   ├── hailo_troubleshooting_guide.md # Issues & fixes
│   └── hailo_model_validation_guide.md # Validation
└── scripts/
    ├── validate_hailo_integration.sh # Validation script
    └── demo_hailo_fire_detection.py  # Demo monitor
```

## Key Features

- **13 TOPS Performance**: Hailo-8L M.2 AI accelerator
- **INT8 Quantization**: 4x model size reduction with <2% accuracy loss
- **Batch Processing**: Optimized for batch size 8
- **Low Power**: ~5W typical, ~10W under load
- **High Throughput**: 40+ FPS single stream, 400+ FPS with batching
- **Low Latency**: 10-25ms per inference

## Integration Method

This integration uses **Frigate NVR** as the primary interface to Hailo hardware:

```yaml
# Frigate handles all Hailo API calls
detectors:
  hailo8l:
    type: hailo8l
    device: PCIe
```

Benefits:
- Frigate manages complex Hailo API
- Proven stability in production
- Easy configuration
- Active community support

## Performance Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| Latency | <25ms | ✓ 10-25ms |
| FPS | >40 | ✓ 40-100 |
| Temperature | <85°C | ✓ <70°C |
| Power | <10W | ✓ 5-10W |
| Accuracy | >90% | ✓ >90% |

## Testing

Run the complete test suite:

```bash
# Check Python 3.10 environment
python3.10 --version

# Run accuracy tests
python3.10 tests/test_hailo_accuracy.py

# Run performance benchmarks
python3.10 tests/test_performance_benchmarks.py

# Run stability tests (5 minutes)
python3.10 tests/test_stability_temperature.py --duration 300
```

## Common Issues

### Device Not Found
```bash
# Check device
ls -la /dev/hailo0

# Fix permissions
sudo chmod 666 /dev/hailo0
```

### Low Performance
```bash
# Check batch size in model
python3.10 converted_models/hailo_utils/inspect_hef.py \
  --hef converted_models/hailo_qat_output/yolo8l_fire_640x640_hailo8l_qat.hef

# Should show: Batch size: 8
```

### API Errors
Use Frigate integration instead of direct API. See `docs/hailo_troubleshooting_guide.md` for details.

## Documentation

- **Quick Start**: [docs/hailo_quick_start.md](docs/hailo_quick_start.md) - Get running in 5 minutes
- **Deployment**: [docs/hailo_deployment_guide.md](docs/hailo_deployment_guide.md) - Production setup
- **Validation**: [docs/hailo_model_validation_guide.md](docs/hailo_model_validation_guide.md) - Testing procedures
- **Troubleshooting**: [docs/hailo_troubleshooting_guide.md](docs/hailo_troubleshooting_guide.md) - Common issues
- **Summary**: [docs/hailo_integration_summary.md](docs/hailo_integration_summary.md) - Technical details

## Model Details

### YOLOv8L Fire Detection Model
- **Input**: 640x640x3 RGB images
- **Output**: Bounding boxes with fire/smoke classification
- **Classes**: fire, smoke, person, vehicle, wildlife
- **Quantization**: INT8 with QAT optimization
- **Calibration**: 1500 wildfire-specific images

### Conversion Process
```bash
# Original model → ONNX → HAR → HEF
python3.10 converted_models/convert_hailo_qat.py
```

## Support

- **Hailo Community**: https://community.hailo.ai/
- **Frigate Docs**: https://docs.frigate.video/
- **Project Issues**: GitHub repository

## License

Same as Wildfire Watch project. Hailo integration components are provided as-is for fire safety applications.

---
*Hailo integration completed 2025-06-28*