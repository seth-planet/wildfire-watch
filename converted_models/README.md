# 🔄 Model Converter - Wildfire Watch

## Overview

The Model Converter transforms YOLOv8/v9/v11 PyTorch models into optimized formats for edge deployment on Wildfire Watch systems. It automatically handles the complex conversion process for multiple hardware accelerators while preserving model accuracy.

## Features

- ✅ **Automatic format detection** - Identifies YOLO version and extracts model metadata
- 🎯 **Multi-target support** - Converts to Hailo, Coral, TensorRT, OpenVINO, and ONNX
- 📊 **Calibration handling** - Downloads or uses custom calibration data for quantization
- 🔧 **Frigate integration** - Generates configuration files for immediate deployment
- 📝 **Label extraction** - Automatically extracts and formats class labels
- 🚀 **Optimization scripts** - Creates device-specific optimization scripts

## Quick Start

### Basic Usage

```bash
# Convert a YOLOv8 model to all formats
python convert_model.py path/to/your_model.pt

# Convert specific model with custom name
python convert_model.py fire_detector.pt --name wildfire_v1

# Convert to specific formats only
python convert_model.py model.pt --formats onnx tflite hailo
```

### Using Pre-trained Models

```bash
# Download a pre-trained model (example with YOLOv8)
yolo export model=yolov8n.pt format=torchscript

# Convert it
python convert_model.py yolov8n.pt --name yolov8n_wildfire
```

## Installation

### Prerequisites

```bash
# Core requirements
pip install ultralytics>=8.0.0
pip install onnx>=1.12.0
pip install numpy>=1.21.0
pip install pyyaml>=5.4.1

# For TensorFlow Lite conversion
pip install tensorflow>=2.10.0

# For Edge TPU compilation (optional)
# Follow: https://coral.ai/docs/edgetpu/compiler/
```

### Hardware-Specific Tools

#### Hailo
1. Register at https://hailo.ai/developer-zone/
2. Download Hailo Dataflow Compiler
3. Install Docker and pull Hailo image

#### OpenVINO
```bash
# Install OpenVINO toolkit
pip install openvino-dev
```

#### TensorRT
- Install on target device only
- Requires NVIDIA GPU with CUDA

## Supported Formats

### 1. ONNX (Universal)
- **Output**: `model.onnx`
- **Use**: Base format for other conversions
- **Compatibility**: All platforms

### 2. TensorFlow Lite (Coral TPU)
- **Outputs**:
  - `model_cpu.tflite` - Float32 CPU inference
  - `model_quant.tflite` - INT8 quantized
  - `model_edgetpu.tflite` - Coral TPU optimized
- **Use**: Google Coral USB/PCIe accelerators
- **Performance**: 10-30ms inference

### 3. Hailo (Hailo-8/8L)
- **Outputs**:
  - `model_hailo8.hef` - Hailo-8 (26 TOPS)
  - `model_hailo8l.hef` - Hailo-8L (13 TOPS)
- **Use**: Hailo M.2 accelerators
- **Performance**: 15-40ms inference

### 4. OpenVINO (Intel)
- **Outputs**:
  - `model_openvino.xml` - Model structure
  - `model_openvino.bin` - Model weights
- **Use**: Intel CPUs, GPUs, VPUs
- **Performance**: 20-50ms inference

### 5. TensorRT (NVIDIA)
- **Output**: `model_tensorrt.engine`
- **Use**: NVIDIA GPUs
- **Performance**: 10-30ms inference

## Detailed Usage

### Custom Calibration Data

For best quantization results, provide your own calibration images:

```bash
# Prepare calibration dataset
mkdir calibration_images
cp path/to/your/images/*.jpg calibration_images/

# Use in conversion
python convert_model.py model.pt --calibration-data calibration_images/
```

**Calibration Best Practices**:
- Use 500-1000 representative images
- Include all classes you want to detect
- Include various lighting conditions
- Include edge cases and difficult scenarios

### Conversion Examples

#### Fire Detection Model
```bash
# Train custom fire detection model
yolo train model=yolov8n.pt data=fire_dataset.yaml epochs=100

# Convert for edge deployment
python convert_model.py runs/detect/train/weights/best.pt \
    --name fire_detector_v1 \
    --calibration-data fire_images/
```

#### Multi-Class Wildlife Model
```bash
# Convert wildlife detection model
python convert_model.py wildlife_yolov8.pt \
    --name wildlife_detector \
    --formats hailo tflite
```

### Output Structure

```
converted_models/
├── fire_detector_v1/
│   ├── fire_detector_v1.onnx
│   ├── fire_detector_v1_cpu.tflite
│   ├── fire_detector_v1_quant.tflite
│   ├── fire_detector_v1_edgetpu.tflite
│   ├── fire_detector_v1_hailo_config.json
│   ├── convert_fire_detector_v1_hailo.sh
│   ├── convert_fire_detector_v1_openvino.sh
│   ├── convert_fire_detector_v1_tensorrt.py
│   ├── fire_detector_v1_frigate_config.yml
│   ├── fire_detector_v1_labels.txt
│   ├── conversion_summary.json
│   └── README.md
```

## Deployment Guide

### 1. Hailo Deployment

```bash
# Inside Hailo Docker container
cd converted_models/fire_detector_v1
./convert_fire_detector_v1_hailo.sh

# Copy to Raspberry Pi 5 with Hailo
scp fire_detector_v1_hailo8l.hef pi@raspberrypi:/models/
```

### 2. Coral TPU Deployment

```bash
# Copy Edge TPU model
docker cp fire_detector_v1_edgetpu.tflite frigate:/models/

# Update Frigate config
docker exec frigate sed -i 's/path: .*/path: \/models\/fire_detector_v1_edgetpu.tflite/' /config/config.yml
```

### 3. GPU Deployment (TensorRT)

```bash
# On target device with TensorRT
python convert_fire_detector_v1_tensorrt.py

# Copy to Frigate
docker cp fire_detector_v1_tensorrt.engine frigate:/models/
```

### 4. Frigate Configuration

The converter generates a Frigate config snippet. Merge it with your existing config:

```yaml
# From generated fire_detector_v1_frigate_config.yml
model:
  path: /models/fire_detector_v1_edgetpu.tflite
  input_tensor: nhwc
  input_pixel_format: rgb
  width: 640
  height: 640
  labels: /models/fire_detector_v1_labels.txt

objects:
  track:
    - fire
    - smoke
    - person
    - vehicle
  filters:
    fire:
      min_area: 300
      max_area: 100000
      min_score: 0.6
      threshold: 0.7
```

## Model Optimization Tips

### 1. Input Size Selection
- **640x640**: Best balance of speed and accuracy
- **320x320**: Faster inference, lower accuracy
- **1280x1280**: Higher accuracy, slower inference

### 2. Quantization Impact
- **INT8**: 3-4x faster, ~5% accuracy loss
- **FP16**: 2x faster, ~1% accuracy loss
- **FP32**: Baseline accuracy, slowest

### 3. Hardware Matching
- **Hailo-8**: Best for high-resolution, multi-camera
- **Coral TPU**: Best for low-power, single camera
- **GPU**: Best for high-accuracy, complex models

## Troubleshooting

### Common Issues

#### 1. ONNX Export Fails
```bash
# Error: Unsupported ONNX opset version
# Solution: Use opset 11 for maximum compatibility
python convert_model.py model.pt --opset 11
```

#### 2. Quantization Accuracy Loss
```bash
# Problem: Model accuracy drops significantly after quantization
# Solution: Use more calibration data
python convert_model.py model.pt --calibration-data large_dataset/
```

#### 3. Hailo Compilation Errors
```bash
# Error: Unsupported layer type
# Solution: Check layer compatibility in Hailo docs
# Or simplify model architecture
```

#### 4. TensorRT Version Mismatch
```bash
# Error: TensorRT version mismatch
# Solution: Generate engine on target device
# Never copy .engine files between devices
```

### Debug Mode

```bash
# Enable verbose logging
LOG_LEVEL=DEBUG python convert_model.py model.pt

# Test single format conversion
python convert_model.py model.pt --formats onnx
```

## Advanced Features

### Custom Layer Handling

```python
# For models with custom layers
from convert_model import ModelConverter

class CustomConverter(ModelConverter):
    def _get_yolo_output_names(self):
        # Override for custom architecture
        return ["output1", "output2", "output3"]

converter = CustomConverter("model.pt")
converter.convert_all()
```

### Batch Processing

```bash
# Convert multiple models
for model in models/*.pt; do
    python convert_model.py "$model" --output-dir "converted/$(basename $model .pt)"
done
```

### Performance Benchmarking

```python
# After conversion, benchmark on target
import time
import numpy as np

# Load model (example for TFLite)
import tflite_runtime.interpreter as tflite

interpreter = tflite.Interpreter(model_path="model_edgetpu.tflite")
interpreter.allocate_tensors()

# Benchmark
times = []
for _ in range(100):
    input_data = np.random.rand(1, 640, 640, 3).astype(np.float32)
    start = time.time()
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    times.append(time.time() - start)

print(f"Average inference time: {np.mean(times)*1000:.2f}ms")
```

## Best Practices

1. **Always validate converted models**
   - Test on sample images
   - Compare outputs with original
   - Check performance metrics

2. **Use appropriate calibration data**
   - Representative of deployment environment
   - Includes all target classes
   - Sufficient quantity (500+ images)

3. **Version your models**
   - Track model lineage
   - Document training parameters
   - Save conversion settings

4. **Test on target hardware**
   - Performance varies by device
   - Thermal throttling affects speed
   - Memory constraints matter

## Integration with Wildfire Watch

The converted models integrate seamlessly with the Wildfire Watch system:

1. **Camera Detector** finds cameras
2. **Security NVR** loads appropriate model
3. **Fire Consensus** validates detections
4. **GPIO Trigger** activates sprinklers

Model files should be placed in:
- Docker: `/models/` directory
- Balena: `/data/models/` persistent storage

## Contributing

To add support for new model architectures:

1. Fork the repository
2. Add architecture detection in `_extract_model_info()`
3. Add output layer mapping in `_get_yolo_output_names()`
4. Submit pull request with test model

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Ultralytics for YOLO implementation
- Hailo team for optimization guides
- Coral team for Edge TPU compiler
- OpenVINO team for model optimizer

## Support

For issues or questions:
1. Check existing GitHub issues
2. Review troubleshooting section
3. Create detailed issue with:
   - Model architecture
   - Conversion command
   - Error messages
   - System information
