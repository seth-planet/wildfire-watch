# 🔄 Model Converter - Wildfire Watch

## Overview

The Model Converter transforms YOLOv8/v9/v11 PyTorch models into optimized formats for edge deployment on Wildfire Watch systems. It automatically handles the complex conversion process for multiple hardware accelerators while preserving model accuracy.

**Note**: This converter uses external tools (some GPL/AGPL licensed) in isolated processes to avoid license contamination of the main codebase.

## Features

- ✅ **Automatic format detection** - Identifies YOLO version and extracts model metadata
- 🎯 **Multi-target support** - Converts to Hailo, Coral, TensorRT, OpenVINO, and ONNX
- 📊 **Calibration handling** - Downloads or uses custom calibration data for quantization
- 🔧 **Frigate integration** - Generates configuration files for immediate deployment
- 📝 **Label extraction** - Automatically extracts and formats class labels
- 🚀 **Non-square support** - Can export models at different sizes (e.g., 320x240)

## Quick Start

### Basic Usage

```bash
# Convert a local YOLOv8 model to all formats
python convert_model.py path/to/your_model.pt

# Download and convert a pre-trained model
python convert_model.py --download yolov8n

# Convert with custom name and size
python convert_model.py fire_detector.pt --name wildfire_v1 --size 416

# Convert to specific formats only
python convert_model.py model.pt --formats onnx tflite hailo
```

## Installation

### Core Requirements

```bash
# Basic requirements (no GPL dependencies)
pip install numpy>=1.21.0
pip install pyyaml>=5.4.1
pip install Pillow>=8.2.0

# For ONNX conversion (optional)
pip install onnx>=1.12.0
pip install torch>=1.9.0  # PyTorch is needed for model loading

# For TensorFlow Lite conversion (optional)
pip install tensorflow>=2.10.0
pip install onnx-tf>=1.10.0

# For model inspection (GPL - used in subprocess only)
pip install ultralytics>=8.0.0
```

### Hardware-Specific Tools

#### Coral Edge TPU
```bash
# Install Edge TPU compiler
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt update
sudo apt install edgetpu-compiler
```

#### Hailo
1. Register at [Hailo Developer Zone](https://hailo.ai/developer-zone/)
2. Download Hailo Dataflow Compiler
3. Install Docker and pull Hailo SDK image:
   ```bash
   docker pull hailo/hailo_sdk:latest
   ```

#### OpenVINO
```bash
# Install OpenVINO toolkit
pip install openvino-dev[tensorflow,onnx]
```

#### TensorRT
- Install on target device only (not portable between devices)
- Requires NVIDIA GPU with CUDA
- Download from [NVIDIA Developer](https://developer.nvidia.com/tensorrt)

## Supported Models

The converter supports:
- **YOLOv8**: All variants (n, s, m, l, x)
- **YOLOv9**: YOLOv9 and YOLOv9-MIT
- **YOLOv11**: Latest YOLO versions
- **Custom YOLO**: Any YOLO-based .pt file

### Pre-configured Download Options
- `yolov8n`: Nano model for edge devices
- `yolov9s`: Small model with good accuracy
- `fire_detector_v1`: Pre-trained wildfire detection model

## Output Formats

### 1. ONNX (Universal)
- **Output**: `model.onnx`
- **Use**: Base format for other conversions
- **Deployment**: ONNX Runtime on any platform
- **Performance**: Variable based on hardware

### 2. TensorFlow Lite (Coral TPU)
- **Outputs**:
  - `model_cpu.tflite` - Float16 CPU inference
  - `model_quant.tflite` - INT8 quantized
  - `model_edgetpu.tflite` - Coral TPU optimized
- **Use**: Google Coral USB/PCIe/M.2 accelerators
- **Performance**: 10-30ms inference on Coral

### 3. Hailo (Hailo-8/8L)
- **Outputs**:
  - `model_hailo8.hef` - Hailo-8 (26 TOPS)
  - `model_hailo8l.hef` - Hailo-8L (13 TOPS)
- **Use**: Hailo M.2 AI accelerators
- **Performance**: 15-40ms inference

### 4. OpenVINO (Intel)
- **Outputs**:
  - `model_openvino.xml` - Model structure
  - `model_openvino.bin` - Model weights
- **Use**: Intel CPUs, integrated GPUs, VPUs
- **Performance**: 20-50ms inference

### 5. TensorRT (NVIDIA)
- **Output**: `model_tensorrt.engine`
- **Use**: NVIDIA GPUs (GTX/RTX/Tesla/Jetson)
- **Performance**: 10-30ms inference
- **Note**: Must be compiled on target device

## Detailed Usage

### Custom Calibration Data

For best quantization results, provide your own calibration images:

```bash
# Prepare calibration dataset (500-1000 images recommended)
mkdir calibration_images
cp /path/to/representative/images/*.jpg calibration_images/

# Use in conversion
python convert_model.py model.pt --calibration-data calibration_images/
```

**Calibration Best Practices**:
- Use 500-1000 representative images
- Include all classes you want to detect
- Include various lighting conditions
- Include edge cases (smoke, haze, night)
- Match deployment camera characteristics

### Non-Square Model Export

For optimized inference on specific camera aspect ratios:

```bash
# Export at 320x240 for 4:3 cameras
python convert_model.py model.pt --size 320x240

# Export at 640x360 for 16:9 cameras  
python convert_model.py model.pt --size 640x360
```

### Conversion Examples

#### Fire Detection Model
```bash
# Download calibration data specific to fire/smoke
wget https://example.com/fire_calibration_images.tgz
tar -xzf fire_calibration_images.tgz

# Convert with fire-specific calibration
python convert_model.py fire_yolov8m.pt \
    --name fire_detector_v2 \
    --calibration-data fire_calibration_images/ \
    --size 640
```

#### Multi-Class Wildlife Model
```bash
# Convert wildlife detection model at lower resolution for battery saving
python convert_model.py wildlife_yolov8s.pt \
    --name wildlife_detector \
    --size 416 \
    --formats hailo tflite
```

#### License Plate Recognition
```bash
# High resolution for small object detection
python convert_model.py lpr_yolov8l.pt \
    --name license_plate_v1 \
    --size 1280 \
    --formats tensorrt onnx
```

## Output Structure

```
converted_models/
└── fire_detector_v1/
    ├── fire_detector_v1.onnx                    # Base ONNX model
    ├── fire_detector_v1_cpu.tflite             # CPU inference
    ├── fire_detector_v1_quant.tflite           # Quantized INT8
    ├── fire_detector_v1_edgetpu.tflite         # Coral optimized
    ├── fire_detector_v1_hailo_config.json      # Hailo configuration
    ├── convert_fire_detector_v1_hailo.sh       # Hailo conversion script
    ├── convert_fire_detector_v1_openvino.sh    # OpenVINO script
    ├── convert_fire_detector_v1_tensorrt.py    # TensorRT script
    ├── fire_detector_v1_frigate_config.yml     # Frigate config snippet
    ├── fire_detector_v1_labels.txt             # Class labels
    ├── conversion_summary.json                  # Conversion metadata
    └── README.md                                # Deployment instructions
```

## Deployment Guide

### 1. Raspberry Pi 5 with Hailo-8L

```bash
# On development machine - prepare model
python convert_model.py fire_model.pt --name fire_v1

# Inside Hailo Docker - compile HEF
docker run -it --rm -v $(pwd):/workspace hailo/hailo_sdk:latest
cd /workspace/converted_models/fire_v1
./convert_fire_v1_hailo.sh

# On Raspberry Pi 5 - deploy
scp fire_v1_hailo8l.hef pi@raspberrypi:/opt/frigate/models/
```

### 2. x86 PC with Coral M.2

```bash
# Convert and deploy
python convert_model.py model.pt --formats tflite
docker cp converted_models/model/model_edgetpu.tflite frigate:/models/

# Update Frigate config
docker exec frigate sed -i 's|path:.*|path: /models/model_edgetpu.tflite|' /config/config.yml
docker restart frigate
```

### 3. NVIDIA Jetson/GPU System

```bash
# Copy conversion script to target
scp converted_models/model/convert_model_tensorrt.py jetson:/home/user/

# On Jetson - compile TensorRT engine
python convert_model_tensorrt.py

# Deploy to Frigate
sudo cp model_tensorrt.engine /opt/frigate/models/
```

### 4. Frigate Configuration Integration

The converter generates a config snippet. Merge with your Frigate config:

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
    - person
    - car
    - wildlife
  filters:
    fire:
      min_area: 300
      max_area: 100000
      min_score: 0.6
      threshold: 0.7
    person:
      min_area: 1000
      max_area: 50000
      min_score: 0.5
      threshold: 0.7
```

## Performance Optimization

### Model Size Selection
- **320x320**: Fastest, lowest accuracy, good for many cameras
- **416x416**: Balanced performance
- **640x640**: Default, best balance
- **1280x1280**: Highest accuracy, slow, for critical areas

### Quantization Impact
- **INT8**: 3-4x faster, ~5% accuracy loss, best for edge
- **FP16**: 2x faster, ~1% accuracy loss
- **FP32**: Baseline accuracy, slowest

### Hardware Matching
- **Coral TPU**: Best for low-power, always-on detection
- **Hailo-8**: Best for high-resolution, multi-camera systems
- **Hailo-8L**: Good balance of power and performance
- **TensorRT**: Best for high-accuracy with NVIDIA GPU
- **OpenVINO**: Best for Intel-based systems
- **CPU**: Fallback when no accelerator available

## Troubleshooting

### Common Issues

#### 1. "ModuleNotFoundError: No module named 'ultralytics'"
```bash
# This is OK - the converter uses it in subprocess only
# Install if you want to use it:
pip install ultralytics
```

#### 2. ONNX Export Fails
```bash
# Error: Unsupported ONNX opset version
# Solution: Try different opset
python convert_model.py model.pt --opset 11
```

#### 3. Quantization Accuracy Loss
```bash
# Problem: Model accuracy drops significantly after quantization
# Solution: Use more/better calibration data
python convert_model.py model.pt --calibration-data large_dataset/
```

#### 4. Hailo Compilation Errors
```bash
# Error: Unsupported layer type
# Solution: Check Hailo supported layers documentation
# Or simplify model architecture before training
```

#### 5. TensorRT Version Mismatch
```bash
# Error: TensorRT version mismatch between conversion and deployment
# Solution: Always compile on target device
# Never copy .engine files between different systems
```

#### 6. Edge TPU Compiler Fails
```bash
# Error: Model not fully quantized
# Solution: Ensure all ops are quantization-friendly
# Some custom layers may not be supported
```

### Debug Mode

```bash
# Enable verbose logging
LOG_LEVEL=DEBUG python convert_model.py model.pt

# Test single format conversion
python convert_model.py model.pt --formats onnx

# Skip calibration download
python convert_model.py model.pt --calibration-data /path/to/existing/
```

## Advanced Features

### Batch Processing

```bash
#!/bin/bash
# Convert multiple models
for model in models/*.pt; do
    name=$(basename "$model" .pt)
    python convert_model.py "$model" \
        --name "$name" \
        --output-dir "converted/$name"
done
```

### Custom Conversion Pipeline

```python
#!/usr/bin/env python3
from pathlib import Path
import subprocess

# Custom conversion with specific settings per model
models = {
    'fire_detector': {'size': 640, 'formats': ['hailo', 'coral']},
    'person_detector': {'size': 416, 'formats': ['tensorrt']},
    'wildlife_detector': {'size': 320, 'formats': ['tflite', 'openvino']}
}

for name, config in models.items():
    cmd = [
        'python', 'convert_model.py',
        f'models/{name}.pt',
        '--name', name,
        '--size', str(config['size']),
        '--formats'] + config['formats']
    
    subprocess.run(cmd)
```

### Model Validation

After conversion, validate the model:

```python
# validate_model.py
import numpy as np
import cv2

# For TFLite
import tflite_runtime.interpreter as tflite

def validate_tflite(model_path, image_path):
    # Load model
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    
    # Get results
    output = interpreter.get_tensor(output_details[0]['index'])
    return output

# Validate
result = validate_tflite('model_edgetpu.tflite', 'test_image.jpg')
print(f"Inference successful, output shape: {result.shape}")
```

## Best Practices

### 1. Model Preparation
- Train with diverse data including edge cases
- Use data augmentation for robustness
- Validate on target hardware during development
- Consider model pruning for size reduction

### 2. Calibration Data
- Use images from actual deployment cameras
- Include all lighting conditions (day/night/dusk)
- Include weather variations (clear/rain/fog)
- Balance across all classes
- Minimum 500 images, ideally 1000+

### 3. Format Selection
- Start with ONNX for compatibility testing
- Use hardware-specific formats for production
- Keep CPU version as fallback
- Test each format thoroughly

### 4. Version Control
- Tag models with version numbers
- Document training parameters
- Save conversion settings
- Track performance metrics

### 5. Deployment Testing
```bash
# Test inference speed
time python test_inference.py model_edgetpu.tflite

# Monitor resource usage
htop  # In another terminal during inference

# Check accuracy on test set
python evaluate_model.py model_edgetpu.tflite test_images/
```

## Integration with Wildfire Watch

The converted models integrate seamlessly:

1. **Camera Detector** finds cameras
2. **Model Converter** prepares AI models
3. **Security NVR** loads appropriate model based on hardware
4. **Fire Consensus** validates multi-camera detections
5. **GPIO Trigger** activates sprinklers

### Model Deployment Paths

```bash
# Docker deployment
/opt/frigate/models/
├── fire_v1_edgetpu.tflite    # Primary model
├── fire_v1_cpu.tflite         # Fallback
├── fire_v1_labels.txt         # Class labels
└── config.yml                 # Model config

# Balena deployment  
/data/models/
├── fire_v1_hailo8l.hef       # Hailo model
├── fire_v1_labels.txt         # Class labels
└── metadata.json              # Model info
```

## Performance Benchmarks

Typical inference times on common hardware:

| Hardware | Model Size | Format | Inference Time | Power |
|----------|------------|--------|----------------|--------|
| Coral USB | 640x640 | EdgeTPU | 15-20ms | 2W |
| Hailo-8L | 640x640 | HEF | 20-25ms | 2.5W |
| Hailo-8 | 640x640 | HEF | 10-15ms | 5W |
| RTX 3060 | 640x640 | TensorRT | 8-12ms | 15W |
| Intel i5 | 640x640 | OpenVINO | 25-35ms | 15W |
| RPi5 CPU | 640x640 | TFLite | 200-300ms | 5W |

## License and Legal

- **Converter Script**: MIT License (this tool)
- **Model Licenses**: Inherit from source model
- **Dependencies**: 
  - Core dependencies: MIT/BSD/Apache
  - Optional tools: Some GPL/AGPL (used in subprocess only)
- **Ultralytics**: AGPL (not linked, only used for extraction)

The converter is designed to avoid GPL/AGPL contamination of your code while still leveraging these tools for conversion.

## Contributing

To add support for new architectures:

1. Fork the repository
2. Add detection in `_extract_model_info_external()`
3. Add conversion logic for new format
4. Add tests and documentation
5. Submit pull request

## Support

For issues:
1. Check existing GitHub issues
2. Review this troubleshooting section
3. Enable debug logging
4. Create issue with:
   - Model architecture
   - Conversion command
   - Error messages
   - System information

## Future Enhancements

Planned improvements:
- YOLO-NAS support
- RKNN format for Rockchip NPUs
- Apple CoreML export
- Model optimization (pruning/distillation)
- Automatic performance benchmarking
- Cloud-based conversion service
