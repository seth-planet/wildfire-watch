# Coral TPU Model Conversion Guide

## Overview

This guide explains how to convert YOLO models to Google Coral Edge TPU format for accelerated inference on edge devices.

## Conversion Scripts

We provide two main conversion scripts, each optimized for different input formats:

### 1. `convert_yolo_to_coral.py` - For PyTorch Models

**Purpose**: Converts PyTorch (.pt) models through the full pipeline to Edge TPU format.

**Pipeline**: PyTorch → ONNX → TensorFlow → TFLite → Edge TPU

**Usage**:
```bash
python3.8 scripts/convert_yolo_to_coral.py yolov8n.pt --size 320
python3.8 scripts/convert_yolo_to_coral.py yolov8l.pt --size 416 --output converted_models/coral
```

**Features**:
- Automatic ONNX export from PyTorch
- Downloads wildfire calibration dataset
- INT8 quantization with representative dataset
- Edge TPU compilation
- Hardware testing if available

### 2. `convert_to_coral.py` - For ONNX Models

**Purpose**: Converts ONNX models (YOLOv8/YOLO-NAS) to Edge TPU format with enhanced compatibility.

**Pipeline**: ONNX → TensorFlow → TFLite → Edge TPU

**Usage**:
```bash
python3.8 scripts/convert_to_coral.py model.onnx --size 320
python3.8 scripts/convert_to_coral.py yolo_nas.onnx --size 416 --output converted_models/coral
```

**Features**:
- Uses onnx2tf for better handling of complex operations
- Multiple conversion strategies for compatibility
- Handles SplitV operations that often cause issues
- Supports both YOLOv8 and YOLO-NAS architectures

## Prerequisites

### Python Version
**IMPORTANT**: Both scripts require Python 3.8 for Coral TPU compatibility.

```bash
# Check Python version
python3.8 --version

# Install if needed
sudo apt-get install python3.8 python3.8-pip
```

### Required Packages

```bash
# Core dependencies
python3.8 -m pip install tensorflow==2.13.0
python3.8 -m pip install tflite-runtime
python3.8 -m pip install onnx-tf

# For enhanced ONNX conversion
python3.8 -m pip install onnx2tf

# For PyTorch models
python3.8 -m pip install torch torchvision
python3.8 -m pip install ultralytics  # For YOLOv8
```

### Edge TPU Compiler

```bash
# Install Edge TPU compiler
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install edgetpu-compiler
```

## Conversion Process

### Step 1: Choose the Right Script

- Have a `.pt` file? Use `convert_yolo_to_coral.py`
- Have a `.onnx` file? Use `convert_to_coral.py`
- Have YOLO-NAS model? Export to ONNX first, then use `convert_to_coral.py`

### Step 2: Select Model Size

Edge TPU works best with specific input sizes:
- **320x320**: Best for edge devices, fastest inference
- **416x416**: Balanced accuracy and speed
- **640x640**: Best accuracy but slower (may not fully utilize Edge TPU)

### Step 3: Run Conversion

```bash
# Example: Convert YOLOv8n PyTorch model
python3.8 scripts/convert_yolo_to_coral.py yolov8n.pt --size 320

# Example: Convert YOLO-NAS ONNX model
python3.8 scripts/convert_to_coral.py yolo_nas_s.onnx --size 320
```

### Step 4: Verify Output

The conversion creates several files:
- `model_320_int8.tflite` - Quantized TFLite model
- `model_320_int8_edgetpu.tflite` - Edge TPU compiled model
- `model_320_int8_edgetpu.log` - Compilation statistics

Check the log file for Edge TPU utilization:
```bash
cat model_320_int8_edgetpu.log | grep "Number of operations"
```

## Performance Expectations

### Model Complexity vs Edge TPU Utilization

| Model | Edge TPU Ops | CPU Ops | Inference Time |
|-------|--------------|---------|----------------|
| YOLOv8n | ~90% | ~10% | 15-25ms |
| YOLOv8s | ~85% | ~15% | 20-30ms |
| YOLOv8m | ~75% | ~25% | 30-50ms |
| YOLOv8l | ~60% | ~40% | 100-200ms |
| YOLOv8x | ~50% | ~50% | 200-500ms |

### Optimization Tips

1. **Use smaller models**: YOLOv8n/s for real-time applications
2. **Reduce input size**: 320x320 is optimal for Edge TPU
3. **Batch size 1**: Edge TPU doesn't benefit from batching
4. **INT8 quantization**: Required for Edge TPU, slight accuracy loss

## Troubleshooting

### Common Issues

1. **SplitV Operation Error**
   - Solution: Use `convert_to_coral.py` with onnx2tf
   - The script handles this automatically

2. **Dimension Mismatch**
   - Ensure model input size matches configuration
   - Check with: `python3.8 -c "import tensorflow as tf; interpreter = tf.lite.Interpreter('model.tflite'); interpreter.allocate_tensors(); print(interpreter.get_input_details())"`

3. **Low Edge TPU Utilization**
   - Complex models may not fully utilize Edge TPU
   - Consider using smaller model variants

4. **Python Version Issues**
   - Must use Python 3.8 for tflite_runtime compatibility
   - Don't use Python 3.12 for Coral TPU work

### Testing on Hardware

```python
# Test script
from pycoral.utils.edgetpu import make_interpreter
import numpy as np
import time

interpreter = make_interpreter('model_edgetpu.tflite')
interpreter.allocate_tensors()

# Run inference
input_data = np.random.randint(0, 255, size=(1, 320, 320, 3), dtype=np.uint8)
interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)

start = time.perf_counter()
interpreter.invoke()
print(f"Inference time: {(time.perf_counter() - start) * 1000:.2f}ms")
```

## Integration with Frigate

When using with Frigate NVR:

1. Place model in Frigate config directory
2. Configure detector:
   ```yaml
   detectors:
     coral:
       type: edgetpu
       device: pci:0  # or usb:0 for USB Coral
   
   model:
     path: /config/model_320_edgetpu.tflite
     input_tensor: nhwc
     input_pixel_format: rgb
     width: 320
     height: 320
   ```

## Best Practices

1. **Always test converted models** before deployment
2. **Keep calibration data representative** of your use case
3. **Monitor Edge TPU temperature** during extended use
4. **Use multiple Edge TPUs** for parallel processing if needed
5. **Profile inference time** to ensure real-time performance

## Summary

- Two specialized scripts for different input formats
- Python 3.8 required for all Coral TPU work
- Smaller models (YOLOv8n/s) work best on Edge TPU
- INT8 quantization is mandatory but maintains good accuracy
- Edge TPU excels at real-time inference on edge devices