# Wildfire Watch YOLO Model Deployment Guide

## Validated Models Available

### 640x640 Model
- **File**: `yolo8l_wildfire_640x640.onnx`
- **Path**: `converted_models/640x640/yolo8l_wildfire_640x640.onnx`
- **Input Shape**: [1, 3, 640, 640]
- **Output Shapes**: [(1, 36, 8400)]
- **Recommended For**: Primary detection (high accuracy)

### 320x320 Model
- **File**: `yolo8l_wildfire_320x320.onnx`
- **Path**: `converted_models/320x320/yolo8l_wildfire_320x320.onnx`
- **Input Shape**: [1, 3, 320, 320]
- **Output Shapes**: [(1, 36, 2100)]
- **Recommended For**: Edge devices (Raspberry Pi, Coral TPU)


## Frigate Configuration

### Option 1: High Accuracy (640x640) - Recommended
```yaml
model:
  path: /models/yolo8l_wildfire_640x640.onnx
  input_tensor: nchw
  input_pixel_format: bgr
  width: 640
  height: 640

detectors:
  wildfire:
    type: onnx
    device: auto

cameras:
  default:
    detect:
      width: 640
      height: 640
    objects:
      filters:
        fire:
          min_area: 1000
          threshold: 0.7
        smoke:
          min_area: 1500
          threshold: 0.6
        person:
          min_area: 2000
          max_area: 100000
        vehicle:
          min_area: 5000
          max_area: 100000
```

### Option 2: Edge Devices (320x320)
```yaml
model:
  path: /models/yolo8l_wildfire_320x320.onnx
  input_tensor: nchw
  input_pixel_format: bgr
  width: 320
  height: 320

detectors:
  wildfire:
    type: onnx
    device: auto

cameras:
  default:
    detect:
      width: 320
      height: 320
    objects:
      filters:
        fire:
          min_area: 500
          threshold: 0.7
        smoke:
          min_area: 750
          threshold: 0.6
        person:
          min_area: 1000
          max_area: 50000
        vehicle:
          min_area: 2500
          max_area: 50000
```

## Deployment Steps

### 1. Copy Model Files
```bash
# For high accuracy (recommended)
cp converted_models/640x640/yolo8l_wildfire_640x640.onnx /path/to/frigate/models/

# For edge devices
cp converted_models/320x320/yolo8l_wildfire_320x320.onnx /path/to/frigate/models/
```

### 2. Update Frigate Configuration
Add the appropriate configuration above to your `config.yml` file.

### 3. Restart Frigate
```bash
docker restart frigate
```

## Model Classes
The models detect the following classes:
- **fire**: Active flames
- **smoke**: Smoke plumes  
- **person**: Human presence (for context)
- **vehicle**: Cars, trucks (for context)

## Performance Notes
- **640x640**: ~20-50ms inference time, better accuracy
- **320x320**: ~10-25ms inference time, suitable for edge devices
- Models are optimized for wildfire detection scenarios

## Troubleshooting
1. **High CPU usage**: Use 320x320 model or enable hardware acceleration
2. **No detections**: Lower the threshold values in object filters
3. **Too many false positives**: Increase min_area values

## Hardware Recommendations
- **640x640 model**: Requires GPU or powerful CPU
- **320x320 model**: Works well on Raspberry Pi 4+, Coral TPU
- Both models support hardware acceleration when available
