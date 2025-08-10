# YOLOv8 Frigate Setup Instructions

This custom detector enables YOLOv8 models with [1, 36, 8400] output format to work with Frigate.

## Installation Steps

### 1. Copy the Detector

Copy `yolov8_frigate.py` to your Frigate configuration directory:

```bash
# If using Docker
docker cp /home/seth/wildfire-watch/converted_models/frigate_yolo_solution/yolov8_frigate.py frigate:/config/custom_detectors/yolov8_frigate.py

# If using direct installation
cp /home/seth/wildfire-watch/converted_models/frigate_yolo_solution/yolov8_frigate.py /path/to/frigate/config/custom_detectors/
```

### 2. Copy Your Model

Copy your YOLOv8 TFLite model:

```bash
# If using Docker
docker cp /home/seth/wildfire-watch/converted_models/frigate_models/yolo8l_fire_640x640_frigate.tflite frigate:/models/yolo8l_fire_640x640_frigate.tflite

# If using direct installation  
cp /home/seth/wildfire-watch/converted_models/frigate_models/yolo8l_fire_640x640_frigate.tflite /path/to/frigate/models/
```

### 3. Update Frigate Configuration

Edit your `frigate.yml` to use the custom detector. See `frigate_config_example.yml` for a complete example.

Key configuration:
```yaml
detectors:
  coral:
    type: yolov8  # This matches the type_key in our detector
    device: usb
    model:
      path: /models/yolo8l_fire_640x640_frigate.tflite
```

### 4. Restart Frigate

```bash
docker restart frigate
# or
systemctl restart frigate
```

### 5. Verify Operation

Check Frigate logs:
```bash
docker logs frigate | grep YOLOv8
```

You should see:
- "YOLOv8 detector initialized with model: /models/..."
- "Input shape: [1, 640, 640, 3]"
- "Output shape: [1, 36, 8400]"

## Testing the Detector

You can test the detector standalone:

```bash
# Test with your model
python3 yolov8_frigate.py /home/seth/wildfire-watch/converted_models/frigate_models/yolo8l_fire_640x640_frigate.tflite
```

## Troubleshooting

1. **Import Error**: If `tflite_runtime` is not found, the detector will fall back to `tensorflow.lite`.

2. **Model Not Found**: Ensure the model path in frigate.yml matches the actual location in the container.

3. **No Detections**: Try lowering the confidence threshold in the model config.

4. **Performance Issues**: 
   - Reduce detection fps in camera config
   - Increase detection interval
   - Use hardware acceleration (Coral TPU)

## Model Requirements

Your YOLOv8 model must:
- Accept input shape [1, 640, 640, 3] as UINT8
- Output shape [1, 36, 8400] where 36 = 4 bbox + 32 classes
- Be in TFLite format (quantized or float32)

## Fire Detection Optimization

For fire detection, consider these settings:
- Lower confidence threshold (0.2-0.3) for early detection
- Higher max_detections to catch multiple fire sources
- Adjust IoU threshold based on fire/smoke overlap patterns
