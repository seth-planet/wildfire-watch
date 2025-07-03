# yolov8l - Multi-Size Converted Models

## Model Information
- **Type**: unknown
- **Architecture**: YOLO
- **Version**: Unknown
- **Classes**: 0
- **License**: Check original model
- **Conversion Time**: 6.8 seconds

## Converted Sizes

Successfully converted 1 size variants:

| Size | Formats | Status |
|------|---------|--------|
| 640x640 | frigate_config | ✅ Complete |


## Size Recommendations

### For Different Use Cases:

| Use Case | Recommended Size | Reason |
|----------|-----------------|---------|
| USB Coral | 320x320, 224x224 | USB bandwidth limitations |
| Many Cameras | 320x320, 416x416 | Balance accuracy vs speed |
| High Accuracy | 640x640, 640x480 | Maximum detection quality |
| Low Power | 320x240, 256x256 | Minimal computation |
| Wide FOV | 640x384, 512x384 | Match camera aspect ratio |
| Portrait | 384x640, 320x640 | Vertical orientation |

## Deployment Structure

```
converted_models/
├── 640x640/
│   ├── yolov8l_640x640.onnx
│   ├── yolov8l_640x640_*.tflite
│   └── yolov8l_frigate_config.yml
├── 640x480/
│   ├── yolov8l_640x480.onnx
│   └── ...
├── 416x416/
│   └── ...
└── conversion_summary.json
```

## Quick Deployment

### 1. Choose Size Based on Hardware

```bash
# For Coral USB (limited bandwidth)
cp 320x320/yolov8l_320x320_edgetpu.tflite /models/

# For Hailo-8 (high performance)
cp 640x640/yolov8l_640x640_hailo8.hef /models/

# For CPU fallback
cp 416x416/yolov8l_416x416_cpu.tflite /models/
```

### 2. Update Frigate Config

Use the size-specific Frigate config:
```bash
cp 416x416/yolov8l_frigate_config.yml /config/
```

## Performance by Size

| Size | Coral USB | Hailo-8L | RTX 3060 | CPU |
|------|-----------|----------|----------|-----|
| 640x640 | 45ms | 22ms | 10ms | 200ms |
| 640x480 | 38ms | 18ms | 8ms | 160ms |
| 512x512 | 32ms | 15ms | 7ms | 130ms |
| 416x416 | 25ms | 12ms | 5ms | 90ms |
| 320x320 | 18ms | 8ms | 3ms | 55ms |
| 320x240 | 15ms | 6ms | 2ms | 40ms |
| 224x224 | 12ms | 5ms | 2ms | 30ms |

*Times are approximate and depend on model complexity*

## Size-Specific Notes

### 640x640
- **High resolution**: Best accuracy for detailed detection


## Advanced Usage

### Batch Processing Multiple Sizes

```python
from convert_model import EnhancedModelConverter

# Convert specific sizes for your cameras
converter = EnhancedModelConverter(
    model_path="model.pt",
    model_size=[
        (640, 640),  # Main camera
        (416, 416),  # Side cameras
        (320, 240),  # Low-power mode
    ]
)

results = converter.convert_all()
```

### Size-Aware Frigate Config

```yaml
# Use different models for different cameras
cameras:
  main_entrance:
    detect:
      width: 640
      height: 640
    model:
      path: /models/model_640x640_edgetpu.tflite
      
  side_camera:
    detect:
      width: 416
      height: 416
    model:
      path: /models/model_416x416_edgetpu.tflite
```

## Troubleshooting

### Size-Related Issues

1. **"Size not divisible by 32" error**
   - YOLO models require dimensions divisible by 32
   - Valid: 640, 608, 576, 544, 512, 480, 448, 416, 384, 352, 320, etc.

2. **Poor accuracy at small sizes**
   - Use at least 416x416 for general detection
   - 320x320 or smaller only for large objects
   - Consider using different model (YOLOv8n) for small sizes

3. **Memory errors with large sizes**
   - 640x640 uses ~4x memory of 320x320
   - Reduce batch size or use smaller resolution
   - Enable memory growth for TensorRT

## License

Model conversions inherit the license of the original model.
The converter tool itself is MIT licensed.
