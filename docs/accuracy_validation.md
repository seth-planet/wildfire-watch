# Model Accuracy Validation System

## Overview

The Wildfire Watch model converter now includes comprehensive accuracy validation to ensure converted models maintain acceptable performance levels across all deployment formats.

## Key Features

### 1. **Baseline Accuracy Measurement**
- Automatically measures PyTorch model baseline metrics:
  - mAP@50 (mean Average Precision at IoU 0.5)
  - mAP@50-95 (mAP at IoU 0.5-0.95)
  - Precision, Recall, F1 Score
  - Inference time
  - Model size

### 2. **Format-Specific Validation**
Each converted format is validated with appropriate degradation thresholds.

**Important:** INT8 formats automatically use QAT (Quantization-Aware Training) when available for better accuracy:

| Format | mAP@50 Threshold | mAP@50-95 Threshold | Notes |
|--------|------------------|---------------------|-------|
| ONNX | 0.5% | 1.0% | Minimal degradation expected |
| TFLite FP16 | 1.0% | 1.5% | Float16 quantization |
| TFLite INT8 | 2.0% | 3.0% | Standard INT8 quantization |
| TensorRT FP16 | 0.5% | 1.0% | GPU-optimized FP16 |
| TensorRT INT8 | 2.0% | 3.0% | Standard INT8 |
| TensorRT INT8 QAT | 1.0% | 1.5% | Better accuracy with QAT |
| Hailo | 3.0% | 4.0% | Hardware INT8 |
| Hailo QAT | 2.0% | 2.5% | Improved with QAT |

### 3. **Multi-Size Support**
- Test multiple input sizes in one conversion: `["640x640", "416x416", "320x320"]`
- **Default: 640x640** for optimal fire detection accuracy
- **320x320**: For hardware-limited devices (Coral TPU, Raspberry Pi)
- Each size validated independently
- Optimal size recommendations based on hardware capabilities

### 4. **TensorRT Multi-Precision**
Automatically generates and validates:
- **FP16**: Half precision for 2x speedup with minimal accuracy loss
- **INT8**: 8-bit integer for 4x speedup (automatically uses QAT when available)
- **INT8 QAT**: Explicitly Quantization-Aware Training for best INT8 accuracy

When `qat_enabled=True` or the model is QAT-compatible, INT8 conversions automatically use QAT for better accuracy.

### 5. **Comprehensive Reporting**
Generates detailed accuracy report with:
- Baseline metrics
- Per-format accuracy measurements
- Degradation percentages
- Speed improvements
- Size reductions
- Pass/Fail status for each format

## Usage

### Basic Usage with Validation
```python
from convert_model import EnhancedModelConverter

converter = EnhancedModelConverter(
    model_path="yolov8n.pt",
    output_dir="converted_models",
    model_name="yolov8n",
    model_size=["640x640", "416x416"],  # Multiple sizes
    qat_enabled=True  # Enable QAT optimizations
)

# Convert with validation
results = converter.convert_all(
    formats=['onnx', 'tflite', 'tensorrt'],
    validate=True,  # Enable accuracy validation
    benchmark=True
)
```

### Command Line Usage
```bash
# Convert with multiple sizes and validation
python3.12 convert_model.py yolov8n.pt \
    --output converted_models \
    --sizes 640 416 320 \
    --formats onnx tflite tensorrt \
    --validate \
    --qat
```

## Accuracy Validation Process

1. **Baseline Measurement**
   - Load PyTorch model
   - Run validation on COCO dataset (or subset)
   - Record baseline metrics

2. **Per-Format Validation**
   - Convert to target format
   - Run same validation dataset
   - Measure accuracy metrics
   - Calculate degradation

3. **Threshold Checking**
   - Compare degradation to acceptable thresholds
   - Mark as PASSED or FAILED
   - Log warnings for excessive degradation

4. **Report Generation**
   - Create comprehensive markdown report
   - Include all metrics and comparisons
   - Provide recommendations

## Example Output

```
Model Accuracy Validation Report
================================

## Baseline (PyTorch)
- mAP@50: 0.850
- mAP@50-95: 0.650
- Precision: 0.900
- Recall: 0.850
- F1 Score: 0.870
- Inference Time: 50.0ms
- Model Size: 10.0MB

## Converted Models

### ONNX_640X640
- mAP@50: 0.846
- mAP@50-95: 0.644
- Inference Time: 40.0ms
- Model Size: 10.0MB

  **Comparison to baseline:**
  - mAP@50 degradation: 0.47%
  - mAP@50-95 degradation: 0.92%
  - Speed improvement: 1.3x
  - Size reduction: 0.0%
  - **Status:** ✅ PASSED

### TENSORRT_INT8_QAT_640X640
- mAP@50: 0.842
- mAP@50-95: 0.641
- Inference Time: 12.0ms
- Model Size: 5.0MB

  **Comparison to baseline:**
  - mAP@50 degradation: 0.94%
  - mAP@50-95 degradation: 1.38%
  - Speed improvement: 4.2x
  - Size reduction: 50.0%
  - **Status:** ✅ PASSED
```

## Best Practices

1. **Always validate when deploying to production**
   ```python
   results = converter.convert_all(validate=True)
   ```

2. **Use QAT for INT8 deployments**
   - Start with QAT-trained models (e.g., `yolov8n_qat.pt`)
   - Enable QAT optimizations: `qat_enabled=True`
   - Expect 1-2% better accuracy than standard INT8

3. **Test multiple sizes for your use case**
   - High accuracy: 640x640
   - Balanced: 416x416
   - Fast/embedded: 320x320 or 224x224

4. **Monitor degradation thresholds**
   - Adjust thresholds in `AccuracyValidator.THRESHOLDS` if needed
   - Stricter thresholds for critical applications
   - Relaxed thresholds for edge devices

## Troubleshooting

### Validation Fails
- Check if validation dataset is available
- Ensure sufficient GPU memory for validation
- Try with smaller validation subset

### High Degradation
- Use QAT-trained models
- Try larger model size (e.g., YOLOv8s instead of YOLOv8n)
- Adjust quantization calibration dataset
- Use FP16 instead of INT8

### Validation Takes Too Long
- Use smaller validation dataset
- Reduce number of validation images
- Use `validate=False` for quick testing

## Supported Models

The conversion system supports various YOLO architectures and RT-DETR models:

### YOLO Models
- **YOLOv8**: `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x`
- **YOLOv9**: `yolov9t`, `yolov9s`, `yolov9m`, `yolov9c`, `yolov9e`
- **YOLOv9-MIT**: `yolov9mit_s`, `yolov9mit_m` (MIT licensed variants)
- **YOLO-NAS**: `yolo_nas_s`, `yolo_nas_m`, `yolo_nas_l`
- **RT-DETR**: `rtdetr_l`, `rtdetr_x`
- **RT-DETRv2**: `rtdetrv2_s`, `rtdetrv2_m`, `rtdetrv2_l`, `rtdetrv2_x`

### Test Script

Use the provided test script to validate multiple models:

```bash
python3.12 scripts/test_model_conversions.py
```

This tests:
- `yolov8n` - YOLOv8 nano (fastest)
- `yolov9t` - YOLOv9 tiny
- `yolov9mit_s` - YOLOv9-MIT small
- `yolo_nas_s` - YOLO-NAS small
- `rtdetrv2_s` - RT-DETRv2 small

## API Reference

### AccuracyValidator
```python
validator = AccuracyValidator(
    val_dataset_path=Path("path/to/coco/val"),
    confidence_threshold=0.25,
    iou_threshold=0.45
)

# Validate model
metrics = validator.validate_pytorch_model(model_path)
validator.set_baseline(metrics)

# Check degradation
is_acceptable, degradation = validator.check_degradation(
    new_metrics, 
    format_type='tensorrt_int8'
)
```

### AccuracyMetrics
```python
@dataclass
class AccuracyMetrics:
    mAP50: float
    mAP50_95: float
    precision: float
    recall: float
    f1_score: float
    inference_time_ms: float
    model_size_mb: float
```

## Integration with Frigate

The accuracy validation ensures models meet Frigate's requirements:
- Accurate fire/smoke detection
- Fast inference for real-time processing
- Optimal model size for edge deployment

Generated Frigate configs automatically select best model based on:
- Hardware capabilities
- Accuracy requirements
- Performance targets