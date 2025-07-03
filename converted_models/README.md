# yolo8l_rev5_quantized - Multi-Size Converted Models

## Model Information
- **Type**: yolov8
- **Architecture**: YOLOv8
- **Version**: yolov8l.pt
- **Classes**: 32
- **License**: Check original model
- **Conversion Time**: 37.8 seconds

## Converted Sizes

Successfully converted 2 size variants:

| Size | Formats | Status |
|------|---------|--------|
| 640x640 | frigate_config | ✅ Complete |
| 320x320 | frigate_config | ✅ Complete |


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
│   ├── yolo8l_rev5_quantized_640x640.onnx
│   ├── yolo8l_rev5_quantized_640x640_*.tflite
│   └── yolo8l_rev5_quantized_frigate_config.yml
├── 640x480/
│   ├── yolo8l_rev5_quantized_640x480.onnx
│   └── ...
├── 416x416/
│   └── ...
└── conversion_summary.json
```

## Quick Deployment

### 1. Choose Size Based on Hardware

```bash
# For Coral USB (limited bandwidth)
# Note: Coral TPU requires Python 3.8 for tflite_runtime
cp 320x320/yolo8l_rev5_quantized_320x320_edgetpu.tflite /models/

# For Hailo-8 (high performance)
cp 640x640/yolo8l_rev5_quantized_640x640_hailo8.hef /models/

# For CPU fallback
cp 416x416/yolo8l_rev5_quantized_416x416_cpu.tflite /models/
```

### 2. Update Frigate Config

Use the size-specific Frigate config:
```bash
cp 416x416/yolo8l_rev5_quantized_frigate_config.yml /config/
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

### 640x640 (Recommended)
- **Optimal resolution**: Best balance of accuracy and performance
- **Use cases**: Primary fire detection, high-risk areas
- **Hardware**: Works well on Hailo, GPU, and modern CPUs
- **Validation Results**: Highest accuracy for fire/smoke detection

### 320x320 (Hardware Limited)
- **Low-power option**: For Raspberry Pi or battery-powered devices
- **Use cases**: Secondary cameras, motion-triggered recording
- **Hardware**: Optimized for Coral TPU and edge devices
- **Note**: Still effective for fire detection at shorter distances


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

## Training Custom YOLO-NAS Models

### Prerequisites

YOLO-NAS training requires Python 3.10 for super-gradients compatibility:

```bash
# Install Python 3.10 if not already available
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev

# Install super-gradients and dependencies
python3.10 -m pip install super-gradients==3.6.0 torch torchvision

# For GPU training (recommended)
python3.10 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Dataset Preparation

1. **Organize your dataset in YOLO format**:
   ```
   dataset/
   ├── train/
   │   ├── images/
   │   │   ├── img001.jpg
   │   │   └── ...
   │   └── labels/
   │       ├── img001.txt
   │       └── ...
   └── validation/
       ├── images/
       └── labels/
   ```

2. **Label format** (each line in .txt file):
   ```
   class_id x_center y_center width height
   ```
   Values are normalized to [0, 1].

3. **Create a names.yaml file**:
   ```yaml
   names:
     0: Person
     1: Bicycle
     2: Car
     # ... other classes ...
     26: Fire
     27: Smoke
   ```

### Training Commands

#### 1. Quick Start with Training Script

```bash
# Train from scratch (no pretrained weights - avoids copyright issues)
python3.10 converted_models/train_custom_yolo_nas.py \
  --dataset_path /path/to/dataset \
  --no_pretrained \
  --epochs 100 \
  --batch_size 16 \
  --experiment_name wildfire_scratch

# Train with pretrained weights (requires internet connection)
python3.10 converted_models/train_custom_yolo_nas.py \
  --dataset_path /path/to/dataset \
  --pretrained \
  --epochs 100 \
  --batch_size 16 \
  --mixed_precision \
  --experiment_name wildfire_pretrained
```

**Important Notes:**
- Use `--no_pretrained` to train from scratch (recommended to avoid copyright issues)
- Use `--pretrained` only if you accept the COCO dataset licensing terms
- The script automatically detects both standard and Roboflow dataset formats

#### 2. Manual Configuration with Unified Trainer

```bash
# Create a configuration file
cat > config.yaml << EOF
model:
  architecture: yolo_nas_s  # Options: yolo_nas_s, yolo_nas_m, yolo_nas_l
  num_classes: 32
  input_size: [640, 640]
  pretrained_weights: null  # Train from scratch (no copyright issues)

dataset:
  data_dir: /path/to/your/dataset
  train_split: train
  val_split: validation
  validate_labels: true

training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.001
  warmup_epochs: 3
  mixed_precision: true
  workers: 8
  lr_scheduler: cosine
  lr_decay_factor: 0.1

validation:
  interval: 1
  conf_threshold: 0.25
  iou_threshold: 0.45
  max_predictions: 300

output_dir: ./output
experiment_name: wildfire_yolo_nas
log_level: INFO
EOF

# Run training
python3.10 converted_models/unified_yolo_trainer.py config.yaml
```

#### 3. Advanced Training with Custom Parameters

```bash
# Create advanced config
cat > advanced_config.yaml << EOF
model:
  architecture: yolo_nas_l  # Larger model for better accuracy
  num_classes: 32
  input_size: [640, 640]
  pretrained_weights: "coco"

dataset:
  data_dir: /path/to/dataset
  train_split: train
  val_split: validation
  validate_labels: true
  cache_annotations: true  # Faster loading after first epoch

training:
  epochs: 300
  batch_size: 32
  learning_rate: 0.0001
  warmup_epochs: 5
  mixed_precision: true
  workers: 16
  lr_scheduler: cosine
  lr_decay_factor: 0.01
  
  # Advanced parameters
  weight_decay: 0.0001
  momentum: 0.9
  accumulate_grad_batches: 4  # Gradient accumulation
  
  # Augmentations
  augmentations:
    - RandomHSV: {hgain: 0.015, sgain: 0.7, vgain: 0.4}
    - RandomFlip: {prob: 0.5}
    - RandomRotate: {degrees: 10}

validation:
  interval: 1
  conf_threshold: 0.25
  iou_threshold: 0.45
  max_predictions: 300
  
  # Metrics
  metrics:
    - mAP@0.5
    - mAP@0.5:0.95
    - Precision
    - Recall

training_hyperparams:
  loss: PPYoloELoss
  optimizer: AdamW
  ema: true
  ema_decay: 0.9999
  
  # Multi-scale training
  multiscale: [320, 640]
  
  # Checkpointing
  save_checkpoint_interval: 10
  save_best_checkpoint: true
  
output_dir: ./experiments
experiment_name: wildfire_yolo_nas_advanced
log_level: DEBUG
EOF

python3.10 converted_models/unified_yolo_trainer.py advanced_config.yaml
```

### Known Issues and Solutions

#### 1. CUDA Index Error Fix

If you encounter `IndexError: too many indices for tensor` or similar CUDA errors, the issue is with the dataloader's target format. The fix has been implemented in `fixed_yolo_nas_collate.py`:

```python
# The fix ensures targets are in the correct format [N, 6]
# where each row is [image_idx, class_id, cx, cy, w, h]
from fixed_yolo_nas_collate import wrap_dataloader_with_fixed_collate

# Wrap your dataloaders
train_loader = wrap_dataloader_with_fixed_collate(train_loader, num_classes)
val_loader = wrap_dataloader_with_fixed_collate(val_loader, num_classes)
```

#### 2. Image Format Requirements

YOLO-NAS expects:
- **Format**: CHW (Channels, Height, Width)
- **Dtype**: float32, normalized to [0, 1]
- **Input size**: Must be divisible by 32

The dataloader handles conversion automatically from common formats (HWC uint8).

### Monitoring Training

Training progress includes:
- Loss values: `PPYoloELoss/loss`, `loss_cls`, `loss_dfl`, `loss_iou`
- Learning rate tracking
- GPU memory usage
- Validation metrics every epoch

### Post-Training

After training completes:

1. **Best model location**:
   ```
   output/checkpoints/experiment_name/ckpt_best.pth
   ```

2. **Convert to deployment formats**:
   ```bash
   # Convert trained model for deployment
   python3.10 converted_models/convert_model.py \
     --model_path output/checkpoints/experiment_name/ckpt_best.pth \
     --model_type yolo_nas \
     --output_dir converted_models/custom_model
   ```

3. **Deploy with Frigate**:
   ```bash
   # Copy converted model
   cp converted_models/custom_model/model_640x640.onnx /models/
   
   # Update Frigate config
   cp converted_models/custom_model/frigate_config.yml /config/
   ```

### Performance Tips

1. **GPU Training**: Use largest batch size that fits in memory
2. **Multi-GPU**: Set `distributed_training: true` in config
3. **Mixed Precision**: Enable for 2x speedup with minimal accuracy loss
4. **Gradient Accumulation**: Use when batch size is limited by memory
5. **Cache Annotations**: Speeds up data loading after first epoch

### Validation and Testing

After training, validate your model:

```bash
# Run accuracy validation
python3.10 scripts/demo_accuracy_validation.py \
  --model_path output/checkpoints/best_model.pth \
  --test_dataset /path/to/test/dataset \
  --confidence_threshold 0.5
```

### INT8 Quantization for Edge Deployment

YOLO-NAS is specifically designed for efficient INT8 quantization, making it ideal for edge devices like Coral TPU. Here's what you need to know:

#### Understanding Quantization Options

1. **YOLO-NAS Architecture is Inherently Quantization-Friendly**
   - Uses quantization-aware blocks (no complex activations)
   - Designed to minimize accuracy loss when converted to INT8
   - Can achieve <1 mAP drop with post-training quantization

2. **Quantization-Aware Training (QAT) vs Post-Training Quantization (PTQ)**
   - **QAT**: Simulates INT8 quantization during training (more complex, slightly better accuracy)
   - **PTQ**: Quantizes after training (simpler, still excellent results with YOLO-NAS)
   - For YOLO-NAS, PTQ often suffices due to its architecture

#### Recommended Approach: Standard Training + Post-Training Quantization

```bash
# Step 1: Train YOLO-NAS normally
python3.10 converted_models/train_yolo_nas_with_qat.py \
  --dataset_path /media/seth/SketchScratch/fiftyone/train_yolo \
  --no_pretrained \
  --epochs 100 \
  --batch_size 16 \
  --experiment_name wildfire_int8

# Step 2: Convert to INT8 after training (requires calibration data)
python3.10 converted_models/convert_model.py \
  --model_path output/checkpoints/wildfire_int8/ckpt_best.pth \
  --model_type yolo_nas \
  --output_formats tflite \
  --calibration_dir /path/to/calibration/images
```

#### Advanced: True Quantization-Aware Training (QAT)

For maximum accuracy with INT8 (adds complexity for ~0.5% improvement):

```bash
# Note: This requires super-gradients QATTrainer setup
python3.10 converted_models/train_yolo_nas_qat.py \
  --dataset_path /media/seth/SketchScratch/fiftyone/train_yolo \
  --no_pretrained \
  --epochs 100 \
  --batch_size 16 \
  --experiment_name wildfire_true_qat
```

### Working Example

Here's a tested example that trains from scratch on the wildfire dataset:

```bash
# Basic training from scratch
python3.10 converted_models/train_custom_yolo_nas.py \
  --dataset_path /media/seth/SketchScratch/fiftyone/train_yolo \
  --no_pretrained \
  --epochs 100 \
  --batch_size 16 \
  --experiment_name wildfire_yolo_nas_scratch \
  --mixed_precision

# For edge deployment (YOLO-NAS is inherently quantization-friendly)
python3.10 converted_models/train_custom_yolo_nas.py \
  --dataset_path /media/seth/SketchScratch/fiftyone/train_yolo \
  --no_pretrained \
  --epochs 100 \
  --batch_size 16 \
  --experiment_name wildfire_edge \
  --mixed_precision
```

These commands:
- Train from scratch (no copyright issues with pretrained weights)
- Use the wildfire dataset with 283,631 training images
- Enable mixed precision for faster training
- Save models to `./output/checkpoints/[experiment_name]/`

### Troubleshooting

1. **Out of Memory**: Reduce batch_size or input_size
2. **Slow Training**: Enable mixed_precision, increase workers
3. **Poor Accuracy**: Increase epochs, try larger model (yolo_nas_m or yolo_nas_l)
4. **Overfitting**: Add more augmentations, use weight_decay
5. **Network Errors**: If pretrained weight download fails, use `--no_pretrained`
6. **Dataset Format**: The script auto-detects standard and Roboflow formats
