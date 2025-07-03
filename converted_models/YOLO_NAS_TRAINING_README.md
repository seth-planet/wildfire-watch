# YOLO-NAS Training and Deployment Pipeline

This directory contains scripts to train a custom YOLO-NAS model on your COCO dataset and deploy it as an INT8 QAT TensorRT model in Frigate.

## Overview

The pipeline consists of three phases:
1. **Training** (48-72 hours): Train YOLO-NAS-S on your custom dataset
2. **Conversion** (2-4 hours): Convert to optimized TensorRT INT8 QAT format
3. **Deployment** (15 minutes): Install in Frigate NVR service

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (16GB+ recommended)
- **Storage**: 50GB+ free disk space
- **Memory**: 16GB+ RAM recommended

### Software Requirements
- CUDA 11.8+ with cuDNN
- **Python 3.10** (required for super-gradients compatibility)
- Docker (for Frigate deployment)

### Dataset Requirements
- COCO format dataset at `~/fiftyone/train_yolo/`
- Minimum 1000+ images for effective training
- Proper `dataset.yaml` configuration

## Quick Start

### Option 1: Complete Pipeline (Recommended)
Run everything in one command:
```bash
cd converted_models
python3.10 complete_yolo_nas_pipeline.py
```

### Option 2: Manual Step-by-Step

#### Step 1: Training Only
```bash
cd converted_models
python3.10 train_yolo_nas.py
```

#### Step 2: Conversion and Deployment
```bash
cd converted_models
python3.10 deploy_trained_yolo_nas.py
```

## Expected Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| Setup | 30 minutes | Install dependencies, prepare dataset |
| Training | 48-72 hours | Train YOLO-NAS-S with QAT |
| Conversion | 2-4 hours | Export to TensorRT INT8 |
| Deployment | 15 minutes | Install in Frigate |

## Dataset Configuration

Your dataset should be structured as:
```
~/fiftyone/train_yolo/
├── dataset.yaml          # Class definitions
├── images/
│   ├── train/            # Training images
│   └── validation/       # Validation images
└── labels/
    ├── train/            # Training labels (YOLO format)
    └── validation/       # Validation labels
```

The `dataset.yaml` should contain:
```yaml
names:
  0: Person
  1: Car
  # ... your classes
  26: Fire
  # ... more classes
path: /media/seth/SketchScratch/fiftyone/train_yolo
train: ./images/train/
validation: ./images/validation/
```

## Training Configuration

The training uses these parameters:
- **Model**: YOLO-NAS-S (smallest, fastest)
- **Input Size**: 640x640 (optimal for Frigate)
- **Epochs**: 200 (with early stopping)
- **Batch Size**: 8 (adjust based on GPU memory)
- **QAT**: Enabled for last 50 epochs
- **Optimizer**: AdamW with cosine learning rate

## Output Files

After completion, you'll find:

### Training Outputs
- `output/yolo_nas_s_trained.pth` - Trained PyTorch model
- `output/checkpoints/` - Training checkpoints
- `output/yolo_nas_training.log` - Training logs

### Conversion Outputs
- `output/converted_yolo_nas/640x640/wildfire_yolo_nas_s_tensorrt.engine` - TensorRT engine
- `output/wildfire_yolo_nas_labels.txt` - Class labels
- `output/wildfire_yolo_nas_frigate_config.yml` - Frigate configuration

### Deployment Outputs
- `output/deploy_to_frigate.sh` - Deployment script

## Monitoring Progress

### Training Progress
```bash
# Watch training logs
tail -f output/yolo_nas_training.log

# Check GPU usage
nvidia-smi

# View TensorBoard (if available)
tensorboard --logdir output/checkpoints/
```

### Conversion Progress
```bash
# Watch conversion logs
tail -f output/yolo_nas_deployment.log
```

## Troubleshooting

### Common Issues

#### Out of Memory Error
- Reduce batch size in training config
- Use smaller model (already using YOLO-NAS-S)
- Close other GPU applications

#### Training Too Slow
- Check GPU utilization with `nvidia-smi`
- Verify CUDA and PyTorch GPU support
- Consider reducing image resolution

#### Conversion Fails
- Ensure trained model exists: `output/yolo_nas_s_trained.pth`
- Check TensorRT installation
- Verify sufficient disk space

#### Frigate Deployment Issues
- Ensure Frigate container is running
- Check Docker permissions
- Verify model paths in configuration

### Performance Optimization

#### For Faster Training
- Use multiple GPUs if available
- Increase batch size (if memory allows)
- Use mixed precision training (enabled by default)

#### For Better Accuracy
- Increase training epochs
- Use larger model (YOLO-NAS-M or L)
- Add data augmentation
- Collect more training data

## Model Performance

Expected performance on RTX 4090:
- **Training Time**: ~48 hours for 200 epochs
- **Inference Speed**: ~5-8ms per frame
- **Model Size**: ~15MB (TensorRT INT8)
- **Accuracy**: >90% of FP32 accuracy with QAT

## Frigate Integration

After deployment, update your Frigate configuration:

```yaml
model:
  path: /models/wildfire_yolo_nas_s_tensorrt.engine
  input_tensor: nchw
  input_pixel_format: rgb
  width: 640
  height: 640
  labelmap_path: /models/wildfire_yolo_nas_labels.txt
  model_type: yolov8

detectors:
  tensorrt:
    type: tensorrt
    device: 0

objects:
  track: [fire, person, car, truck]
  filters:
    fire:
      min_area: 100
      threshold: 0.7
```

## Advanced Usage

### Resume Training
If training is interrupted, you can resume from the last checkpoint:
```bash
# Edit the training script to set resume=True
# The script will automatically find the latest checkpoint
```

### Custom Training Parameters
Edit `output/training_config.yaml` before training to customize:
- Learning rate
- Batch size
- Epochs
- Data augmentation
- QAT settings

### Multiple Model Sizes
Train different model sizes by changing the architecture:
- `yolo_nas_s` (fastest, smallest)
- `yolo_nas_m` (balanced)
- `yolo_nas_l` (most accurate, largest)

## Support

For issues:
1. Check the log files in `output/`
2. Verify prerequisites are met
3. Review troubleshooting section
4. Check Wildfire Watch documentation

## License

This training pipeline is part of the Wildfire Watch project and follows the same license terms.