# YOLO-NAS Wildfire Detection Training Commands

This document provides comprehensive commands for training YOLO-NAS models for wildfire detection, including Quantization-Aware Training (QAT) and deployment to various formats.

## Prerequisites

### Python Environment Setup
```bash
# YOLO-NAS requires Python 3.10 for super-gradients compatibility
python3.10 -m venv venv_yolo_nas
source venv_yolo_nas/bin/activate

# Install dependencies
python3.10 -m pip install --upgrade pip
python3.10 -m pip install super-gradients==3.5.0
python3.10 -m pip install torch torchvision
python3.10 -m pip install onnx onnxruntime
python3.10 -m pip install opencv-python pyyaml
python3.10 -m pip install tqdm matplotlib tensorboard
```

### Dataset Preparation
```bash
# Download wildfire dataset (example structure)
# Expected structure:
# dataset/
# ├── images/
# │   ├── train/
# │   └── validation/
# ├── labels/
# │   ├── train/
# │   └── validation/
# └── dataset_info.yaml

# Download wildfire calibration data for QAT
wget https://huggingface.co/datasets/mailseth/wildfire-watch/resolve/main/wildfire_calibration_data.tar.gz
tar -xzf wildfire_calibration_data.tar.gz -C ./calibration_data/
```

## 1. Basic YOLO-NAS Training (FP32)

### Command-line Training
```bash
# Train YOLO-NAS Small model
python3.10 converted_models/unified_yolo_trainer.py \
    --config configs/yolo_nas_wildfire.yaml \
    --model-architecture yolo_nas_s \
    --data-dir /path/to/wildfire/dataset \
    --output-dir ./outputs/yolo_nas_fire \
    --epochs 100 \
    --batch-size 16 \
    --learning-rate 0.001 \
    --num-workers 8

# Train YOLO-NAS Medium model (better accuracy, slower)
python3.10 converted_models/unified_yolo_trainer.py \
    --model-architecture yolo_nas_m \
    --data-dir /path/to/wildfire/dataset \
    --output-dir ./outputs/yolo_nas_m_fire \
    --epochs 150 \
    --batch-size 8

# Train YOLO-NAS Large model (best accuracy, slowest)
python3.10 converted_models/unified_yolo_trainer.py \
    --model-architecture yolo_nas_l \
    --data-dir /path/to/wildfire/dataset \
    --output-dir ./outputs/yolo_nas_l_fire \
    --epochs 200 \
    --batch-size 4
```

### Python Script Training
```python
#!/usr/bin/env python3.10
from converted_models.unified_yolo_trainer import UnifiedYOLOTrainer

# Initialize trainer
trainer = UnifiedYOLOTrainer()

# Configure training
trainer.config = {
    'model': {
        'architecture': 'yolo_nas_s',
        'num_classes': None,  # Auto-detect from dataset
        'input_size': [640, 640],
        'pretrained_weights': 'coco'  # Use COCO pretrained weights
    },
    'dataset': {
        'data_dir': '/path/to/wildfire/dataset',
        'train_split': 'train',
        'val_split': 'validation',
        'validate_labels': True
    },
    'training': {
        'epochs': 100,
        'batch_size': 16,
        'learning_rate': 0.001,
        'workers': 8,
        'mixed_precision': True
    },
    'output_dir': './outputs/yolo_nas_fire',
    'experiment_name': 'wildfire_detection'
}

# Run training
trainer.train()
```

## 2. Quantization-Aware Training (QAT)

### QAT Command-line Training
```bash
# Train with QAT enabled (recommended for edge deployment)
python3.10 converted_models/unified_yolo_trainer.py \
    --model-architecture yolo_nas_s \
    --data-dir /path/to/wildfire/dataset \
    --output-dir ./outputs/yolo_nas_qat \
    --epochs 150 \
    --qat-enabled \
    --qat-start-epoch 100 \
    --qat-calibration-batches 100 \
    --use-wildfire-calibration-data \
    --batch-size 16
```

### QAT Python Script
```python
#!/usr/bin/env python3.10
from converted_models.unified_yolo_trainer import UnifiedYOLOTrainer

trainer = UnifiedYOLOTrainer()
trainer.config = {
    'model': {
        'architecture': 'yolo_nas_s',
        'num_classes': None,
        'input_size': [640, 640],
        'pretrained_weights': 'coco'
    },
    'dataset': {
        'data_dir': '/path/to/wildfire/dataset',
        'train_split': 'train',
        'val_split': 'validation'
    },
    'training': {
        'epochs': 150,
        'batch_size': 16,
        'learning_rate': 0.001,
        'workers': 8
    },
    'qat': {
        'enabled': True,
        'start_epoch': 100,  # Start QAT after 100 epochs of FP32 training
        'calibration_batches': 100,
        'use_wildfire_calibration_data': True,
        'calibration_data_url': 'https://huggingface.co/datasets/mailseth/wildfire-watch/resolve/main/wildfire_calibration_data.tar.gz'
    },
    'output_dir': './outputs/yolo_nas_qat',
    'experiment_name': 'wildfire_qat'
}

trainer.train()
```

## 3. Model Export Commands

### Export to ONNX
```bash
# Export trained model to ONNX
python3.10 -c "
from converted_models.model_exporter import ModelExporter
from pathlib import Path

exporter = ModelExporter()
onnx_path = exporter.export_to_onnx(
    model_path=Path('./outputs/yolo_nas_qat/best_model.pth'),
    output_dir=Path('./outputs/exported'),
    input_size=(640, 640)
)
print(f'ONNX model exported to: {onnx_path}')
"
```

### Convert to Hailo HEF
```bash
# Convert ONNX to Hailo HEF format
python3.10 -c "
from converted_models.model_exporter import ModelExporter
from pathlib import Path

exporter = ModelExporter()
hef_path = exporter.convert_to_hailo_hef(
    onnx_path=Path('./outputs/exported/yolo_nas_qat.onnx'),
    output_dir=Path('./outputs/exported'),
    calibration_data=Path('./calibration_data/'),
    hailo_arch='hailo8l'
)
print(f'HEF model created: {hef_path}')
"
```

### Convert to TensorFlow Lite
```bash
# Convert to TFLite for Coral TPU (requires Python 3.8)
python3.8 -c "
from converted_models.model_exporter import ModelExporter
from pathlib import Path

exporter = ModelExporter()
tflite_path = exporter.convert_to_tflite(
    onnx_path=Path('./outputs/exported/yolo_nas_qat.onnx'),
    output_dir=Path('./outputs/exported'),
    quantize=True,
    target_platform='edgetpu'
)
print(f'TFLite model created: {tflite_path}')
"
```

## 4. Model Validation Commands

### Validate Model Accuracy
```bash
# Validate model accuracy between formats
python3.10 -c "
from converted_models.model_validator import ModelAccuracyValidator
from converted_models.inference_runner import InferenceRunner
from pathlib import Path

validator = ModelAccuracyValidator()
runner = InferenceRunner()

# Get test images
test_images = list(Path('./dataset/images/validation').glob('*.jpg'))[:20]

# Run inference on different formats
pytorch_results = runner.run_inference_pytorch(
    Path('./outputs/yolo_nas_qat/best_model.pth'),
    test_images
)

onnx_results = runner.run_inference_onnx(
    Path('./outputs/exported/yolo_nas_qat.onnx'),
    test_images
)

# Validate accuracy
passed, metrics = validator.validate_model_outputs(
    pytorch_results,
    onnx_results,
    required_agreement=0.99
)

print(f'Validation passed: {passed}')
print(f'Overall agreement: {metrics[\"overall_agreement\"]:.2%}')
print(f'Fire class agreement: {metrics[\"fire_class_metrics\"][\"mean_agreement\"]:.2%}')
"
```

## 5. Frigate Integration Commands

### Generate Frigate Deployment Package
```bash
# Create Frigate deployment package
python3.10 -c "
from converted_models.frigate_integrator import FrigateIntegrator
from pathlib import Path

integrator = FrigateIntegrator('yolo_nas_wildfire')

# Class names for wildfire dataset
class_names = ['fire', 'smoke', 'flame', 'ember'] + [f'class_{i}' for i in range(4, 32)]

deployment_files = integrator.create_deployment_package(
    model_path=Path('./outputs/exported/yolo_nas_qat.hef'),
    output_dir=Path('./frigate_deployment'),
    class_names=class_names,
    detector_type='hailo',
    include_test_config=True
)

print(f'Deployment package created with {len(deployment_files)} files')
"
```

### Deploy to Frigate
```bash
# Deploy model to Frigate
cd ./frigate_deployment
./deploy_to_frigate.sh

# Or manually copy files
cp yolo_nas_wildfire.hef /opt/frigate/models/
cp yolo_nas_wildfire_labels.txt /opt/frigate/models/

# Update Frigate config.yml with the generated configuration
```

## 6. Testing Commands

### Run End-to-End Test
```bash
# Run comprehensive E2E test with Python 3.10
python3.10 -m pytest tests/test_yolo_nas_qat_hailo_e2e.py -v --timeout=14400

# Run with specific test phases
python3.10 -m pytest tests/test_yolo_nas_qat_hailo_e2e.py::test_yolo_nas_qat_hailo_e2e -v

# Keep test artifacts for debugging
KEEP_TEST_ARTIFACTS=1 python3.10 -m pytest tests/test_yolo_nas_qat_hailo_e2e.py -v
```

### Quick Training Test
```bash
# Test training pipeline with minimal epochs
python3.10 converted_models/unified_yolo_trainer.py \
    --model-architecture yolo_nas_s \
    --data-dir /path/to/wildfire/dataset \
    --output-dir ./test_output \
    --epochs 2 \
    --batch-size 4 \
    --qat-enabled \
    --qat-start-epoch 1 \
    --test-mode
```

## 7. Performance Benchmarking

### Benchmark Inference Speed
```bash
# Benchmark different model formats
python3.10 -c "
from converted_models.inference_runner import InferenceRunner
from pathlib import Path
import time

runner = InferenceRunner()
test_images = list(Path('./dataset/images/validation').glob('*.jpg'))[:100]

# Benchmark PyTorch
start = time.time()
runner.run_inference_pytorch(
    Path('./outputs/yolo_nas_qat/best_model.pth'),
    test_images
)
pytorch_time = time.time() - start

# Benchmark ONNX
start = time.time()
runner.run_inference_onnx(
    Path('./outputs/exported/yolo_nas_qat.onnx'),
    test_images
)
onnx_time = time.time() - start

print(f'PyTorch: {pytorch_time/len(test_images)*1000:.1f} ms/image')
print(f'ONNX: {onnx_time/len(test_images)*1000:.1f} ms/image')
"
```

## 8. Advanced Training Options

### Multi-GPU Training
```bash
# Train on multiple GPUs
python3.10 converted_models/unified_yolo_trainer.py \
    --model-architecture yolo_nas_m \
    --data-dir /path/to/wildfire/dataset \
    --output-dir ./outputs/multi_gpu \
    --epochs 100 \
    --batch-size 32 \
    --multi-gpu \
    --num-gpus 2
```

### Resume Training from Checkpoint
```bash
# Resume training from checkpoint
python3.10 converted_models/unified_yolo_trainer.py \
    --model-architecture yolo_nas_s \
    --data-dir /path/to/wildfire/dataset \
    --output-dir ./outputs/resumed \
    --resume-from ./outputs/yolo_nas_fire/checkpoints/epoch_50.pth \
    --epochs 150
```

### Custom Augmentation
```bash
# Train with custom augmentation
python3.10 converted_models/unified_yolo_trainer.py \
    --model-architecture yolo_nas_s \
    --data-dir /path/to/wildfire/dataset \
    --output-dir ./outputs/augmented \
    --epochs 100 \
    --augmentation-config configs/fire_augmentation.yaml
```

## Environment Variables

```bash
# Set environment variables for training
export CUDA_VISIBLE_DEVICES=0,1  # Use specific GPUs
export TORCH_HOME=/path/to/model/cache  # Model cache directory
export YOLO_NAS_CACHE=/path/to/yolo_nas/cache  # YOLO-NAS specific cache
export WANDB_API_KEY=your_key  # For Weights & Biases logging
export CLEARML_API_KEY=your_key  # For ClearML logging
```

## Troubleshooting

### Common Issues and Solutions

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   --batch-size 8
   # Enable gradient accumulation
   --gradient-accumulation 4
   ```

2. **QAT Training Fails**
   ```bash
   # Ensure proper FP32 training before QAT
   --qat-start-epoch 100  # Start QAT later
   # Use more calibration data
   --qat-calibration-batches 200
   ```

3. **Model Export Fails**
   ```bash
   # Check model compatibility
   python3.10 -m converted_models.model_validator --check-export-compatibility
   ```

## Notes

- Always use Python 3.10 for YOLO-NAS training (super-gradients requirement)
- Use Python 3.8 for Coral TPU/TFLite operations
- QAT is recommended for edge deployment (Hailo, Coral TPU)
- Fire class ID is typically 26 in the dataset
- Wildfire calibration data improves QAT accuracy significantly
- Model sizes: 640x640 (recommended), 416x416 (balanced), 320x320 (edge)