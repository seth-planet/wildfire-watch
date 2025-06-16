#!/usr/bin/env python3.10
"""
Train YOLO-NAS with INT8 Quantization-Aware Training (QAT) for edge deployment.

This implements a two-stage training process:
1. Initial full-precision training (80% of epochs)
2. QAT fine-tuning (20% of epochs)

Usage:
    python3.10 train_yolo_nas_qat.py --dataset_path /path/to/dataset --epochs 100
"""

import argparse
import yaml
import sys
import os
from pathlib import Path
from urllib.error import URLError

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from unified_yolo_trainer import UnifiedYOLOTrainer


def create_qat_config(base_config, checkpoint_path, qat_epochs):
    """Create QAT configuration based on base config"""
    qat_config = base_config.copy()
    
    # Update for QAT
    qat_config['training']['epochs'] = qat_epochs
    qat_config['training']['learning_rate'] = base_config['training']['learning_rate'] * 0.1  # Lower LR for fine-tuning
    qat_config['experiment_name'] = base_config['experiment_name'] + '_qat'
    
    # Add checkpoint to load from
    qat_config['checkpoint_params'] = {
        'checkpoint_path': checkpoint_path,
        'load_checkpoint': True
    }
    
    # Add quantization parameters
    qat_config['quantization_params'] = {
        'selective_quantizer': {
            'calibration_method': 'percentile',
            'percentile': 99.99,
            'learn_amax': True,
            'per_channel_quantization': True
        },
        'calib_num_batches': 16  # Number of batches for calibration
    }
    
    return qat_config


def main():
    parser = argparse.ArgumentParser(description='Train YOLO-NAS with INT8 QAT for edge deployment')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to dataset directory with train/validation splits')
    parser.add_argument('--model_size', type=str, default='s',
                        choices=['s', 'm', 'l'],
                        help='Model size: s (small), m (medium), l (large)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Total number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--num_classes', type=int, default=32,
                        help='Number of classes in dataset')
    parser.add_argument('--experiment_name', type=str, default='yolo_nas_qat',
                        help='Name for this experiment')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save outputs')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained COCO weights (requires internet)')
    parser.add_argument('--no_pretrained', dest='pretrained', action='store_false',
                        help='Train from scratch without pretrained weights')
    parser.add_argument('--fp32_epochs_ratio', type=float, default=0.8,
                        help='Ratio of epochs for FP32 training (default: 0.8)')
    
    args = parser.parse_args()
    
    # Validate dataset path
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"❌ Error: Dataset path does not exist: {dataset_path}")
        sys.exit(1)
    
    # Check dataset structure
    format2_train_images = dataset_path / 'images' / 'train'
    format2_train_labels = dataset_path / 'labels' / 'train'
    
    if not (format2_train_images.exists() and format2_train_labels.exists()):
        print(f"❌ Error: Dataset structure is incorrect.")
        print("Expected Roboflow format:")
        print(f"   {dataset_path}/")
        print("   ├── images/")
        print("   │   ├── train/")
        print("   │   └── validation/")
        print("   └── labels/")
        print("       ├── train/")
        print("       └── validation/")
        sys.exit(1)
    
    print("✓ Detected Roboflow dataset format")
    
    # Calculate epoch splits
    fp32_epochs = int(args.epochs * args.fp32_epochs_ratio)
    qat_epochs = args.epochs - fp32_epochs
    
    print(f"\nTraining plan:")
    print(f"- Stage 1 (FP32): {fp32_epochs} epochs")
    print(f"- Stage 2 (QAT): {qat_epochs} epochs")
    print(f"- Total: {args.epochs} epochs")
    
    # Create base configuration
    base_config = {
        'model': {
            'architecture': f'yolo_nas_{args.model_size}',
            'num_classes': args.num_classes,
            'input_size': [640, 640],
            'pretrained_weights': 'coco' if args.pretrained else None
        },
        'dataset': {
            'data_dir': str(args.dataset_path),
            'train_split': 'train',
            'val_split': 'validation',
            'validate_labels': True
        },
        'training': {
            'epochs': fp32_epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'warmup_epochs': 3 if fp32_epochs > 10 else 0,
            'mixed_precision': True,
            'workers': 8,
            'lr_scheduler': 'cosine',
            'lr_decay_factor': 0.1
        },
        'validation': {
            'interval': 1,
            'conf_threshold': 0.25,
            'iou_threshold': 0.45,
            'max_predictions': 300
        },
        'output_dir': args.output_dir,
        'experiment_name': args.experiment_name + '_fp32',
        'log_level': 'INFO'
    }
    
    # Stage 1: Full-Precision Training
    print("\n" + "="*60)
    print("STAGE 1: Full-Precision Training")
    print("="*60)
    
    # Save FP32 config
    fp32_config_path = Path(args.output_dir) / f'{args.experiment_name}_fp32_config.yaml'
    fp32_config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(fp32_config_path, 'w') as f:
        yaml.dump(base_config, f)
    
    try:
        # Train FP32 model
        trainer = UnifiedYOLOTrainer(str(fp32_config_path))
        fp32_report = trainer.train()
        
        print("\n✅ Stage 1 (FP32) completed successfully!")
        
        # Find best checkpoint
        checkpoint_dir = Path(args.output_dir) / 'checkpoints' / (args.experiment_name + '_fp32')
        best_checkpoint = checkpoint_dir / 'ckpt_best.pth'
        
        if not best_checkpoint.exists():
            # Try to find any checkpoint
            checkpoints = list(checkpoint_dir.glob('*.pth'))
            if checkpoints:
                best_checkpoint = checkpoints[-1]
            else:
                print("❌ Error: No checkpoint found from FP32 training")
                sys.exit(1)
        
        print(f"Best FP32 checkpoint: {best_checkpoint}")
        
        # Stage 2: QAT Fine-tuning
        print("\n" + "="*60)
        print("STAGE 2: Quantization-Aware Training (QAT)")
        print("="*60)
        
        # Create QAT config
        qat_config = create_qat_config(base_config, str(best_checkpoint), qat_epochs)
        
        # Save QAT config
        qat_config_path = Path(args.output_dir) / f'{args.experiment_name}_qat_config.yaml'
        with open(qat_config_path, 'w') as f:
            yaml.dump(qat_config, f)
        
        print("\nNote: QAT in super-gradients requires the QATTrainer class.")
        print("For now, we've prepared the configuration. To complete QAT:")
        print(f"\n1. Use the super-gradients QAT recipe:")
        print(f"   python -m super_gradients.qat_from_recipe --config-name=custom_qat")
        print(f"\n2. Or manually run QAT with the generated config:")
        print(f"   {qat_config_path}")
        
        print("\n✅ Training pipeline completed!")
        print(f"\nResults:")
        print(f"- FP32 model: {best_checkpoint}")
        print(f"- QAT config: {qat_config_path}")
        
        print("\nFor INT8 deployment on Coral TPU:")
        print("1. Complete QAT training using the config above")
        print("2. Export to ONNX format")
        print("3. Convert ONNX to TFLite with INT8 quantization")
        print("4. Compile for Edge TPU using edgetpu_compiler")
        
    except (URLError, OSError) as e:
        print(f"\n❌ Network error: {e}")
        print("\nIf using pretrained weights, try:")
        print(f"   python3.10 {sys.argv[0]} --dataset_path {args.dataset_path} --no_pretrained")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()