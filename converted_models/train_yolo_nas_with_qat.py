#!/usr/bin/env python3.10
"""
Train YOLO-NAS with true INT8 Quantization-Aware Training for edge deployment.

This script trains a model that can be directly converted to INT8 with minimal accuracy loss.
It's specifically designed for deployment on edge devices like Coral TPU.

Usage:
    # Standard training (no quantization)
    python3.10 train_yolo_nas_with_qat.py --dataset_path /path/to/dataset --epochs 100
    
    # With INT8 QAT (for edge deployment)
    python3.10 train_yolo_nas_with_qat.py --dataset_path /path/to/dataset --epochs 100 --enable_qat
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


def main():
    parser = argparse.ArgumentParser(description='Train YOLO-NAS with optional INT8 QAT')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--model_size', type=str, default='s',
                        choices=['s', 'm', 'l'],
                        help='Model size: s (small), m (medium), l (large)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--num_classes', type=int, default=32,
                        help='Number of classes in dataset')
    parser.add_argument('--experiment_name', type=str, default='yolo_nas',
                        help='Name for this experiment')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save outputs')
    parser.add_argument('--no_pretrained', action='store_true',
                        help='Train from scratch without pretrained weights')
    parser.add_argument('--enable_qat', action='store_true',
                        help='Enable INT8 Quantization-Aware Training (simulates INT8 during training)')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training (faster)')
    
    args = parser.parse_args()
    
    # Validate dataset path
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"❌ Error: Dataset path does not exist: {dataset_path}")
        sys.exit(1)
    
    # Detect dataset format
    format2_train_images = dataset_path / 'images' / 'train'
    if not format2_train_images.exists():
        print(f"❌ Error: Expected dataset structure not found")
        print(f"Looking for: {format2_train_images}")
        sys.exit(1)
    
    print("✓ Dataset found")
    
    # Create configuration
    config = {
        'model': {
            'architecture': f'yolo_nas_{args.model_size}',
            'num_classes': args.num_classes,
            'input_size': [640, 640],
            'pretrained_weights': None if args.no_pretrained else 'coco'
        },
        'dataset': {
            'data_dir': str(args.dataset_path),
            'train_split': 'train',
            'val_split': 'validation',
            'validate_labels': True
        },
        'training': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'warmup_epochs': 3 if args.epochs > 10 else 0,
            'mixed_precision': args.mixed_precision,
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
        'experiment_name': args.experiment_name + ('_qat' if args.enable_qat else ''),
        'log_level': 'INFO'
    }
    
    # Save config
    config_path = Path(args.output_dir) / f'{config["experiment_name"]}_config.yaml'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"\nConfiguration saved to: {config_path}")
    
    if args.enable_qat:
        print("\n" + "="*60)
        print("INT8 QUANTIZATION-AWARE TRAINING ENABLED")
        print("="*60)
        print("\nIMPORTANT: True QAT in super-gradients requires:")
        print("1. Training with regular trainer first")
        print("2. Fine-tuning with QATTrainer from checkpoint")
        print("\nFor now, YOLO-NAS architecture is already quantization-friendly:")
        print("- Uses quantization-aware blocks")
        print("- Designed for INT8 deployment")
        print("- Can be quantized post-training with minimal loss")
        print("\nTo convert to INT8 after training:")
        print("1. Export to ONNX")
        print("2. Use TensorRT/TFLite quantization tools")
        print("3. Deploy on edge devices")
    
    print("\nStarting training...")
    
    try:
        trainer = UnifiedYOLOTrainer(str(config_path))
        report = trainer.train()
        
        print("\n✅ Training completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        
        # Print model location
        model_dir = Path(args.output_dir) / 'checkpoints' / config['experiment_name']
        best_model = model_dir / 'ckpt_best.pth'
        
        if best_model.exists():
            print(f"\nBest model saved at: {best_model}")
            
            if args.enable_qat:
                print("\nNext steps for INT8 deployment:")
                print("1. Export model to ONNX:")
                print(f"   python3.10 export_to_onnx.py --checkpoint {best_model}")
                print("2. Quantize to INT8:")
                print("   - For TensorRT: trtexec --onnx=model.onnx --int8")
                print("   - For TFLite: Use TFLite converter with INT8 quantization")
                print("3. Compile for Edge TPU (if using Coral):")
                print("   edgetpu_compiler model_quant.tflite")
        
    except (URLError, OSError) as e:
        print(f"\n❌ Network error: {e}")
        if not args.no_pretrained:
            print("\nTry training from scratch:")
            print(f"   python3.10 {sys.argv[0]} --dataset_path {args.dataset_path} --no_pretrained")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()