#!/usr/bin/env python3.10
"""
Simple script to train a custom YOLO-NAS model on your dataset.

Usage:
    python3.10 train_custom_yolo_nas.py --dataset_path /path/to/dataset --epochs 100
"""

import argparse
import yaml
import sys
import os
from pathlib import Path
from urllib.error import URLError
from unified_yolo_trainer import UnifiedYOLOTrainer


def main():
    parser = argparse.ArgumentParser(description='Train YOLO-NAS on custom dataset')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to dataset directory with train/validation splits')
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
    parser.add_argument('--experiment_name', type=str, default='custom_yolo_nas',
                        help='Name for this experiment')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save outputs')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained COCO weights (requires internet connection)')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training (faster)')
    parser.add_argument('--no_pretrained', dest='pretrained', action='store_false',
                        help='Train from scratch without pretrained weights')
    
    args = parser.parse_args()
    
    # Validate dataset path
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"❌ Error: Dataset path does not exist: {dataset_path}")
        sys.exit(1)
    
    # Check for dataset structure - supports two formats
    # Format 1: train/images, train/labels (standard)
    format1_train_images = dataset_path / 'train' / 'images'
    format1_train_labels = dataset_path / 'train' / 'labels'
    format1_val_images = dataset_path / 'validation' / 'images'
    format1_val_labels = dataset_path / 'validation' / 'labels'
    
    # Format 2: images/train, labels/train (roboflow)
    format2_train_images = dataset_path / 'images' / 'train'
    format2_train_labels = dataset_path / 'labels' / 'train'
    format2_val_images = dataset_path / 'images' / 'validation'
    format2_val_labels = dataset_path / 'labels' / 'validation'
    
    # Check which format exists
    if format1_train_images.exists() and format1_train_labels.exists():
        train_split = 'train'
        val_split = 'validation'
        print("✓ Detected standard dataset format (train/images)")
    elif format2_train_images.exists() and format2_train_labels.exists():
        train_split = 'train'  # The split name, not the full path
        val_split = 'validation' 
        print("✓ Detected Roboflow dataset format (images/train)")
        # For Roboflow format, the dataset loader expects just the split name
    else:
        print(f"❌ Error: Dataset structure is incorrect.")
        print("\nExpected one of these structures:")
        print("\nFormat 1 (Standard):")
        print(f"   {dataset_path}/")
        print("   ├── train/")
        print("   │   ├── images/")
        print("   │   └── labels/")
        print("   └── validation/")
        print("       ├── images/")
        print("       └── labels/")
        print("\nFormat 2 (Roboflow):")
        print(f"   {dataset_path}/")
        print("   ├── images/")
        print("   │   ├── train/")
        print("   │   └── validation/")
        print("   └── labels/")
        print("       ├── train/")
        print("       └── validation/")
        sys.exit(1)
    
    # Create configuration
    config = {
        'model': {
            'architecture': f'yolo_nas_{args.model_size}',
            'num_classes': args.num_classes,
            'input_size': [640, 640],
            'pretrained_weights': 'coco' if args.pretrained else None
        },
        'dataset': {
            'data_dir': args.dataset_path,
            'train_split': train_split,
            'val_split': val_split,
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
        'experiment_name': args.experiment_name,
        'log_level': 'INFO'
    }
    
    # Save config
    config_path = Path(args.output_dir) / f'{args.experiment_name}_config.yaml'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"Configuration saved to: {config_path}")
    print("\nStarting training with configuration:")
    print(yaml.dump(config, default_flow_style=False))
    
    # Create trainer and start training
    trainer = UnifiedYOLOTrainer(str(config_path))
    
    try:
        report = trainer.train()
        print("\n✅ Training completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        
        # Print best model location
        best_model = Path(args.output_dir) / 'checkpoints' / args.experiment_name / 'ckpt_best.pth'
        if best_model.exists():
            print(f"\nBest model saved at: {best_model}")
            print("\nTo convert for deployment, run:")
            print(f"python3.10 convert_model.py --model_path {best_model} --model_type yolo_nas")
            
    except (URLError, OSError) as e:
        print(f"\n❌ Network error while downloading pretrained weights: {e}")
        print("\nPossible solutions:")
        print("1. Check your internet connection")
        print("2. Train from scratch without pretrained weights:")
        print(f"   python3.10 {sys.argv[0]} --dataset_path {args.dataset_path} --no_pretrained")
        print("\n3. If you need pretrained weights, download manually:")
        print(f"   - YOLO-NAS-S: https://sghub.deci.ai/models/yolo_nas_s_coco.pth")
        print(f"   - YOLO-NAS-M: https://sghub.deci.ai/models/yolo_nas_m_coco.pth")
        print(f"   - YOLO-NAS-L: https://sghub.deci.ai/models/yolo_nas_l_coco.pth")
        print(f"   Save to: ~/.cache/torch/hub/checkpoints/")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()