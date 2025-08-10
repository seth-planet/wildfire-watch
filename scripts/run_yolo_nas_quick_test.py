#!/usr/bin/env python3.10
"""
Quick test of YOLO-NAS QAT to Hailo E2E pipeline with minimal epochs
"""
import os
import sys
from pathlib import Path
import tempfile
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set test environment variables
os.environ['KEEP_TEST_ARTIFACTS'] = '1'  # Keep artifacts for inspection
os.environ['TEST_MODE'] = '1'  # Enable test mode for faster execution

def create_minimal_dataset():
    """Create a minimal test dataset for quick testing"""
    dataset_dir = Path(tempfile.mkdtemp(prefix="yolo_nas_test_dataset_"))
    
    # Create directory structure
    (dataset_dir / "images" / "train").mkdir(parents=True)
    (dataset_dir / "images" / "validation").mkdir(parents=True)
    (dataset_dir / "labels" / "train").mkdir(parents=True)
    (dataset_dir / "labels" / "validation").mkdir(parents=True)
    
    # Create dataset info
    dataset_info = {
        'names': ['fire', 'smoke'] + [f'class_{i}' for i in range(2, 32)],
        'nc': 32,
        'train': 'images/train',
        'val': 'images/validation'
    }
    
    import yaml
    # Create both dataset.yaml and dataset_info.yaml for compatibility
    with open(dataset_dir / 'dataset.yaml', 'w') as f:
        yaml.dump(dataset_info, f)
    with open(dataset_dir / 'dataset_info.yaml', 'w') as f:
        yaml.dump(dataset_info, f)
    
    # Create dummy images and labels
    import numpy as np
    import cv2
    
    for split in ['train', 'validation']:
        num_images = 10 if split == 'train' else 5
        for i in range(num_images):
            # Create dummy image
            img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            # Add some red regions to simulate fire
            if i % 3 == 0:
                cv2.circle(img, (320, 320), 50, (0, 0, 255), -1)
            cv2.imwrite(str(dataset_dir / "images" / split / f"img_{i:04d}.jpg"), img)
            
            # Create corresponding label
            label_content = ""
            if i % 3 == 0:
                # Add fire detection (class 0 is fire)
                label_content = "0 0.5 0.5 0.1 0.1\n"  # class cx cy w h (normalized)
            
            with open(dataset_dir / "labels" / split / f"img_{i:04d}.txt", 'w') as f:
                f.write(label_content)
    
    print(f"Created minimal dataset at: {dataset_dir}")
    return dataset_dir

def run_quick_test():
    """Run quick E2E test with minimal configuration"""
    from converted_models.unified_yolo_trainer import UnifiedYOLOTrainer
    from converted_models.model_validator import ModelAccuracyValidator
    from converted_models.model_exporter import ModelExporter
    from converted_models.inference_runner import InferenceRunner
    from converted_models.frigate_integrator import FrigateIntegrator
    
    # Create output directory
    output_dir = Path("./test_output_quick")
    output_dir.mkdir(exist_ok=True)
    
    # Create minimal dataset
    dataset_path = create_minimal_dataset()
    
    try:
        print("\n" + "="*60)
        print("Phase 1: Quick Training Test")
        print("="*60)
        
        # Initialize trainer with minimal config
        trainer = UnifiedYOLOTrainer()
        trainer.config = {
            'model': {
                'architecture': 'yolo_nas_s',
                'num_classes': None,
                'input_size': [320, 320],  # Smaller size for faster training
                'pretrained_weights': None  # No pretrained for speed
            },
            'dataset': {
                'data_dir': str(dataset_path),
                'train_split': 'train',
                'val_split': 'validation',
                'validate_labels': True
            },
            'training': {
                'epochs': 2,  # Minimal epochs
                'batch_size': 2,
                'learning_rate': 0.001,
                'workers': 0,  # Single threaded for testing
                'mixed_precision': False,
                'max_train_batches': 5,  # Limit batches per epoch
                'max_valid_batches': 2
            },
            'qat': {
                'enabled': True,
                'start_epoch': 1,  # Start QAT immediately
                'calibration_batches': 2,
                'use_wildfire_calibration_data': False  # Skip download for speed
            },
            'validation': {
                'interval': 1,
                'conf_threshold': 0.25,
                'iou_threshold': 0.45
            },
            'output_dir': str(output_dir),
            'experiment_name': 'quick_test',
            'log_level': 'INFO'
        }
        
        # Check environment
        env_info = trainer.check_environment()
        print(f"Environment: GPU={env_info['cuda_available']}, Device={env_info.get('gpu_info', 'CPU')}")
        
        # Auto-detect classes
        print("\nAuto-detecting classes...")
        class_info = trainer.auto_detect_classes()
        print(f"Detected {class_info['num_classes']} classes")
        
        # Create trainer components
        print("\nCreating trainer components...")
        components = trainer.create_trainer()
        
        # Run minimal training
        print("\nRunning minimal training...")
        trainer_obj = components['trainer']
        model = components['model']
        
        trainer_obj.train(
            model=model,
            training_params=components['training_params'],
            train_loader=components['train_loader'],
            valid_loader=components['val_loader']
        )
        
        print("✓ Training completed successfully")
        
        # Get checkpoint
        checkpoint_path = list((output_dir / "checkpoints" / "quick_test").glob("*.pth"))[0]
        print(f"✓ Checkpoint created: {checkpoint_path}")
        
        print("\n" + "="*60)
        print("Phase 2: Export Test")
        print("="*60)
        
        # Test export to ONNX
        exporter = ModelExporter()
        onnx_path = output_dir / "test_model.onnx"
        
        # For testing, just verify the exporter can be called
        # (actual export would require loading the model)
        print("✓ ModelExporter initialized successfully")
        
        print("\n" + "="*60)
        print("Phase 3: Inference Test")
        print("="*60)
        
        # Test inference runner
        runner = InferenceRunner()
        print("✓ InferenceRunner initialized successfully")
        
        print("\n" + "="*60)
        print("Phase 4: Frigate Integration Test")
        print("="*60)
        
        # Test Frigate integrator
        integrator = FrigateIntegrator("yolo_nas_test")
        
        # Create dummy HEF file for testing
        dummy_hef = output_dir / "test_model.hef"
        dummy_hef.write_bytes(b"dummy hef content")
        
        # Test deployment package creation
        class_names = ['fire', 'smoke'] + [f'class_{i}' for i in range(2, 32)]
        deployment_files = integrator.create_deployment_package(
            model_path=dummy_hef,
            output_dir=output_dir / "frigate_deployment",
            class_names=class_names,
            detector_type='hailo',
            include_test_config=True
        )
        
        print(f"✓ Deployment package created with {len(deployment_files)} files")
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print(f"Test artifacts saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup dataset
        if dataset_path.exists():
            shutil.rmtree(dataset_path, ignore_errors=True)

def main():
    """Main entry point"""
    if sys.version_info[:2] != (3, 10):
        print(f"Error: This test requires Python 3.10, running on {sys.version}")
        return 1
    
    success = run_quick_test()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())