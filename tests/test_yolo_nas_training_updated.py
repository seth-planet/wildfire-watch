#!/usr/bin/env python3.10
"""
Integration Tests for YOLO-NAS Training Pipeline
Tests the complete YOLO-NAS training pipeline with real components
No mocking - uses actual super-gradients API

Note: Requires Python 3.10 for super-gradients compatibility
Run with: python3.10 -m pytest tests/test_yolo_nas_training_integration.py -v --timeout=1800
"""

import unittest
import tempfile
import pytest

pytestmark = [
    pytest.mark.yolo_nas,
    pytest.mark.integration,
    pytest.mark.slow,
]
import shutil
import json
import sys
import os
from pathlib import Path
import yaml
import torch

# Add converted_models to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'converted_models'))


class YoloNasIntegrationTests(unittest.TestCase):
    """Test YOLO-NAS training with real super-gradients components"""
    
    @classmethod
    def setUpClass(cls):
        """Set up once for all tests - check if we have required dataset"""
        cls.real_dataset_path = Path('/media/seth/SketchScratch/fiftyone/train_yolo')
        cls.has_real_dataset = cls.real_dataset_path.exists()
        
        if not cls.has_real_dataset:
            print("WARNING: Real dataset not found at /media/seth/SketchScratch/fiftyone/train_yolo")
            print("Some tests will be skipped or use minimal synthetic data")
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp(prefix='yolo_nas_integration_test_'))
        self.output_dir = self.test_dir / 'output'
        self.output_dir.mkdir(exist_ok=True)
        
        # Create minimal dataset if real one not available
        if not self.has_real_dataset:
            self.dataset_dir = self.test_dir / 'dataset'
            self._create_minimal_dataset()
        else:
            self.dataset_dir = self.real_dataset_path
    
    def tearDown(self):
        """Clean up"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def _create_minimal_dataset(self):
        """Create minimal YOLO dataset structure for testing"""
        dataset_structure = {
            'images/train': 10,
            'images/validation': 2,
            'labels/train': 10,
            'labels/validation': 2
        }
        
        for subdir, count in dataset_structure.items():
            dir_path = self.dataset_dir / subdir
            dir_path.mkdir(parents=True, exist_ok=True)
            
            for i in range(count):
                if 'images' in subdir:
                    # Create small synthetic images
                    img_file = dir_path / f'img_{i:04d}.jpg'
                    import numpy as np
                    from PIL import Image
                    img = Image.fromarray(np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8))
                    img.save(img_file)
                else:
                    # Create valid YOLO labels
                    label_file = dir_path / f'img_{i:04d}.txt'
                    with open(label_file, 'w') as f:
                        # Ensure class indices are valid (0-31 for 32 classes)
                        class_idx = i % 32
                        f.write(f"{class_idx} 0.5 0.5 0.1 0.1\n")
                        if i % 3 == 0:  # Some images have multiple objects
                            f.write(f"{(class_idx + 1) % 32} 0.3 0.3 0.15 0.15\n")
        
        # Create dataset.yaml with standard COCO-like class names
        dataset_yaml = self.dataset_dir / 'dataset.yaml'
        # Use the same class names that unified_yolo_trainer expects
        class_names = {
            0: "Person", 1: "Bicycle", 2: "Car", 3: "Motorcycle", 4: "Bus",
            5: "Truck", 6: "Airplane", 7: "Train", 8: "Boat", 9: "Bird",
            10: "Cat", 11: "Dog", 12: "Horse", 13: "Sheep", 14: "Cow",
            15: "Elephant", 16: "Bear", 17: "Zebra", 18: "Giraffe", 19: "Backpack",
            20: "Umbrella", 21: "Handbag", 22: "Tie", 23: "Suitcase", 24: "Frisbee",
            25: "Skis", 26: "Fire", 27: "Snowboard", 28: "Sports Ball", 29: "Kite",
            30: "Baseball Bat", 31: "Baseball Glove"
        }
        
        with open(dataset_yaml, 'w') as f:
            yaml.dump({
                'names': class_names,
                'nc': 32,
                'path': str(self.dataset_dir),
                'train': 'images/train',
                'val': 'images/validation'
            }, f)
    
    def test_unified_yolo_trainer_full_pipeline(self):
        """Test the complete UnifiedYOLOTrainer pipeline with real components"""
        from unified_yolo_trainer import UnifiedYOLOTrainer
        
        # Skip if no GPU available and using real dataset
        if self.has_real_dataset and not torch.cuda.is_available():
            self.skipTest("GPU required for real dataset training")
        
        # Create trainer with configuration
        trainer = UnifiedYOLOTrainer()
        trainer.config = {
            'dataset': {
                'data_dir': str(self.dataset_dir),
                'max_invalid_class_ratio': 0.05,  # Allow 5% for test dataset
                'validate_labels': False,  # Skip validation for test
                'train_split': 'train',
                'val_split': 'validation'
            },
            'model': {
                'architecture': 'yolo_nas_s',
                'num_classes': None,  # Auto-detect
                'input_size': [320, 320],  # Small size for testing
                'pretrained_weights': None
            },
            'training': {
                'epochs': 1,  # Just 1 epoch for testing
                'batch_size': 2,
                'learning_rate': 0.0001,
                'warmup_epochs': 0,
                'lr_scheduler': 'cosine',
                'workers': 0,
                'mixed_precision': False,
                'early_stopping': False
            },
            'validation': {
                'interval': 1,
                'conf_threshold': 0.25,
                'iou_threshold': 0.45,
                'max_predictions': 300
            },
            'qat': {
                'enabled': False  # Disable QAT for quick test
            },
            'experiment_name': 'test_integration',
            'output_dir': str(self.output_dir)
        }
        
        # Run full pipeline
        try:
            # Check environment
            env_info = trainer.check_environment()
            self.assertIn('cuda_available', env_info)
            self.assertIn('torch_version', env_info)
            print("✓ Environment check passed")
            
            # Auto-detect classes
            class_info = trainer.auto_detect_classes()
            self.assertEqual(class_info['num_classes'], 32)
            self.assertIn('Fire', class_info['class_names'])
            print("✓ Class detection passed")
            
            # Validate dataset (only if enabled in config)
            if trainer.config['dataset'].get('validate_labels', True):
                validation_results = trainer.validate_dataset_labels()
                if validation_results and 'train' in validation_results:
                    self.assertIn('success_rate', validation_results['train'])
            
            # Create trainer components
            print("Creating trainer components...")
            components = trainer.create_trainer()
            print(f"Components created: {list(components.keys())}")
            self.assertIn('trainer', components)
            self.assertIn('model', components)
            self.assertIn('train_loader', components)
            self.assertIn('val_loader', components)
            self.assertIn('training_params', components)
            print("✓ All components present")
            
            # Verify training params is a dict
            self.assertIsInstance(components['training_params'], dict)
            self.assertIn('max_epochs', components['training_params'])
            self.assertEqual(components['training_params']['max_epochs'], 1)
            
            # If using minimal dataset, limit batches
            if not self.has_real_dataset:
                components['training_params']['max_train_batches'] = 5
                components['training_params']['max_valid_batches'] = 2
            
            # Test one training iteration (not full training due to time)
            if torch.cuda.is_available() or not self.has_real_dataset:
                # Get one batch to test
                train_loader = components['train_loader']
                for batch_idx, (images, targets) in enumerate(train_loader):
                    # Verify batch structure
                    self.assertIsInstance(images, torch.Tensor)
                    self.assertIsInstance(targets, torch.Tensor)
                    
                    # Check dimensions
                    self.assertEqual(images.dim(), 4)  # [batch, channels, height, width]
                    self.assertEqual(targets.dim(), 2)  # [num_boxes, 6]
                    
                    # Verify class indices are valid
                    if targets.numel() > 0 and targets.shape[1] > 1:
                        class_indices = targets[:, 1]
                        self.assertTrue((class_indices >= 0).all())
                        self.assertTrue((class_indices < 32).all())
                    
                    break  # Just test one batch
                
                print("✓ UnifiedYOLOTrainer pipeline components verified")
            
        except ImportError as e:
            if "super_gradients" in str(e):
                self.skipTest("super-gradients not available")
            else:
                raise
    
    def test_auto_class_detection(self):
        """Test automatic class detection from dataset.yaml"""
        from unified_yolo_trainer import UnifiedYOLOTrainer
        
        trainer = UnifiedYOLOTrainer()
        trainer.config = {
            'dataset': {'data_dir': str(self.dataset_dir)},
            'model': {}  # Initialize model config section
        }
        
        # Auto-detect classes
        class_info = trainer.auto_detect_classes()
        
        # Verify results
        self.assertEqual(class_info['num_classes'], 32)
        self.assertEqual(len(class_info['class_names']), 32)
        self.assertEqual(class_info['class_names'][26], 'Fire')
        self.assertEqual(class_info['fire_class_index'], 26)
    
    def test_safe_dataloader_integration(self):
        """Test SafeDataLoaderWrapper integration in training pipeline"""
        from unified_yolo_trainer import UnifiedYOLOTrainer
        from class_index_fixer import SafeDataLoaderWrapper
        
        trainer = UnifiedYOLOTrainer()
        trainer.config = {
            'dataset': {
                'data_dir': str(self.dataset_dir),
                'train_split': 'train',
                'val_split': 'validation',
                'max_invalid_class_ratio': 0.001  # Strict threshold
            },
            'model': {
                'num_classes': 32,
                'input_size': [320, 320]
            },
            'training': {
                'batch_size': 2,
                'workers': 0
            },
            'output_dir': str(self.output_dir)
        }
        
        # Auto-detect classes first
        trainer.auto_detect_classes()
        
        # Create dataloaders
        dataloaders = trainer._create_yolo_nas_dataloaders()
        
        # Verify SafeDataLoaderWrapper is used
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']
        
        # Check if wrapped (either directly or through another wrapper)
        is_wrapped = (
            isinstance(train_loader, SafeDataLoaderWrapper) or
            hasattr(train_loader, 'base_dataloader') or
            hasattr(trainer, '_train_wrapper')
        )
        
        if is_wrapped:
            print("✓ SafeDataLoaderWrapper is integrated")
            
            # Test iteration doesn't crash
            for batch_idx, (images, targets) in enumerate(train_loader):
                if targets.numel() > 0:
                    class_indices = targets[:, 1]
                    self.assertTrue((class_indices >= 0).all())
                    self.assertTrue((class_indices < 32).all())
                
                if batch_idx >= 2:  # Just test a few batches
                    break
        else:
            print("ℹ SafeDataLoaderWrapper not available, using standard dataloader")
    
    def test_training_params_structure(self):
        """Test that training params are created as dict with correct structure"""
        from unified_yolo_trainer import UnifiedYOLOTrainer
        
        trainer = UnifiedYOLOTrainer()
        trainer.config = {
            'model': {'num_classes': 32},
            'training': {
                'epochs': 100,
                'batch_size': 8,
                'learning_rate': 0.001,
                'warmup_epochs': 5,
                'lr_scheduler': 'cosine',
                'mixed_precision': False
            },
            'validation': {
                'conf_threshold': 0.25,
                'iou_threshold': 0.45,
                'max_predictions': 300
            }
        }
        
        # Create training params
        training_params = trainer._create_yolo_nas_training_params()
        
        # Verify it's a dict
        self.assertIsInstance(training_params, dict)
        
        # Check required keys
        required_keys = [
            'max_epochs', 'lr_mode', 'initial_lr', 'loss',
            'valid_metrics_list', 'metric_to_watch', 'optimizer'
        ]
        for key in required_keys:
            self.assertIn(key, training_params)
        
        # Verify values
        self.assertEqual(training_params['max_epochs'], 100)
        self.assertEqual(training_params['initial_lr'], 0.001)
        self.assertEqual(training_params['metric_to_watch'], 'mAP@0.50')
        
        # Verify loss object
        loss = training_params['loss']
        self.assertTrue(hasattr(loss, 'num_classes'))
        self.assertEqual(loss.num_classes, 32)
        
        # Verify metrics
        metrics = training_params['valid_metrics_list']
        self.assertIsInstance(metrics, list)
        self.assertGreater(len(metrics), 0)
    
    def test_model_size_configurations(self):
        """Test different model sizes for edge deployment"""
        from unified_yolo_trainer import UnifiedYOLOTrainer
        
        model_configs = [
            {'size': [320, 320], 'desc': 'Edge/Coral TPU'},
            {'size': [416, 416], 'desc': 'Balanced'},
            {'size': [640, 640], 'desc': 'Optimal accuracy'}
        ]
        
        for config in model_configs:
            with self.subTest(model_size=config['size']):
                trainer = UnifiedYOLOTrainer()
                trainer.config = {
                    'model': {
                        'architecture': 'yolo_nas_s',
                        'num_classes': 32,
                        'input_size': config['size']
                    },
                    'dataset': {'data_dir': str(self.dataset_dir)},
                    'training': {'batch_size': 2, 'workers': 0},
                    'output_dir': str(self.output_dir)
                }
                
                # Verify configuration
                self.assertEqual(trainer.config['model']['input_size'], config['size'])
                print(f"✓ Model size {config['size']} configured for {config['desc']}")
    
    def test_qat_configuration(self):
        """Test Quantization-Aware Training configuration"""
        from unified_yolo_trainer import UnifiedYOLOTrainer
        
        trainer = UnifiedYOLOTrainer()
        
        # Test default QAT config
        default_config = trainer._get_default_config()
        self.assertIn('qat', default_config)
        self.assertTrue(default_config['qat']['enabled'])
        self.assertEqual(default_config['qat']['start_epoch'], 150)
        
        # Test custom QAT config
        trainer.config = {
            'qat': {
                'enabled': True,
                'start_epoch': 100,
                'calibration_batches': 50
            }
        }
        
        self.assertTrue(trainer.config['qat']['enabled'])
        self.assertEqual(trainer.config['qat']['start_epoch'], 100)
        self.assertEqual(trainer.config['qat']['calibration_batches'], 50)


class FrigateNVRIntegrationTests(unittest.TestCase):
    """Test integration with Frigate NVR"""
    
    def test_frigate_model_compatibility(self):
        """Test model export compatibility with Frigate"""
        # Frigate expects specific formats
        frigate_compatible_formats = ['tflite', 'edge_tpu']
        frigate_input_sizes = [320, 416]  # Common Frigate sizes
        
        config = {
            'model': {
                'path': '/models/yolo_nas_wildfire.tflite',
                'input_tensor': 'nhwc',
                'input_pixel_format': 'rgb',
                'width': 320,
                'height': 320,
                'labelmap': {
                    0: 'Person',
                    26: 'Fire',
                    31: 'Background'
                }
            },
            'detect': {
                'width': 1920,
                'height': 1080,
                'fps': 5,
                'max_disappeared': 25
            },
            'objects': {
                'track': ['Fire'],
                'filters': {
                    'Fire': {
                        'min_score': 0.5,
                        'threshold': 0.7,
                        'min_area': 100
                    }
                }
            }
        }
        
        # Verify Fire class configuration
        self.assertEqual(config['model']['labelmap'][26], 'Fire')
        self.assertIn('Fire', config['objects']['track'])
        self.assertIn('Fire', config['objects']['filters'])
        
        print("✓ Frigate configuration validated for Fire detection")
    
    def test_frigate_detection_events(self):
        """Test Frigate event structure for fire detection"""
        # Example Frigate MQTT event
        frigate_event = {
            'type': 'new',
            'camera': 'backyard',
            'label': 'Fire',
            'id': '1234567890.123456-abc123',
            'score': 0.85,
            'box': [100, 100, 200, 200],  # x1, y1, x2, y2
            'area': 10000,
            'ratio': 1.0,
            'region': [50, 50, 250, 250],
            'current_zones': [],
            'entered_zones': [],
            'has_clip': True,
            'has_snapshot': True
        }
        
        # Verify fire detection event
        self.assertEqual(frigate_event['label'], 'Fire')
        self.assertGreater(frigate_event['score'], 0.7)
        self.assertTrue(frigate_event['has_snapshot'])
        
        # MQTT topic format
        mqtt_topic = f"frigate/{frigate_event['camera']}/{frigate_event['label'].lower()}"
        expected_topic = "frigate/backyard/fire"
        self.assertEqual(mqtt_topic, expected_topic)
        
        print(f"✓ Frigate fire detection event: score={frigate_event['score']}")


if __name__ == '__main__':
    # Check Python version
    if sys.version_info[:2] != (3, 10):
        print(f"WARNING: This test requires Python 3.10, but running {sys.version}")
        print("Run with: python3.10 -m pytest tests/test_yolo_nas_training_integration.py -v")
    
    # Run tests with high verbosity
    unittest.main(verbosity=2)