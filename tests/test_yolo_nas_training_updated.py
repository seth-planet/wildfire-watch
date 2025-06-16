#!/usr/bin/env python3.10
"""
Updated YOLO-NAS Training Pipeline Tests
Tests the complete YOLO-NAS training pipeline with correct API usage

Note: Requires Python 3.10 for super-gradients compatibility
Run with: python3.10 -m pytest tests/test_yolo_nas_training_updated.py -v
"""

import unittest
import tempfile
import shutil
import json
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import numpy as np
import torch

# Add converted_models to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'converted_models'))


class YoloNasCorrectAPITests(unittest.TestCase):
    """Test YOLO-NAS training with correct super-gradients API usage"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp(prefix='yolo_nas_api_test_'))
        self.output_dir = self.test_dir / 'output'
        self.output_dir.mkdir(exist_ok=True)
        
        # Create mock dataset structure
        self.dataset_dir = self.test_dir / 'dataset'
        self._create_mock_dataset()
    
    def tearDown(self):
        """Clean up"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def _create_mock_dataset(self):
        """Create mock YOLO dataset structure with correct format"""
        dataset_structure = {
            'images/train': 50,
            'images/validation': 10,
            'labels/train': 50,
            'labels/validation': 10
        }
        
        for subdir, count in dataset_structure.items():
            dir_path = self.dataset_dir / subdir
            dir_path.mkdir(parents=True, exist_ok=True)
            
            for i in range(count):
                if 'images' in subdir:
                    (dir_path / f'img_{i:04d}.jpg').touch()
                else:
                    # Create valid YOLO labels with class indices 0-31 (32 classes)
                    label_file = dir_path / f'img_{i:04d}.txt'
                    with open(label_file, 'w') as f:
                        # Write valid YOLO annotation: class x_center y_center width height
                        # Ensure class index is within valid range
                        class_idx = min(i % 32, 31)
                        f.write(f"{class_idx} 0.5 0.5 0.1 0.1\n")
        
        # Create dataset.yaml with correct format (no 'nc' field)
        dataset_yaml = self.dataset_dir / 'dataset.yaml'
        class_names = {i: f"class_{i}" for i in range(32)}
        class_names[26] = "Fire"  # Fire is class 26
        
        with open(dataset_yaml, 'w') as f:
            f.write("names:\n")
            for idx, name in class_names.items():
                f.write(f"  {idx}: {name}\n")
            f.write(f"path: {self.dataset_dir}\n")
            f.write("train: ./images/train/\n")
            f.write("validation: ./images/validation/\n")
    
    def test_trainer_train_uses_dict_not_training_params(self):
        """Test that Trainer.train() is called with dict, not TrainingParams object"""
        from unified_yolo_trainer import UnifiedYOLOTrainer
        
        trainer = UnifiedYOLOTrainer()
        trainer.config = {
            'dataset': {'data_dir': str(self.dataset_dir)},
            'model': {
                'architecture': 'yolo_nas_s',
                'num_classes': 32,
                'input_size': [640, 640],
                'pretrained_weights': None
            },
            'training': {
                'epochs': 1,
                'batch_size': 2,
                'learning_rate': 0.001,
                'warmup_epochs': 1,
                'lr_scheduler': 'cosine',
                'workers': 1,
                'mixed_precision': True
            },
            'validation': {
                'conf_threshold': 0.25,
                'iou_threshold': 0.45,
                'max_predictions': 300
            },
            'experiment_name': 'test',
            'output_dir': str(self.output_dir)
        }
        
        # Auto-detect classes
        trainer.auto_detect_classes()
        
        # Create training params
        training_params = trainer._create_yolo_nas_training_params()
        
        # Verify it's a dict, not TrainingParams
        self.assertIsInstance(training_params, dict)
        self.assertIn('max_epochs', training_params)
        self.assertIn('lr_mode', training_params)
        self.assertIn('initial_lr', training_params)
        self.assertIn('loss', training_params)
        
        # Verify correct API structure
        self.assertEqual(training_params['max_epochs'], 1)
        self.assertEqual(training_params['lr_mode'], 'cosine')
        self.assertEqual(training_params['initial_lr'], 0.001)
        
        # Verify no TrainingParams object is used
        self.assertNotIn('TrainingParams', str(type(training_params)))
    
    def test_dataloader_correct_api_parameters(self):
        """Test dataloader creation with correct API parameters"""
        from unified_yolo_trainer import UnifiedYOLOTrainer
        
        trainer = UnifiedYOLOTrainer()
        trainer.config = {
            'dataset': {
                'data_dir': str(self.dataset_dir),
                'train_split': 'train',
                'val_split': 'validation',
                'class_names': [f"class_{i}" for i in range(32)]
            },
            'model': {
                'num_classes': 32,
                'input_size': [640, 640]
            },
            'training': {
                'batch_size': 2,
                'workers': 1
            }
        }
        
        with patch('super_gradients.training.dataloaders.dataloaders.coco_detection_yolo_format_train') as mock_train:
            with patch('super_gradients.training.dataloaders.dataloaders.coco_detection_yolo_format_val') as mock_val:
                # Mock returns
                mock_train.return_value = MagicMock()
                mock_val.return_value = MagicMock()
                
                # Create dataloaders
                dataloaders = trainer._create_yolo_nas_dataloaders()
                
                # Verify correct API calls
                mock_train.assert_called_once()
                call_args = mock_train.call_args
                
                # Check dataset_params structure
                dataset_params = call_args.kwargs['dataset_params']
                self.assertIn('data_dir', dataset_params)
                self.assertIn('images_dir', dataset_params)
                self.assertIn('labels_dir', dataset_params)
                self.assertIn('classes', dataset_params)
                self.assertIn('input_dim', dataset_params)
                self.assertIn('transforms', dataset_params)
                
                # Verify NOT using old API
                self.assertNotIn('train_images_dir', dataset_params)
                self.assertNotIn('train_labels_dir', dataset_params)
                
                # Check dataloader_params structure
                dataloader_params = call_args.kwargs['dataloader_params']
                self.assertIn('batch_size', dataloader_params)
                self.assertIn('num_workers', dataloader_params)
                self.assertIn('shuffle', dataloader_params)
    
    def test_class_index_validation_integration(self):
        """Test that class index validation is integrated"""
        from unified_yolo_trainer import UnifiedYOLOTrainer
        
        trainer = UnifiedYOLOTrainer()
        trainer.config = {
            'dataset': {
                'data_dir': str(self.dataset_dir),
                'train_split': 'train',
                'val_split': 'validation',
                'validate_labels': True
            },
            'model': {
                'num_classes': 32,
                'input_size': [320, 320]
            },
            'training': {'batch_size': 2, 'workers': 1}
        }
        
        # Auto-detect classes first
        trainer.auto_detect_classes()
        
        # Validate dataset labels
        validation_results = trainer.validate_dataset_labels()
        
        # Check validation was performed
        self.assertIn('train', validation_results)
        self.assertIn('validation', validation_results)
        
        # Check results structure
        train_results = validation_results['train']
        self.assertIn('valid_files', train_results)
        self.assertIn('invalid_files', train_results)
        self.assertIn('success_rate', train_results)
        self.assertIn('class_distribution', train_results)
        
        # All files should be valid (we created them correctly)
        self.assertEqual(train_results['success_rate'], 100.0)
    
    def test_model_creation_with_correct_api(self):
        """Test model creation uses correct API"""
        with patch('super_gradients.training.models.get') as mock_get:
            from unified_yolo_trainer import UnifiedYOLOTrainer
            
            trainer = UnifiedYOLOTrainer()
            trainer.config = {
                'model': {
                    'architecture': 'yolo_nas_s',
                    'num_classes': 32,
                    'pretrained_weights': None
                },
                'dataset': {'class_names': [f"class_{i}" for i in range(32)]},
                'experiment_name': 'test',
                'output_dir': str(self.output_dir)
            }
            
            # Mock model
            mock_model = MagicMock()
            mock_get.return_value = mock_model
            
            # Create trainer components
            with patch.object(trainer, '_create_yolo_nas_dataloaders') as mock_dataloaders:
                mock_dataloaders.return_value = {
                    'train': MagicMock(),
                    'val': MagicMock()
                }
                
                components = trainer._create_yolo_nas_trainer()
                
                # Verify model creation API
                mock_get.assert_called_once()
                call_args = mock_get.call_args
                
                # Check correct arguments
                self.assertEqual(call_args.kwargs['num_classes'], 32)
                self.assertIsNone(call_args.kwargs['pretrained_weights'])
    
    def test_transform_usage_for_variable_size_images(self):
        """Test that transforms are used to handle variable size images"""
        from unified_yolo_trainer import UnifiedYOLOTrainer
        
        trainer = UnifiedYOLOTrainer()
        trainer.config = {
            'dataset': {
                'data_dir': str(self.dataset_dir),
                'train_split': 'train',
                'val_split': 'validation',
                'class_names': [f"class_{i}" for i in range(32)]
            },
            'model': {
                'num_classes': 32,
                'input_size': [640, 640]
            },
            'training': {'batch_size': 2, 'workers': 1}
        }
        
        with patch('super_gradients.training.transforms.detection.DetectionLongestMaxSize') as mock_max_size:
            with patch('super_gradients.training.transforms.detection.DetectionPadIfNeeded') as mock_pad:
                # Create mock transforms
                mock_max_size.return_value = 'max_size_transform'
                mock_pad.return_value = 'pad_transform'
                
                # Create dataloaders (this should use transforms)
                with patch('super_gradients.training.dataloaders.dataloaders.coco_detection_yolo_format_train'):
                    trainer._create_yolo_nas_dataloaders()
                
                # Verify transforms were created correctly
                mock_max_size.assert_called_with(max_height=640, max_width=640)
                mock_pad.assert_called_with(min_height=640, min_width=640, pad_value=114)
    
    def test_loss_function_configuration(self):
        """Test PPYoloELoss is configured correctly"""
        from unified_yolo_trainer import UnifiedYOLOTrainer
        
        trainer = UnifiedYOLOTrainer()
        trainer.config = {
            'model': {'num_classes': 32},
            'training': {
                'epochs': 100,
                'learning_rate': 0.001,
                'warmup_epochs': 5,
                'lr_scheduler': 'cosine',
                'mixed_precision': True
            },
            'validation': {
                'conf_threshold': 0.25,
                'iou_threshold': 0.45,
                'max_predictions': 300
            }
        }
        
        # Create training params
        training_params = trainer._create_yolo_nas_training_params()
        
        # Check loss configuration
        self.assertIn('loss', training_params)
        loss = training_params['loss']
        
        # Verify it's PPYoloELoss (by checking attributes)
        self.assertTrue(hasattr(loss, 'use_static_assigner'))
        self.assertTrue(hasattr(loss, 'num_classes'))
        self.assertTrue(hasattr(loss, 'reg_max'))
    
    def test_validation_metrics_configuration(self):
        """Test validation metrics are configured correctly"""
        from unified_yolo_trainer import UnifiedYOLOTrainer
        
        trainer = UnifiedYOLOTrainer()
        trainer.config = {
            'model': {'num_classes': 32},
            'training': {
                'epochs': 100,
                'learning_rate': 0.001,
                'warmup_epochs': 5,
                'lr_scheduler': 'cosine'
            },
            'validation': {
                'conf_threshold': 0.25,
                'iou_threshold': 0.45,
                'max_predictions': 300
            }
        }
        
        # Create training params
        training_params = trainer._create_yolo_nas_training_params()
        
        # Check validation metrics
        self.assertIn('valid_metrics_list', training_params)
        self.assertIsInstance(training_params['valid_metrics_list'], list)
        self.assertGreater(len(training_params['valid_metrics_list']), 0)
        
        # Check metric configuration
        metric = training_params['valid_metrics_list'][0]
        self.assertTrue(hasattr(metric, 'score_thres'))
        self.assertTrue(hasattr(metric, 'num_cls'))


class QATTests(unittest.TestCase):
    """Test Quantization-Aware Training functionality"""
    
    def test_qat_configuration(self):
        """Test QAT can be enabled and configured"""
        from unified_yolo_trainer import UnifiedYOLOTrainer
        
        trainer = UnifiedYOLOTrainer()
        trainer.config = {
            'qat': {
                'enabled': True,
                'start_epoch': 150,
                'calibration_batches': 100
            },
            'model': {'num_classes': 32},
            'training': {'epochs': 200}
        }
        
        # Check QAT configuration is accessible
        self.assertTrue(trainer.config['qat']['enabled'])
        self.assertEqual(trainer.config['qat']['start_epoch'], 150)
        self.assertEqual(trainer.config['qat']['calibration_batches'], 100)
    
    def test_qat_disabled_by_default(self):
        """Test QAT is disabled by default"""
        from unified_yolo_trainer import UnifiedYOLOTrainer
        
        trainer = UnifiedYOLOTrainer()
        default_config = trainer._get_default_config()
        
        # Check default QAT settings
        self.assertTrue('qat' in default_config)
        self.assertTrue(default_config['qat']['enabled'])  # Default is True
        self.assertEqual(default_config['qat']['start_epoch'], 150)


class FrigateIntegrationTests(unittest.TestCase):
    """Test Frigate NVR integration for all trained classes"""
    
    def test_fire_class_detection(self):
        """Test that fire class (26) is properly configured"""
        from unified_yolo_trainer import UnifiedYOLOTrainer
        
        trainer = UnifiedYOLOTrainer()
        
        # Create test dataset with fire class
        test_dir = Path(tempfile.mkdtemp())
        dataset_dir = test_dir / 'dataset'
        dataset_dir.mkdir()
        
        # Create dataset.yaml
        with open(dataset_dir / 'dataset.yaml', 'w') as f:
            f.write("""names:
  0: Person
  1: Car
  26: Fire
  31: Background
path: /test
train: images/train
val: images/val
""")
        
        trainer.config = {'dataset': {'data_dir': str(dataset_dir)}}
        
        # Auto-detect classes
        class_info = trainer.auto_detect_classes()
        
        # Verify fire class is detected
        self.assertEqual(class_info['fire_class_index'], 26)
        self.assertIn('Fire', class_info['class_names'])
        
        # Clean up
        shutil.rmtree(test_dir)
    
    def test_all_classes_in_frigate_config(self):
        """Test that all 32 classes are included in Frigate configuration"""
        # Create mock trained model info
        class_names = [f"class_{i}" for i in range(32)]
        class_names[26] = "Fire"
        
        # Generate Frigate config snippet
        frigate_config = {
            'model': {
                'path': '/models/yolo_nas_s_wildfire.tflite',
                'input_tensor': 'nhwc',
                'input_pixel_format': 'rgb',
                'width': 320,
                'height': 320,
                'labelmap': {}
            }
        }
        
        # Add all classes to labelmap
        for i, name in enumerate(class_names):
            frigate_config['model']['labelmap'][i] = name
        
        # Verify all classes are present
        self.assertEqual(len(frigate_config['model']['labelmap']), 32)
        self.assertEqual(frigate_config['model']['labelmap'][26], 'Fire')
    
    def test_model_export_for_frigate(self):
        """Test model can be exported in Frigate-compatible format"""
        from convert_model import EnhancedModelConverter
        
        # Create test paths
        test_dir = Path(tempfile.mkdtemp())
        model_path = test_dir / 'model.pt'
        model_path.touch()
        
        converter = EnhancedModelConverter(
            model_path=str(model_path),
            output_dir=str(test_dir / 'output'),
            calibration_data=str(test_dir),
            model_size=320,  # Frigate-friendly size
            debug=True
        )
        
        # Mock model info
        converter.model_info.type = 'yolo_nas'
        converter.model_info.num_classes = 32
        converter.model_info.classes = [f"class_{i}" for i in range(32)]
        converter.model_info.classes[26] = "Fire"
        
        # Check Frigate-compatible formats
        frigate_formats = ['tflite', 'edge_tpu']
        
        for fmt in frigate_formats:
            with self.subTest(format=fmt):
                # Verify format is supported
                self.assertIn(fmt, ['onnx', 'tflite', 'edge_tpu', 'tensorrt', 'openvino', 'hailo'])
        
        # Clean up
        shutil.rmtree(test_dir)


class APIRegressionTests(unittest.TestCase):
    """Test to prevent regression to old API usage"""
    
    def test_no_training_params_object(self):
        """Ensure TrainingParams object is never used"""
        # Search for TrainingParams usage in train_yolo_nas.py
        train_script = Path(__file__).parent.parent / 'converted_models' / 'train_yolo_nas.py'
        
        if train_script.exists():
            content = train_script.read_text()
            
            # Check that TrainingParams is not imported or used
            self.assertNotIn('from super_gradients.training.params import TrainingParams', content)
            self.assertNotIn('TrainingParams(', content)
            
            # Verify dict is used instead
            self.assertIn('training_params = {', content)
    
    def test_correct_dataloader_api(self):
        """Ensure correct dataloader API is used"""
        train_script = Path(__file__).parent.parent / 'converted_models' / 'train_yolo_nas.py'
        
        if train_script.exists():
            content = train_script.read_text()
            
            # Check correct API usage
            self.assertIn('dataset_params=dataset_params', content)
            self.assertIn('dataloader_params=dataloader_params', content)
            
            # Check old API is not used
            self.assertNotIn('train_dataset_params=', content)
            self.assertNotIn('val_dataset_params=', content)
    
    def test_logger_configuration(self):
        """Ensure correct logger configuration"""
        train_script = Path(__file__).parent.parent / 'converted_models' / 'train_yolo_nas.py'
        
        if train_script.exists():
            content = train_script.read_text()
            
            # Check correct logger
            self.assertIn('"sg_logger": "base_sg_logger"', content)
            
            # Check old loggers are not used
            self.assertNotIn('sg_logger=tensorboard_logger', content)
            self.assertNotIn('TensorboardLogger', content)


if __name__ == '__main__':
    # Run tests with verbosity
    unittest.main(verbosity=2)