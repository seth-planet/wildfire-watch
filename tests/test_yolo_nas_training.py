#!/usr/bin/env python3.10
"""
YOLO-NAS Training Pipeline Tests
Tests the complete YOLO-NAS training pipeline with proper API usage

Note: Requires Python 3.10 for super-gradients compatibility
Run with: python3.10 -m pytest tests/test_yolo_nas_training.py -v
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

# Add converted_models to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'converted_models'))


class YoloNasTrainingAPITests(unittest.TestCase):
    """Test YOLO-NAS training API usage and compatibility"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp(prefix='yolo_nas_test_'))
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
        """Create mock YOLO dataset structure"""
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
                        f.write(f"{min(i % 32, 31)} 0.5 0.5 0.1 0.1\n")
        
        # Create dataset.yaml
        dataset_yaml = self.dataset_dir / 'dataset.yaml'
        class_names = {i: f"class_{i}" for i in range(32)}
        class_names[26] = "Fire"  # Fire is class 26
        
        with open(dataset_yaml, 'w') as f:
            f.write(f"""
train: images/train
val: images/validation
nc: 32
names: {class_names}
""")
    
    def test_trainer_api_parameters(self):
        """Test that Trainer.train() is called with correct parameters"""
        from train_yolo_nas import create_training_script, prepare_dataset_config
        
        # Create test config
        config = {
            'dataset': {
                'data_dir': str(self.dataset_dir),
                'train_images_dir': 'images/train',
                'train_labels_dir': 'labels/train',
                'val_images_dir': 'images/validation',
                'val_labels_dir': 'labels/validation',
                'class_names': [f"class_{i}" for i in range(32)],
                'nc': 32
            },
            'model': {
                'architecture': 'yolo_nas_s',
                'num_classes': 32,
                'input_size': [640, 640],
                'pretrained_weights': None
            },
            'training': {
                'epochs': 1,  # Minimal for testing
                'batch_size': 2,
                'learning_rate': 0.001,
                'warmup_epochs': 1,
                'cos_lr': True,
                'workers': 1,
                'save_ckpt_epoch_list': [1],
                'checkpoints_dir': str(self.output_dir / 'checkpoints'),
                'experiment_name': 'test_yolo_nas'
            },
            'qat': {'enabled': False, 'start_epoch': 20},
            'validation': {
                'conf_threshold': 0.25,
                'iou_threshold': 0.45
            }
        }
        
        # Create training script
        script_path = create_training_script(config)
        self.assertTrue(script_path.exists())
        
        # Read generated script and verify API usage
        with open(script_path, 'r') as f:
            script_content = f.read()
        
        # Verify correct API calls are generated
        self.assertIn('trainer.train(', script_content)
        self.assertIn('model=model,', script_content)
        self.assertIn('training_params=training_params,', script_content)
        self.assertIn('train_loader=train_dataloader,', script_content)
        self.assertIn('valid_loader=val_dataloader', script_content)
        
        # Verify training_params is a dict, not TrainingParams object
        self.assertIn('training_params = {', script_content)
        self.assertNotIn('TrainingParams(', script_content)
    
    def test_dataloader_api_parameters(self):
        """Test correct dataloader API parameter structure"""
        from train_yolo_nas import create_training_script
        
        config = {
            'dataset': {
                'data_dir': str(self.dataset_dir),
                'train_images_dir': 'images/train',
                'train_labels_dir': 'labels/train',
                'val_images_dir': 'images/validation',
                'val_labels_dir': 'labels/validation',
                'class_names': [f"class_{i}" for i in range(32)],
                'nc': 32
            },
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
                'cos_lr': True,
                'workers': 1,
                'save_ckpt_epoch_list': [1],
                'checkpoints_dir': str(self.output_dir),
                'experiment_name': 'test'
            },
            'qat': {'enabled': False, 'start_epoch': 20},
            'validation': {'conf_threshold': 0.25, 'iou_threshold': 0.45}
        }
        
        script_path = create_training_script(config)
        with open(script_path, 'r') as f:
            script_content = f.read()
        
        # Verify correct dataloader API usage
        self.assertIn('coco_detection_yolo_format_train(', script_content)
        self.assertIn('dataset_params=dataset_params,', script_content)
        self.assertIn('dataloader_params=dataloader_params', script_content)
        
        # Verify dataset_params structure
        self.assertIn('"data_dir":', script_content)
        self.assertIn('"images_dir":', script_content)
        self.assertIn('"labels_dir":', script_content)
        self.assertIn('"classes":', script_content)
        
        # Verify dataloader_params structure
        self.assertIn('"batch_size":', script_content)
        self.assertIn('"num_workers":', script_content)
        self.assertIn('"shuffle":', script_content)
    
    def test_transform_usage(self):
        """Test that available transforms are used correctly"""
        from train_yolo_nas import create_training_script
        
        config = {
            'dataset': {
                'data_dir': str(self.dataset_dir),
                'train_images_dir': 'images/train',
                'train_labels_dir': 'labels/train',
                'val_images_dir': 'images/validation',
                'val_labels_dir': 'labels/validation',
                'class_names': [f"class_{i}" for i in range(32)],
                'nc': 32
            },
            'model': {
                'architecture': 'yolo_nas_s',
                'num_classes': 32,
                'input_size': [640, 640],
                'pretrained_weights': None
            },
            'training': {
                'epochs': 1,
                'batch_size': 2,
                'experiment_name': 'test',
                'checkpoints_dir': str(self.output_dir),
                'learning_rate': 0.001,
                'warmup_epochs': 0,
                'cos_lr': True,
                'workers': 2,
                'save_ckpt_epoch_list': [1]
            },
            'qat': {'enabled': False, 'start_epoch': 20},
            'validation': {'conf_threshold': 0.25}
        }
        
        script_path = create_training_script(config)
        with open(script_path, 'r') as f:
            script_content = f.read()
        
        # Verify correct transforms are imported and used
        self.assertIn('DetectionLongestMaxSize', script_content)
        self.assertIn('DetectionPadIfNeeded', script_content)
        
        # Verify transforms are configured correctly
        self.assertIn('DetectionLongestMaxSize(max_height=640, max_width=640)', script_content)
        self.assertIn('DetectionPadIfNeeded(min_height=640, min_width=640, pad_value=114)', script_content)
        
        # Verify transforms are added to dataset_params
        self.assertIn('"transforms": transforms', script_content)
    
    def test_model_creation_api(self):
        """Test model creation with correct API"""
        from train_yolo_nas import create_training_script
        
        config = {
            'model': {
                'architecture': 'yolo_nas_s',
                'num_classes': 32,
                'pretrained_weights': None,
                'input_size': [640, 640]
            },
            'training': {
                'experiment_name': 'test',
                'checkpoints_dir': str(self.output_dir),
                'batch_size': 8,
                'epochs': 10,
                'learning_rate': 0.001,
                'warmup_epochs': 1,
                'cos_lr': True,
                'workers': 2,
                'save_ckpt_epoch_list': [5, 10]
            },
            'dataset': {
                'data_dir': str(self.dataset_dir),
                'train_images_dir': 'images/train',
                'train_labels_dir': 'labels/train',
                'val_images_dir': 'images/validation',
                'val_labels_dir': 'labels/validation',
                'class_names': [f"class_{i}" for i in range(32)],
                'nc': 32
            },
            'qat': {'enabled': False, 'start_epoch': 20},
            'validation': {'conf_threshold': 0.25}
        }
        
        script_path = create_training_script(config)
        with open(script_path, 'r') as f:
            script_content = f.read()
        
        # Verify model creation API
        self.assertIn('models.get(', script_content)
        self.assertIn('Models.YOLO_NAS_S,', script_content)
        self.assertIn('num_classes=32,', script_content)
        self.assertIn('pretrained_weights=None', script_content)
    
    def test_training_params_structure(self):
        """Test training parameters dict structure matches API"""
        from train_yolo_nas import create_training_script
        
        config = {
            'model': {
                'architecture': 'yolo_nas_s',
                'num_classes': 32,
                'input_size': [640, 640],
                'pretrained_weights': None
            },
            'training': {
                'epochs': 5,
                'learning_rate': 0.001,
                'warmup_epochs': 2,
                'cos_lr': True,
                'batch_size': 8,
                'experiment_name': 'test',
                'checkpoints_dir': str(self.output_dir),
                'workers': 2,
                'save_ckpt_epoch_list': [1, 3, 5]
            },
            'dataset': {
                'data_dir': str(self.dataset_dir),
                'train_images_dir': 'images/train',
                'train_labels_dir': 'labels/train',
                'val_images_dir': 'images/validation',
                'val_labels_dir': 'labels/validation',
                'class_names': [f"class_{i}" for i in range(32)],
                'nc': 32
            },
            'qat': {'enabled': False, 'start_epoch': 20},
            'validation': {'conf_threshold': 0.25}
        }
        
        script_path = create_training_script(config)
        with open(script_path, 'r') as f:
            script_content = f.read()
        
        # Verify required training parameters are present
        required_params = [
            '"max_epochs": 5,',
            '"lr_mode": "cosine"',
            '"initial_lr": 0.001,',
            '"loss": PPYoloELoss(',
            '"lr_warmup_epochs": 2,',
            '"optimizer": "AdamW",',
            '"mixed_precision": True,',
            '"sg_logger": "base_sg_logger",'
        ]
        
        for param in required_params:
            self.assertIn(param, script_content, f"Missing required parameter: {param}")


class DatasetValidationTests(unittest.TestCase):
    """Test dataset validation and label filtering"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp(prefix='dataset_validation_'))
        self.dataset_dir = self.test_dir / 'dataset'
    
    def tearDown(self):
        """Clean up"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def _create_dataset_with_invalid_labels(self):
        """Create dataset with some invalid label classes"""
        # Create directory structure
        for subdir in ['images/train', 'labels/train']:
            (self.dataset_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        # Create images and labels - some with invalid class indices
        test_cases = [
            (0, True),   # Valid: class 0
            (31, True),  # Valid: class 31 (max for 32 classes)
            (32, False), # Invalid: class 32 (out of range)
            (50, False), # Invalid: class 50 (way out of range)
            (26, True),  # Valid: fire class
        ]
        
        for i, (class_id, is_valid) in enumerate(test_cases):
            # Create image file
            (self.dataset_dir / 'images/train' / f'img_{i:04d}.jpg').touch()
            
            # Create label file
            label_file = self.dataset_dir / 'labels/train' / f'img_{i:04d}.txt'
            with open(label_file, 'w') as f:
                f.write(f"{class_id} 0.5 0.5 0.1 0.1\n")
        
        return test_cases
    
    def test_label_validation_function(self):
        """Test function to validate label files"""
        test_cases = self._create_dataset_with_invalid_labels()
        
        # Create label validation function
        def validate_labels(labels_dir, num_classes):
            """Validate that all labels are within valid class range"""
            invalid_files = []
            labels_dir = Path(labels_dir)
            
            for label_file in labels_dir.glob('*.txt'):
                try:
                    with open(label_file, 'r') as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if not line:
                                continue
                            
                            parts = line.split()
                            if len(parts) < 5:
                                continue
                            
                            class_id = int(parts[0])
                            if class_id < 0 or class_id >= num_classes:
                                invalid_files.append({
                                    'file': str(label_file),
                                    'line': line_num,
                                    'class_id': class_id,
                                    'valid_range': f'0-{num_classes-1}'
                                })
                except (ValueError, IOError) as e:
                    invalid_files.append({
                        'file': str(label_file),
                        'error': str(e)
                    })
            
            return invalid_files
        
        # Test validation
        invalid_labels = validate_labels(
            self.dataset_dir / 'labels/train',
            32  # 32 classes (0-31)
        )
        
        # Should find 2 invalid files (class 32 and class 50)
        self.assertEqual(len(invalid_labels), 2)
        
        # Check specific invalid class IDs found
        invalid_class_ids = [item['class_id'] for item in invalid_labels if 'class_id' in item]
        self.assertIn(32, invalid_class_ids)
        self.assertIn(50, invalid_class_ids)
    
    def test_dataset_filtering_strategy(self):
        """Test strategy for filtering out invalid images"""
        test_cases = self._create_dataset_with_invalid_labels()
        
        def filter_dataset(images_dir, labels_dir, num_classes):
            """Filter out images with invalid labels"""
            images_dir = Path(images_dir)
            labels_dir = Path(labels_dir)
            
            valid_files = []
            invalid_files = []
            
            for image_file in images_dir.glob('*.jpg'):
                label_file = labels_dir / (image_file.stem + '.txt')
                
                if not label_file.exists():
                    invalid_files.append({
                        'image': str(image_file),
                        'reason': 'No corresponding label file'
                    })
                    continue
                
                try:
                    is_valid = True
                    with open(label_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            
                            parts = line.split()
                            if len(parts) < 5:
                                continue
                            
                            class_id = int(parts[0])
                            if class_id < 0 or class_id >= num_classes:
                                is_valid = False
                                invalid_files.append({
                                    'image': str(image_file),
                                    'label': str(label_file),
                                    'class_id': class_id,
                                    'reason': f'Class ID {class_id} out of range 0-{num_classes-1}'
                                })
                                break
                    
                    if is_valid:
                        valid_files.append({
                            'image': str(image_file),
                            'label': str(label_file)
                        })
                        
                except (ValueError, IOError) as e:
                    invalid_files.append({
                        'image': str(image_file),
                        'reason': f'Error reading label: {e}'
                    })
            
            return valid_files, invalid_files
        
        # Test filtering
        valid_files, invalid_files = filter_dataset(
            self.dataset_dir / 'images/train',
            self.dataset_dir / 'labels/train',
            32
        )
        
        # Should have 3 valid files (classes 0, 31, 26) and 2 invalid (classes 32, 50)
        self.assertEqual(len(valid_files), 3)
        self.assertEqual(len(invalid_files), 2)
        
        # Check that fire class (26) is preserved
        fire_labels = [f for f in valid_files if 'img_0004' in f['image']]
        self.assertEqual(len(fire_labels), 1)
    
    def test_dataset_stats_logging(self):
        """Test that dataset filtering is properly logged"""
        test_cases = self._create_dataset_with_invalid_labels()
        
        def log_dataset_stats(valid_files, invalid_files, total_files):
            """Log dataset filtering statistics"""
            stats = {
                'total_images': total_files,
                'valid_images': len(valid_files),
                'invalid_images': len(invalid_files),
                'valid_percentage': (len(valid_files) / total_files) * 100 if total_files > 0 else 0,
                'invalid_reasons': {}
            }
            
            # Count invalid reasons
            for invalid in invalid_files:
                reason = invalid.get('reason', 'Unknown')
                stats['invalid_reasons'][reason] = stats['invalid_reasons'].get(reason, 0) + 1
            
            return stats
        
        # Mock filtering results
        valid_files = [{'image': f'img_{i}.jpg'} for i in [0, 1, 4]]  # Classes 0, 31, 26
        invalid_files = [
            {'image': 'img_2.jpg', 'reason': 'Class ID 32 out of range 0-31'},
            {'image': 'img_3.jpg', 'reason': 'Class ID 50 out of range 0-31'}
        ]
        
        stats = log_dataset_stats(valid_files, invalid_files, 5)
        
        self.assertEqual(stats['total_images'], 5)
        self.assertEqual(stats['valid_images'], 3)
        self.assertEqual(stats['invalid_images'], 2)
        self.assertEqual(stats['valid_percentage'], 60.0)
        self.assertIn('Class ID 32 out of range', str(stats['invalid_reasons']))


class TrainingPipelineIntegrationTests(unittest.TestCase):
    """Integration tests for the complete training pipeline"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp(prefix='training_integration_'))
        self.output_dir = self.test_dir / 'output'
        self.output_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    @patch('super_gradients.Trainer')
    @patch('super_gradients.training.models')
    def test_training_pipeline_with_mocked_components(self, mock_models, mock_trainer_class):
        """Test complete training pipeline with mocked super-gradients components"""
        from train_yolo_nas import prepare_dataset_config, create_training_script
        
        # Set up mocks
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer
        mock_model = MagicMock()
        mock_models.get.return_value = mock_model
        
        # Create dataset
        dataset_dir = self.test_dir / 'dataset'
        (dataset_dir / 'images/train').mkdir(parents=True)
        (dataset_dir / 'labels/train').mkdir(parents=True)
        (dataset_dir / 'images/validation').mkdir(parents=True)
        (dataset_dir / 'labels/validation').mkdir(parents=True)
        
        # Create valid dataset.yaml
        with open(dataset_dir / 'dataset.yaml', 'w') as f:
            f.write("""
train: images/train
val: images/validation
nc: 32
names: {0: 'person', 26: 'fire', 31: 'background'}
""")
        
        # Create training script with mocked dataset
        with patch('pathlib.Path.home') as mock_home:
            mock_home.return_value = self.test_dir
            with patch('pathlib.Path.exists') as mock_exists:
                mock_exists.return_value = True
                
                # Mock prepare_dataset_config
                config = {
                    'dataset': {
                        'data_dir': str(dataset_dir),
                        'train_images_dir': 'images/train',
                        'train_labels_dir': 'labels/train',
                        'val_images_dir': 'images/validation', 
                        'val_labels_dir': 'labels/validation',
                        'class_names': ['person', 'fire', 'background'],
                        'nc': 32
                    },
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
                        'cos_lr': True,
                        'workers': 1,
                        'save_ckpt_epoch_list': [1],
                        'checkpoints_dir': str(self.output_dir),
                        'experiment_name': 'test'
                    },
                    'qat': {'enabled': False, 'start_epoch': 20},
                    'validation': {'conf_threshold': 0.25, 'iou_threshold': 0.45}
                }
                
                # Create training script
                script_path = create_training_script(config)
                self.assertTrue(script_path.exists())
                
                # Verify script contains correct API calls
                with open(script_path, 'r') as f:
                    content = f.read()
                
                # Check for correct API usage patterns
                self.assertIn('Trainer(', content)
                self.assertIn('coco_detection_yolo_format_train(', content)
                self.assertIn('training_params = {', content)
                self.assertIn('DetectionLongestMaxSize', content)
                self.assertIn('DetectionPadIfNeeded', content)


if __name__ == '__main__':
    # Run tests with verbosity
    unittest.main(verbosity=2)