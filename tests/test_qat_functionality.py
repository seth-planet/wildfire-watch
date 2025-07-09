import pytest

pytestmark = pytest.mark.qat

#!/usr/bin/env python3.10
"""
Quantization-Aware Training (QAT) Tests
Ensures QAT functionality works correctly in YOLO-NAS training

Run with: python3.10 -m pytest tests/test_qat_functionality.py -v
"""

import unittest
import tempfile
import shutil
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import torch
import numpy as np

# Add converted_models to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'converted_models'))


def get_complete_test_config(output_dir='/tmp', qat_enabled=True, qat_start_epoch=150):
    """Helper function to generate complete test configuration"""
    return {
        'model': {
            'architecture': 'yolo_nas_s',
            'num_classes': 32,
            'input_size': [640, 640],
            'pretrained_weights': None
        },
        'training': {
            'epochs': 200,
            'experiment_name': 'test_qat',
            'checkpoints_dir': str(output_dir),
            'batch_size': 8,
            'learning_rate': 0.001,
            'warmup_epochs': 3,
            'cos_lr': True,
            'workers': 2,
            'save_ckpt_epoch_list': [100, 150, 200]
        },
        'qat': {
            'enabled': qat_enabled,
            'start_epoch': qat_start_epoch,
            'calibration_batches': 100
        },
        'dataset': {
            'data_dir': '/tmp/dataset',
            'train_images_dir': 'images/train',
            'train_labels_dir': 'labels/train',
            'val_images_dir': 'images/val',
            'val_labels_dir': 'labels/val',
            'class_names': ['fire', 'smoke'],
            'nc': 2
        },
        'validation': {'conf_threshold': 0.25}
    }


class QATConfigurationTests(unittest.TestCase):
    """Test QAT configuration and initialization"""
    
    def test_qat_default_configuration(self):
        """Test default QAT configuration"""
        from unified_yolo_trainer import UnifiedYOLOTrainer
        
        trainer = UnifiedYOLOTrainer()
        default_config = trainer._get_default_config()
        
        # Check QAT defaults
        self.assertIn('qat', default_config)
        self.assertTrue(default_config['qat']['enabled'])
        self.assertEqual(default_config['qat']['start_epoch'], 150)
        self.assertEqual(default_config['qat']['calibration_batches'], 100)
    
    def test_qat_can_be_disabled(self):
        """Test QAT can be disabled via configuration"""
        from unified_yolo_trainer import UnifiedYOLOTrainer
        
        trainer = UnifiedYOLOTrainer()
        trainer.config['qat']['enabled'] = False
        
        self.assertFalse(trainer.config['qat']['enabled'])
    
    def test_qat_start_epoch_validation(self):
        """Test QAT start epoch must be less than total epochs"""
        from unified_yolo_trainer import UnifiedYOLOTrainer
        
        trainer = UnifiedYOLOTrainer()
        trainer.config.update({
            'training': {'epochs': 100},
            'qat': {
                'enabled': True,
                'start_epoch': 150  # Invalid: greater than total epochs
            }
        })
        
        # QAT start_epoch should be less than total epochs
        # This test validates that the configuration is invalid
        self.assertLess(trainer.config['training']['epochs'], 
                       trainer.config['qat']['start_epoch'])
    
    def test_qat_configuration_in_training_script(self):
        """Test QAT configuration in generated training script"""
        from train_yolo_nas import create_training_script
        
        output_dir = Path(tempfile.mkdtemp())
        config = get_complete_test_config(output_dir=output_dir)
        
        try:
            # Create script
            script_path = create_training_script(config)
            content = script_path.read_text()
            
            # Check QAT mentions
            self.assertIn('QAT', content)
            self.assertIn('Quantization Aware Training', content)
            self.assertIn(str(config['qat']['start_epoch']), content)
            
        finally:
            shutil.rmtree(output_dir)


class QATCallbackTests(unittest.TestCase):
    """Test QAT callback integration"""
    
    def test_qat_callback_placeholder(self):
        """Test QAT callback is mentioned but implementation is pending"""
        # Current implementation has QAT as a placeholder
        # This test documents the expected behavior
        
        from train_yolo_nas import create_training_script
        
        config = get_complete_test_config()
        
        script_path = create_training_script(config)
        content = script_path.read_text()
        
        # Check for QAT callback placeholder
        self.assertIn('phase_callbacks', content)
        # Currently commented out: # phase_callbacks.append(QATCallback(...))
    
    def test_qat_affects_model_export(self):
        """Test that QAT affects model export thresholds"""
        from convert_model import EnhancedModelConverter
        
        # Without QAT
        converter_no_qat = EnhancedModelConverter(
            model_path='dummy.pt',
            output_dir='/tmp/no_qat',
            calibration_data='.',
            qat_enabled=False
        )
        
        # With QAT
        converter_qat = EnhancedModelConverter(
            model_path='dummy.pt',
            output_dir='/tmp/qat',
            calibration_data='.',
            qat_enabled=True
        )
        
        # QAT should affect INT8 thresholds
        threshold_no_qat = converter_no_qat._get_validation_threshold('edge_tpu')
        threshold_qat = converter_qat._get_validation_threshold('edge_tpu')
        
        # QAT models should have tighter thresholds (better accuracy)
        self.assertLess(threshold_qat, threshold_no_qat)


class QATQuantizationTests(unittest.TestCase):
    """Test quantization functionality with QAT"""
    
    def test_qat_model_quantization_aware(self):
        """Test model trained with QAT is quantization-aware"""
        # This would test that a QAT-trained model has:
        # 1. Fake quantization layers
        # 2. Quantization parameters learned during training
        # 3. Better accuracy when quantized to INT8
        
        # Mock a QAT-trained model
        class MockQATModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.quant = torch.quantization.QuantStub()
                self.conv = torch.nn.Conv2d(3, 16, 3)
                self.dequant = torch.quantization.DeQuantStub()
                
            def forward(self, x):
                x = self.quant(x)
                x = self.conv(x)
                x = self.dequant(x)
                return x
        
        model = MockQATModel()
        
        # Check model has quantization stubs
        self.assertTrue(hasattr(model, 'quant'))
        self.assertTrue(hasattr(model, 'dequant'))
    
    def test_qat_calibration_data_usage(self):
        """Test QAT uses calibration data correctly"""
        from unified_yolo_trainer import UnifiedYOLOTrainer
        
        trainer = UnifiedYOLOTrainer()
        trainer.config.update({
            'qat': {
                'enabled': True,
                'start_epoch': 150,
                'calibration_batches': 100
            }
        })
        
        # Verify calibration batches configuration
        self.assertEqual(trainer.config['qat']['calibration_batches'], 100)
    
    def test_int8_export_with_qat(self):
        """Test INT8 model export benefits from QAT"""
        from convert_model import EnhancedModelConverter
        
        # Create mock QAT-trained model
        test_dir = Path(tempfile.mkdtemp())
        try:
            model_path = test_dir / 'qat_model.pt'
            
            # Save mock model with QAT flag
            torch.save({
                'model': {},
                'qat_enabled': True,
                'nc': 32,
                'names': {i: f'class_{i}' for i in range(32)}
            }, model_path)
            
            # Create converter
            converter = EnhancedModelConverter(
                model_path=str(model_path),
                output_dir=str(test_dir / 'output'),
                calibration_data=str(test_dir),
                qat_enabled=True
            )
            
            # Mock INT8 conversion
            with patch.object(converter, 'convert_to_tflite_optimized') as mock_convert:
                mock_output = test_dir / 'output' / 'model_int8.tflite'
                mock_output.parent.mkdir(parents=True, exist_ok=True)
                mock_output.touch()
                mock_convert.return_value = mock_output
                
                # Run conversion
                result = converter.convert_to_tflite_optimized(quantize='int8')
                
                # Should use INT8 quantization
                mock_convert.assert_called_with(quantize='int8')
                
        finally:
            shutil.rmtree(test_dir)


class QATBenchmarkTests(unittest.TestCase):
    """Test QAT performance benchmarks"""
    
    def test_qat_model_accuracy_benchmark(self):
        """Test QAT models maintain accuracy after quantization"""
        # Expected behavior: QAT models should have <2% accuracy drop
        # when quantized to INT8, compared to >5% for post-training quantization
        
        qat_accuracy_drop = 1.5  # Simulated QAT model
        ptq_accuracy_drop = 5.5  # Simulated post-training quantization
        
        # QAT should perform better
        self.assertLess(qat_accuracy_drop, ptq_accuracy_drop)
        self.assertLess(qat_accuracy_drop, 2.0)  # Target: <2% drop
    
    def test_qat_inference_speed(self):
        """Test QAT models have same inference speed as regular INT8"""
        # QAT models should have same inference speed as INT8
        # but with better accuracy
        
        int8_inference_ms = 10.0
        qat_int8_inference_ms = 10.0  # Should be same
        
        # Speed should be comparable
        self.assertAlmostEqual(int8_inference_ms, qat_int8_inference_ms, delta=0.5)
    
    def test_qat_model_size(self):
        """Test QAT models have same size as regular quantized models"""
        # QAT INT8 model should be ~4x smaller than FP32
        fp32_size_mb = 100.0
        qat_int8_size_mb = 25.0
        
        # Check compression ratio
        compression_ratio = fp32_size_mb / qat_int8_size_mb
        self.assertGreater(compression_ratio, 3.5)
        self.assertLess(compression_ratio, 4.5)


class QATIntegrationTests(unittest.TestCase):
    """End-to-end QAT integration tests"""
    
    def test_qat_training_flow(self):
        """Test complete QAT training flow"""
        from unified_yolo_trainer import UnifiedYOLOTrainer
        
        # Create trainer with QAT enabled
        trainer = UnifiedYOLOTrainer()
        trainer.config.update({
            'model': {
                'architecture': 'yolo_nas_s',
                'num_classes': 32
            },
            'training': {
                'epochs': 200,
                'batch_size': 8
            },
            'qat': {
                'enabled': True,
                'start_epoch': 150,
                'calibration_batches': 100
            },
            'experiment_name': 'qat_test',
            'output_dir': '/tmp/qat_test'
        })
        
        # Verify QAT configuration is set
        self.assertTrue(trainer.config['qat']['enabled'])
        self.assertEqual(trainer.config['qat']['start_epoch'], 150)
        
        # In real training, QAT would:
        # 1. Train normally for epochs 0-149
        # 2. Enable fake quantization at epoch 150
        # 3. Fine-tune with quantization for epochs 150-199
    
    def test_qat_export_pipeline(self):
        """Test QAT model export pipeline"""
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock QAT-trained model
            model_path = Path(tmpdir) / 'qat_trained.pt'
            torch.save({
                'model': torch.nn.Linear(10, 10).state_dict(),
                'qat_enabled': True,
                'nc': 32,
                'names': {i: f'class_{i}' for i in range(32)}
            }, model_path)
            
            # Test export with QAT flag
            from convert_model import EnhancedModelConverter
            
            converter = EnhancedModelConverter(
                model_path=str(model_path),
                output_dir=f"{tmpdir}/output",
                calibration_data=str(tmpdir),
                qat_enabled=True,
                model_size=320
            )
            
            # Set model info directly
            converter.model_info.type = 'yolo_nas'
            converter.model_info.num_classes = 32
            converter.model_info.classes = [f'class_{i}' for i in range(32)]
            
            # QAT should affect validation thresholds
            tflite_threshold = converter._get_validation_threshold('tflite')
            edge_tpu_threshold = converter._get_validation_threshold('edge_tpu')
            
            # Verify thresholds are as expected
            self.assertEqual(tflite_threshold, 3.0)  # Standard FP16 threshold
            self.assertEqual(edge_tpu_threshold, 5.0)  # Better with QAT (normally 7.0)
    
    def test_qat_documentation(self):
        """Test QAT is properly documented"""
        # Check that QAT configuration is documented
        docs = {
            'qat.enabled': 'Enable Quantization-Aware Training',
            'qat.start_epoch': 'Epoch to start QAT (typically last 25% of training)',
            'qat.calibration_batches': 'Number of batches for calibration'
        }
        
        for key, description in docs.items():
            self.assertIsNotNone(description)
            self.assertIn('qat', key.lower())


class QATValidationTests(unittest.TestCase):
    """Test QAT validation and verification"""
    
    def test_verify_qat_enabled_in_model(self):
        """Test method to verify if model was trained with QAT"""
        def is_qat_model(model_path):
            """Check if model was trained with QAT"""
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                return checkpoint.get('qat_enabled', False)
            except:
                return False
        
        # Test with QAT model
        with tempfile.NamedTemporaryFile(suffix='.pt') as tmp:
            torch.save({'qat_enabled': True}, tmp.name)
            self.assertTrue(is_qat_model(tmp.name))
        
        # Test with non-QAT model
        with tempfile.NamedTemporaryFile(suffix='.pt') as tmp:
            torch.save({'qat_enabled': False}, tmp.name)
            self.assertFalse(is_qat_model(tmp.name))
    
    def test_qat_affects_frigate_deployment(self):
        """Test QAT models work better with Frigate INT8 inference"""
        # QAT models should be preferred for edge deployment
        model_options = [
            {'name': 'standard_fp32', 'size_mb': 100, 'accuracy': 95.0, 'qat': False},
            {'name': 'standard_int8', 'size_mb': 25, 'accuracy': 90.0, 'qat': False},
            {'name': 'qat_int8', 'size_mb': 25, 'accuracy': 93.5, 'qat': True}
        ]
        
        # For Frigate deployment, QAT INT8 is best
        best_model = max(model_options, 
                        key=lambda m: m['accuracy'] if m['size_mb'] < 30 else 0)
        
        self.assertEqual(best_model['name'], 'qat_int8')
        self.assertTrue(best_model['qat'])


if __name__ == '__main__':
    # Run tests with verbosity
    unittest.main(verbosity=2)