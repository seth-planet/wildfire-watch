#!/usr/bin/env python3.12
#!/usr/bin/env python3
"""
Tests for Model Converter
Tests conversion functionality without requiring GPL dependencies
"""
import os
import sys
import json
import tempfile
import shutil
import subprocess
from pathlib import Path
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from convert_model import ModelConverter, download_model

class TestModelConverter(unittest.TestCase):
    """Test ModelConverter functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_model_path = Path(self.temp_dir) / "test_model.pt"
        
        # Create a minimal fake model file
        with open(self.test_model_path, 'wb') as f:
            f.write(b"FAKE_MODEL_DATA")
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test ModelConverter initialization"""
        converter = ModelConverter(
            model_path=self.test_model_path,
            output_dir=self.temp_dir,
            model_name="test_model"
        )
        
        self.assertEqual(converter.model_name, "test_model")
        self.assertEqual(converter.model_size, (640, 640))
        self.assertTrue(converter.calibration_data.exists())
    
    def test_custom_size_initialization(self):
        """Test initialization with custom size"""
        converter = ModelConverter(
            model_path=self.test_model_path,
            output_dir=self.temp_dir,
            model_size=(416, 416)
        )
        
        self.assertEqual(converter.model_size, (416, 416))
    
    @patch('subprocess.run')
    def test_extract_model_info_external(self, mock_run):
        """Test model info extraction"""
        # Mock successful extraction
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            'type': 'yolov8',
            'classes': ['person', 'car', 'fire'],
            'num_classes': 3
        })
        mock_run.return_value = mock_result
        
        converter = ModelConverter(
            model_path=self.test_model_path,
            output_dir=self.temp_dir
        )
        converter._extract_model_info_external()
        
        self.assertEqual(converter.model_info['type'], 'yolov8')
        self.assertEqual(converter.model_info['num_classes'], 3)
        self.assertEqual(len(converter.model_info['classes']), 3)
    
    @patch('subprocess.run')
    def test_convert_to_onnx(self, mock_run):
        """Test ONNX conversion"""
        # Mock successful conversion
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "SUCCESS"
        mock_run.return_value = mock_result
        
        converter = ModelConverter(
            model_path=self.test_model_path,
            output_dir=self.temp_dir
        )
        
        # Create fake ONNX file
        onnx_path = converter.output_dir / f"{converter.model_name}.onnx"
        with open(onnx_path, 'wb') as f:
            f.write(b"FAKE_ONNX_DATA")
        
        result = converter.convert_to_onnx()
        
        self.assertEqual(result, onnx_path)
        self.assertTrue(onnx_path.exists())
    
    def test_create_synthetic_calibration(self):
        """Test synthetic calibration data creation"""
        converter = ModelConverter(
            model_path=self.test_model_path,
            output_dir=self.temp_dir
        )
        
        # Remove calibration data to force synthetic creation
        shutil.rmtree(converter.calibration_data)
        converter.calibration_data.mkdir()
        
        # Create synthetic data
        converter._create_synthetic_calibration()
        
        # Check files were created
        jpg_files = list(converter.calibration_data.glob("*.jpg"))
        self.assertEqual(len(jpg_files), 100)
    
    def test_generate_frigate_config(self):
        """Test Frigate configuration generation"""
        converter = ModelConverter(
            model_path=self.test_model_path,
            output_dir=self.temp_dir
        )
        
        # Set up model info
        converter.model_info = {
            'classes': ['person', 'car', 'fire', 'smoke'],
            'num_classes': 4
        }
        
        config_path = converter.generate_frigate_config()
        
        self.assertTrue(config_path.exists())
        
        # Load and verify config
        with open(config_path, 'r') as f:
            import yaml
            config = yaml.safe_load(f)
        
        self.assertIn('model', config)
        self.assertIn('objects', config)
        self.assertIn('fire', config['objects']['track'])
        self.assertIn('smoke', config['objects']['track'])
        
        # Check labels file
        labels_path = converter.output_dir / f"{converter.model_name}_labels.txt"
        self.assertTrue(labels_path.exists())
        
        with open(labels_path, 'r') as f:
            labels = f.read().strip().split('\n')
        
        self.assertEqual(len(labels), 4)
    
    @patch('subprocess.run')
    def test_compile_edge_tpu(self, mock_run):
        """Test Edge TPU compilation"""
        # Mock successful compilation
        mock_result = Mock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        converter = ModelConverter(
            model_path=self.test_model_path,
            output_dir=self.temp_dir
        )
        
        # Create fake quantized model
        quant_path = converter.output_dir / "test_quant.tflite"
        with open(quant_path, 'wb') as f:
            f.write(b"FAKE_QUANTIZED_MODEL")
        
        # Create fake compiled model
        compiled_path = converter.output_dir / "test_quant_edgetpu.tflite"
        with open(compiled_path, 'wb') as f:
            f.write(b"FAKE_COMPILED_MODEL")
        
        result = converter._compile_edge_tpu(quant_path)
        
        self.assertIsNotNone(result)
    
    def test_get_output_layer_names(self):
        """Test output layer name generation"""
        converter = ModelConverter(
            model_path=self.test_model_path,
            output_dir=self.temp_dir
        )
        
        names = converter._get_output_layer_names()
        
        self.assertIsInstance(names, list)
        self.assertGreater(len(names), 0)
    
    @patch('subprocess.run')
    def test_check_hailo_sdk(self, mock_run):
        """Test Hailo SDK detection"""
        converter = ModelConverter(
            model_path=self.test_model_path,
            output_dir=self.temp_dir
        )
        
        # Test SDK found
        mock_run.return_value.returncode = 0
        self.assertTrue(converter._check_hailo_sdk())
        
        # Test SDK not found
        mock_run.return_value.returncode = 1
        self.assertFalse(converter._check_hailo_sdk())
        
        # Test command not found
        mock_run.side_effect = FileNotFoundError()
        self.assertFalse(converter._check_hailo_sdk())
    
    def test_create_hailo_script(self):
        """Test Hailo conversion script creation"""
        converter = ModelConverter(
            model_path=self.test_model_path,
            output_dir=self.temp_dir
        )
        
        onnx_path = converter.output_dir / "test.onnx"
        config_path = converter.output_dir / "test_config.json"
        
        script_path = converter._create_hailo_script(onnx_path, config_path)
        
        self.assertTrue(script_path.exists())
        self.assertTrue(os.access(script_path, os.X_OK))
        
        # Check script content
        with open(script_path, 'r') as f:
            content = f.read()
        
        self.assertIn("hailo parser", content)
        self.assertIn("hailo optimize", content)
        self.assertIn("hailo compiler", content)
    
    def test_create_readme(self):
        """Test README generation"""
        converter = ModelConverter(
            model_path=self.test_model_path,
            output_dir=self.temp_dir
        )
        
        results = {
            'model_info': {
                'type': 'yolov8',
                'num_classes': 3,
                'classes': ['person', 'car', 'fire']
            },
            'outputs': {
                'onnx': {'path': Path('test.onnx')},
                'tflite': {
                    'cpu': Path('test_cpu.tflite'),
                    'edge_tpu': Path('test_edgetpu.tflite')
                }
            }
        }
        
        converter._create_readme(results)
        
        readme_path = converter.output_dir / "README.md"
        self.assertTrue(readme_path.exists())
        
        with open(readme_path, 'r') as f:
            content = f.read()
        
        self.assertIn("yolov8", content)
        self.assertIn("Deployment with Frigate", content)
        self.assertIn("test_cpu.tflite", content)
    
    @patch('urllib.request.urlretrieve')
    def test_download_model(self, mock_urlretrieve):
        """Test model download functionality"""
        output_path = Path(self.temp_dir)
        
        # Test successful download
        model_path = download_model('yolov8n', output_path)
        
        self.assertEqual(model_path.name, 'yolov8n.pt')
        mock_urlretrieve.assert_called_once()
        
        # Test unknown model
        with self.assertRaises(ValueError):
            download_model('unknown_model', output_path)
    
    def test_convert_all_error_handling(self):
        """Test error handling in convert_all"""
        converter = ModelConverter(
            model_path=self.test_model_path,
            output_dir=self.temp_dir
        )
        
        # Mock all conversion methods to raise exceptions
        converter.convert_to_onnx = Mock(side_effect=Exception("ONNX failed"))
        converter.convert_to_tflite = Mock(side_effect=Exception("TFLite failed"))
        converter.convert_to_hailo = Mock(side_effect=Exception("Hailo failed"))
        
        # Should not crash
        results = converter.convert_all()
        
        self.assertIn('model_info', results)
        self.assertIn('outputs', results)
        
        # Check summary was created
        summary_path = converter.output_dir / "conversion_summary.json"
        self.assertTrue(summary_path.exists())

class TestModelConverterIntegration(unittest.TestCase):
    """Integration tests for model converter"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    @patch('subprocess.run')
    def test_full_conversion_flow(self, mock_run):
        """Test complete conversion flow"""
        # Create test model
        model_path = Path(self.temp_dir) / "test_model.pt"
        with open(model_path, 'wb') as f:
            f.write(b"FAKE_MODEL")
        
        # Mock all subprocess calls to succeed
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "SUCCESS"
        
        # Create converter
        converter = ModelConverter(
            model_path=model_path,
            output_dir=self.temp_dir,
            model_name="integration_test"
        )
        
        # Mock model info
        converter.model_info = {
            'type': 'yolov8',
            'classes': ['fire', 'smoke', 'person'],
            'num_classes': 3
        }
        
        # Run conversion
        results = converter.convert_all()
        
        # Verify results structure
        self.assertIn('model_info', results)
        self.assertIn('outputs', results)
        
        # Check that some files were created
        files = list(Path(self.temp_dir).glob('*'))
        self.assertGreater(len(files), 0)
        
        # Check README exists
        readme_path = Path(self.temp_dir) / "README.md"
        self.assertTrue(readme_path.exists())
        
        # Check summary exists
        summary_path = Path(self.temp_dir) / "conversion_summary.json"
        self.assertTrue(summary_path.exists())

if __name__ == '__main__':
    unittest.main()
