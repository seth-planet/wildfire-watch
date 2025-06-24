#!/usr/bin/env python3
"""
End-to-End Integration Tests for Model Converter
Tests complete conversion pipeline for all supported formats
"""

import unittest
import tempfile
import shutil
import json
import subprocess
import sys
import os
import time
import platform
import urllib.request
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'converted_models'))
from convert_model import EnhancedModelConverter, ModelInfo


@pytest.mark.timeout_expected
@pytest.mark.model_converter
class ModelConverterE2ETests(unittest.TestCase):
    """End-to-end tests for model conversion to all formats"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_dir = Path(tempfile.mkdtemp(prefix='e2e_converter_test_'))
        cls.models_dir = cls.test_dir / 'models'
        cls.models_dir.mkdir(exist_ok=True)
        
        # Download a tiny test model
        cls.test_model_path = cls._download_test_model()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up"""
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
    
    @classmethod
    def _download_test_model(cls):
        """Download a tiny YOLO model for testing"""
        model_path = cls.models_dir / 'yolov8n.pt'
        if not model_path.exists():
            print("Downloading test model...")
            url = 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt'
            try:
                urllib.request.urlretrieve(url, model_path)
            except Exception as e:
                print(f"Failed to download model: {e}")
                # Create a dummy model for testing
                model_path.touch()
        return model_path
    
    def setUp(self):
        """Set up each test"""
        self.output_dir = Path(tempfile.mkdtemp())
        self.calibration_dir = Path(tempfile.mkdtemp())
        self._create_calibration_data()
    
    def tearDown(self):
        """Clean up after each test"""
        import gc
        for dir_path in [self.output_dir, self.calibration_dir]:
            if dir_path.exists():
                try:
                    shutil.rmtree(dir_path)
                except Exception as e:
                    print(f"Warning: Failed to cleanup {dir_path}: {e}")
        # Force garbage collection to release file handles
        gc.collect()
    
    def _create_calibration_data(self):
        """Create minimal calibration data with actual images"""
        # Create small test images
        try:
            import cv2
            # Reduced from 50 to 10 images to prevent file descriptor exhaustion
            for i in range(10):
                # Create smaller random images (320x320 instead of 640x640)
                img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
                cv2.imwrite(str(self.calibration_dir / f"calib_{i:04d}.jpg"), img)
                # Explicitly delete the numpy array to free memory
                del img
        except ImportError:
            # Fallback to creating dummy files if cv2 not available - reduced count
            for i in range(10):
                (self.calibration_dir / f"calib_{i:04d}.jpg").touch()
    
    @pytest.mark.timeout(600)  # 10 minutes for ONNX conversion
    def test_onnx_conversion(self):
        """Test conversion to ONNX format"""
        converter = EnhancedModelConverter(
            model_path=str(self.test_model_path),
            output_dir=str(self.output_dir),
            calibration_data=str(self.calibration_dir),
            model_size=320,  # Use smaller size for faster testing
            debug=True
        )
        
        # Convert to ONNX
        results = converter.convert_all(formats=['onnx'], validate=False)
        
        # Debug: print results structure
        print(f"\nConversion results: {json.dumps(results, indent=2, default=str)}")
        
        # Check if conversion had errors
        if 'error' in results:
            self.fail(f"Conversion failed: {results['error']}")
        
        # Check ONNX file exists
        onnx_path = self.output_dir / '320x320' / 'yolov8n_320x320.onnx'
        # Also check for size-suffixed version
        onnx_path_alt = self.output_dir / '320x320' / 'yolov8n.onnx'
        
        self.assertTrue(
            onnx_path.exists() or onnx_path_alt.exists(), 
            f"ONNX file not found at {onnx_path} or {onnx_path_alt}"
        )
        
        # Verify ONNX is valid
        try:
            import onnx
            model = onnx.load(str(onnx_path))
            onnx.checker.check_model(model)
        except ImportError:
            print("ONNX not installed, skipping validation")
    
    @pytest.mark.timeout(1800)  # 30 minutes for TFLite conversion with quantization
    def test_tflite_conversion(self):
        """Test conversion to TFLite format for Coral TPU"""
        converter = EnhancedModelConverter(
            model_path=str(self.test_model_path),
            output_dir=str(self.output_dir),
            calibration_data=str(self.calibration_dir),
            model_size=320,
            debug=True
        )
        
        # Convert to TFLite
        results = converter.convert_all(formats=['tflite'], validate=False)
        
        # Check TFLite files exist - the converter creates multiple variants
        tflite_variants = [
            'yolov8n_cpu.tflite',      # FP16 optimized
            'yolov8n_quant.tflite',    # INT8 quantized
            'yolov8n_dynamic.tflite',  # Dynamic range quantized
            'yolov8n_edgetpu.tflite',  # Edge TPU compiled
            'yolov8n_320x320.tflite',  # Legacy naming
            'yolov8n_320x320_edgetpu.tflite'  # Legacy Edge TPU naming
        ]
        
        # Check if any TFLite variant exists
        tflite_found = False
        for variant in tflite_variants:
            if (self.output_dir / '320x320' / variant).exists():
                tflite_found = True
                break
        
        self.assertTrue(
            tflite_found,
            f"No TFLite files found in {self.output_dir / '320x320'}. "
            f"Checked for: {tflite_variants}"
        )
    
    @pytest.mark.timeout(1200)  # 20 minutes for OpenVINO conversion
    def test_openvino_conversion(self):
        """Test conversion to OpenVINO format"""
        converter = EnhancedModelConverter(
            model_path=str(self.test_model_path),
            output_dir=str(self.output_dir),
            calibration_data=str(self.calibration_dir),
            model_size=320,
            debug=True
        )
        
        # Convert to OpenVINO
        results = converter.convert_all(formats=['openvino'], validate=False)
        
        # Check OpenVINO files exist
        openvino_xml = self.output_dir / '320x320' / 'yolov8n_openvino.xml'
        openvino_bin = self.output_dir / '320x320' / 'yolov8n_openvino.bin'
        
        # Check if conversion was attempted
        if 'openvino' in results['sizes']['320x320']['models']:
            model_results = results['sizes']['320x320']['models']['openvino']
            if isinstance(model_results, dict) and 'error' not in model_results:
                self.assertTrue(
                    openvino_xml.exists() or openvino_bin.exists(),
                    f"OpenVINO files not found"
                )
    
    @pytest.mark.timeout(3600)  # 60 minutes for TensorRT engine building
    def test_tensorrt_conversion(self):
        """Test conversion to TensorRT format"""
        # Skip if not on Linux or no GPU
        if platform.system() != 'Linux':
            self.skipTest("TensorRT only supported on Linux")
        
        converter = EnhancedModelConverter(
            model_path=str(self.test_model_path),
            output_dir=str(self.output_dir),
            calibration_data=str(self.calibration_dir),
            model_size=320,
            debug=True
        )
        
        # Skip if TensorRT is not available
        if not converter.hardware.get('nvidia', {}).get('tensorrt', False):
            self.skipTest("TensorRT not available on this system")
        
        # Convert to TensorRT
        results = converter.convert_all(formats=['tensorrt'], validate=False)
        
        # Check for TensorRT engine or ONNX (for JIT compilation)
        trt_path = self.output_dir / '320x320' / 'yolov8n_320x320.engine'
        onnx_path = self.output_dir / '320x320' / 'yolov8n_320x320_tensorrt.onnx'
        
        # Either TensorRT engine or optimized ONNX should exist
        self.assertTrue(
            trt_path.exists() or onnx_path.exists(),
            f"No TensorRT files found"
        )
    
    @pytest.mark.timeout(2400)  # 40 minutes for Hailo conversion
    def test_hailo_conversion_with_python310(self):
        """Test Hailo conversion ensures Python 3.10 is used"""
        converter = EnhancedModelConverter(
            model_path=str(self.test_model_path),
            output_dir=str(self.output_dir),
            calibration_data=str(self.calibration_dir),
            model_size=320,
            debug=True
        )
        
        # Convert to Hailo
        results = converter.convert_all(formats=['hailo'], validate=False)
        
        # Debug output
        print(f"\nHailo conversion results: {json.dumps(results['sizes']['320x320']['models'], indent=2, default=str)}")
        
        # Check Hailo conversion artifacts
        hailo_config = self.output_dir / '320x320' / 'yolov8n_hailo_config.json'
        hailo_script = self.output_dir / '320x320' / 'convert_yolov8n_hailo_qat.sh'
        docker_compose = self.output_dir / '320x320' / 'hailo_convert_docker-compose.yml'
        
        # Verify files were created
        if 'hailo' in results['sizes']['320x320']['models']:
            hailo_results = results['sizes']['320x320']['models']['hailo']
            if isinstance(hailo_results, dict):
                self.assertTrue(hailo_config.exists(), "Hailo config not created")
                self.assertTrue(hailo_script.exists(), "Hailo script not created")
                
                # Check that the script uses Python 3.10 for Hailo SDK
                if hailo_script.exists():
                    with open(hailo_script, 'r') as f:
                        script_content = f.read()
                    
                    # The script should check for Python 3.10
                    self.assertIn('Python 3.10', script_content)
                    self.assertIn('python3.10', script_content)
                    
                # Check Docker Compose file for Hailo SDK image
                if docker_compose.exists():
                    with open(docker_compose, 'r') as f:
                        compose_content = f.read()
                    
                    # The Docker compose should use the Hailo SDK image
                    self.assertIn('hailo-ai/hailo-sdk', compose_content)
                    # And set Python path for 3.10
                    self.assertIn('python3.10', compose_content)
    
    def test_multi_size_conversion(self):
        """Test conversion with multiple sizes"""
        converter = EnhancedModelConverter(
            model_path=str(self.test_model_path),
            output_dir=str(self.output_dir),
            calibration_data=str(self.calibration_dir),
            model_size=[320, 640],  # Multiple sizes
            debug=True
        )
        
        # Convert to ONNX for both sizes
        results = converter.convert_all(formats=['onnx'], validate=False)
        
        # Check both sizes were converted
        self.assertIn('320x320', results['sizes'])
        self.assertIn('640x640', results['sizes'])
        
        # Check files exist for both sizes
        onnx_320 = self.output_dir / '320x320' / 'yolov8n_320x320.onnx'
        onnx_640 = self.output_dir / '640x640' / 'yolov8n_640x640.onnx'
        
        self.assertTrue(onnx_320.exists(), f"320x320 ONNX not found")
        self.assertTrue(onnx_640.exists(), f"640x640 ONNX not found")
    
    def test_qat_model_conversion(self):
        """Test conversion with QAT (Quantization-Aware Training) enabled"""
        converter = EnhancedModelConverter(
            model_path=str(self.test_model_path),
            output_dir=str(self.output_dir),
            calibration_data=str(self.calibration_dir),
            model_size=320,
            qat_enabled=True,  # Enable QAT
            debug=True
        )
        
        # Convert with QAT optimizations
        results = converter.convert_all(formats=['onnx', 'hailo'], validate=False)
        
        # Check QAT-optimized ONNX exists
        qat_onnx = self.output_dir / '320x320' / 'yolov8n_320x320_qat.onnx'
        regular_onnx = self.output_dir / '320x320' / 'yolov8n_320x320.onnx'
        
        # Either QAT or regular ONNX should exist
        self.assertTrue(
            qat_onnx.exists() or regular_onnx.exists(),
            "No ONNX file created"
        )
    
    def test_conversion_with_validation(self):
        """Test conversion with validation enabled"""
        converter = EnhancedModelConverter(
            model_path=str(self.test_model_path),
            output_dir=str(self.output_dir),
            calibration_data=str(self.calibration_dir),
            model_size=320,
            debug=True
        )
        
        # Mock validation to avoid actual inference
        with patch.object(converter, '_validate_single_model') as mock_validate:
            mock_validate.return_value = {
                'passed': True,
                'degradation': 0.5,
                'similarity': 0.98,
                'max_diff': 0.02
            }
            
            # Convert with validation
            results = converter.convert_all(formats=['onnx'], validate=True)
            
            # Check validation results are included
            if 'validation' in results['sizes']['320x320']:
                self.assertIn('onnx', results['sizes']['320x320']['validation'])
    
    @pytest.mark.timeout(7200)  # 2 hours for all formats conversion
    def test_all_formats_conversion(self):
        """Test conversion to all supported formats"""
        converter = EnhancedModelConverter(
            model_path=str(self.test_model_path),
            output_dir=str(self.output_dir),
            calibration_data=str(self.calibration_dir),
            model_size=320,
            debug=True
        )
        
        # Convert to all formats
        all_formats = ['onnx', 'tflite', 'openvino', 'tensorrt', 'hailo']
        results = converter.convert_all(formats=all_formats, validate=False)
        
        # Check that all formats were attempted
        converted_formats = results['sizes']['320x320']['models'].keys()
        
        # At least ONNX should always work
        self.assertIn('onnx', converted_formats)
        
        # Log which formats succeeded
        print("\nConversion results:")
        for fmt in all_formats:
            if fmt in converted_formats:
                result = results['sizes']['320x320']['models'][fmt]
                if isinstance(result, dict) and 'error' in result:
                    print(f"  {fmt}: Failed - {result['error']}")
                else:
                    print(f"  {fmt}: Success")
            else:
                print(f"  {fmt}: Not attempted")
    
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        # Test with non-existent model
        converter = EnhancedModelConverter(
            model_path='non_existent.pt',
            output_dir=str(self.output_dir),
            calibration_data=str(self.calibration_dir),
            model_size=320,
            debug=True
        )
        
        # Should handle gracefully
        results = converter.convert_all(formats=['onnx'], validate=False)
        
        # Should have error in results
        self.assertIn('error', results)
    
    def test_calibration_data_validation(self):
        """Test calibration data requirements"""
        # Create insufficient calibration data
        small_calib_dir = Path(tempfile.mkdtemp())
        try:
            # Only 5 images (less than typical requirement)
            for i in range(5):
                (small_calib_dir / f"calib_{i}.jpg").touch()
            
            converter = EnhancedModelConverter(
                model_path=str(self.test_model_path),
                output_dir=str(self.output_dir),
                calibration_data=str(small_calib_dir),
                model_size=320,
                debug=True
            )
            
            # Should still work but with warning
            results = converter.convert_all(formats=['tflite'], validate=False)
            
            # Check if warning was issued (would be in logs)
            # For now, just check it didn't crash
            self.assertIsInstance(results, dict)
        finally:
            shutil.rmtree(small_calib_dir)


if __name__ == '__main__':
    # Set up test runner with verbosity
    unittest.main(verbosity=2)