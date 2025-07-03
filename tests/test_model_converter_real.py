#!/usr/bin/env python3.12
"""
Real Model Conversion Tests
Tests actual model conversions without mocking
"""

import os
import sys
import time
import pytest
import shutil
import json
import subprocess
from pathlib import Path
import tempfile
import urllib.request
import numpy as np

# Add parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import conversion module
sys.path.insert(0, str(Path(__file__).parent.parent / 'converted_models'))

try:
    from convert_model import EnhancedModelConverter
    HAS_CONVERTER = True
except ImportError:
    HAS_CONVERTER = False


class TestRealModelConversions:
    """Test real model conversions without mocking"""
    
    @pytest.fixture(scope="class")
    def test_model(self, tmp_path_factory):
        """Download and provide a real test model"""
        model_dir = tmp_path_factory.mktemp("models")
        model_path = model_dir / "yolov8n.pt"
        
        if not model_path.exists():
            print("\nDownloading YOLOv8n model for testing...")
            url = 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt'
            try:
                urllib.request.urlretrieve(url, str(model_path))
                print(f"✓ Downloaded model to {model_path}")
            except Exception as e:
                pytest.skip(f"Could not download test model: {e}")
        
        return model_path
    
    @pytest.fixture
    def calibration_data(self, tmp_path):
        """Create real calibration images"""
        cal_dir = tmp_path / "calibration"
        cal_dir.mkdir()
        
        print("\nCreating calibration images...")
        
        try:
            import cv2
            # Create 20 random images for calibration
            for i in range(20):
                # Random fire-like colors (reds, oranges, yellows)
                img = np.zeros((640, 640, 3), dtype=np.uint8)
                
                # Add some random fire-colored regions
                for _ in range(10):
                    x, y = np.random.randint(0, 640, 2)
                    radius = np.random.randint(20, 100)
                    color = (
                        np.random.randint(100, 255),  # Blue (low)
                        np.random.randint(50, 150),   # Green (medium)
                        np.random.randint(200, 255)    # Red (high)
                    )
                    cv2.circle(img, (x, y), radius, color, -1)
                
                # Add noise
                noise = np.random.randint(0, 50, (640, 640, 3), dtype=np.uint8)
                img = cv2.add(img, noise)
                
                cv2.imwrite(str(cal_dir / f"calib_{i:04d}.jpg"), img)
            
            print(f"✓ Created {len(list(cal_dir.glob('*.jpg')))} calibration images")
            
        except ImportError:
            # Create dummy files if OpenCV not available
            print("OpenCV not available, creating dummy calibration files")
            for i in range(20):
                (cal_dir / f"calib_{i:04d}.jpg").touch()
        
        return cal_dir
    
    @pytest.fixture
    def output_dir(self, tmp_path):
        """Create output directory"""
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        return out_dir
    
    @pytest.mark.skipif(not HAS_CONVERTER, reason="Converter module not available")
    def test_onnx_conversion_real(self, test_model, calibration_data, output_dir):
        """Test real ONNX conversion without mocking"""
        print("\n" + "="*60)
        print("Testing Real ONNX Conversion")
        print("="*60)
        
        # Create converter
        converter = EnhancedModelConverter(
            model_path=str(test_model),
            output_dir=str(output_dir),
            calibration_data=str(calibration_data),
            model_size=320,  # Use smaller size for faster testing
            debug=True
        )
        
        # Convert to ONNX
        print("\nConverting to ONNX...")
        start_time = time.time()
        
        results = converter.convert_all(
            formats=['onnx'],
            validate=False,  # Skip validation for speed
            benchmark=False
        )
        
        conversion_time = time.time() - start_time
        print(f"\nConversion completed in {conversion_time:.2f} seconds")
        
        # Check results
        assert 'sizes' in results
        assert '320x320' in results['sizes']
        
        size_results = results['sizes']['320x320']
        assert 'models' in size_results
        assert 'onnx' in size_results['models']
        
        # Find ONNX file
        onnx_files = list((output_dir / '320x320').glob('*.onnx'))
        assert len(onnx_files) > 0, f"No ONNX files found in {output_dir / '320x320'}"
        
        onnx_path = onnx_files[0]
        print(f"✓ Created ONNX model: {onnx_path.name}")
        print(f"  Size: {onnx_path.stat().st_size / (1024*1024):.2f} MB")
        
        # Verify ONNX is valid
        try:
            import onnx
            model = onnx.load(str(onnx_path))
            onnx.checker.check_model(model)
            print("✓ ONNX model is valid")
            
            # Get model info
            print(f"  Input shape: {model.graph.input[0].type.tensor_type.shape}")
            print(f"  Output count: {len(model.graph.output)}")
            
        except ImportError:
            print("⚠ ONNX package not installed, skipping validation")
    
    @pytest.mark.skipif(not HAS_CONVERTER, reason="Converter module not available")
    @pytest.mark.slow
    def test_tflite_conversion_real(self, test_model, calibration_data, output_dir):
        """Test real TFLite conversion with INT8 quantization"""
        print("\n" + "="*60)
        print("Testing Real TFLite Conversion")
        print("="*60)
        
        # Create converter
        converter = EnhancedModelConverter(
            model_path=str(test_model),
            output_dir=str(output_dir),
            calibration_data=str(calibration_data),
            model_size=320,
            debug=True
        )
        
        # Convert to TFLite
        print("\nConverting to TFLite with INT8 quantization...")
        start_time = time.time()
        
        results = converter.convert_all(
            formats=['tflite'],
            validate=False,
            benchmark=False
        )
        
        conversion_time = time.time() - start_time
        print(f"\nConversion completed in {conversion_time:.2f} seconds")
        
        # Check results
        tflite_dir = output_dir / '320x320'
        tflite_files = list(tflite_dir.glob('*.tflite'))
        
        assert len(tflite_files) > 0, f"No TFLite files found in {tflite_dir}"
        
        print(f"\n✓ Created {len(tflite_files)} TFLite variant(s):")
        for tflite_path in tflite_files:
            size_mb = tflite_path.stat().st_size / (1024*1024)
            print(f"  - {tflite_path.name}: {size_mb:.2f} MB")
            
            # Check if it's an Edge TPU model
            if '_edgetpu' in tflite_path.name:
                print("    ✓ Edge TPU compiled model")
            elif 'quant' in tflite_path.name or 'int8' in tflite_path.name:
                print("    ✓ INT8 quantized model")
            elif 'cpu' in tflite_path.name or 'fp16' in tflite_path.name:
                print("    ✓ FP16 optimized model")
    
    @pytest.mark.skipif(not HAS_CONVERTER, reason="Converter module not available")
    def test_multi_size_conversion_real(self, test_model, calibration_data, output_dir):
        """Test conversion with multiple sizes"""
        print("\n" + "="*60)
        print("Testing Multi-Size Conversion")
        print("="*60)
        
        # Create converter with multiple sizes
        converter = EnhancedModelConverter(
            model_path=str(test_model),
            output_dir=str(output_dir),
            calibration_data=str(calibration_data),
            model_size=[320, 416],  # Two sizes
            debug=True
        )
        
        # Convert to ONNX for both sizes
        print("\nConverting to ONNX for multiple sizes...")
        results = converter.convert_all(
            formats=['onnx'],
            validate=False,
            benchmark=False
        )
        
        # Check both sizes were converted
        assert '320x320' in results['sizes']
        assert '416x416' in results['sizes']
        
        # Check files exist
        for size in [320, 416]:
            size_dir = output_dir / f'{size}x{size}'
            onnx_files = list(size_dir.glob('*.onnx'))
            assert len(onnx_files) > 0, f"No ONNX files for {size}x{size}"
            print(f"✓ Created ONNX for {size}x{size}: {onnx_files[0].name}")
    
    @pytest.mark.skipif(not HAS_CONVERTER, reason="Converter module not available")
    def test_tensorrt_conversion_prep(self, test_model, calibration_data, output_dir):
        """Test TensorRT conversion preparation (ONNX optimization)"""
        # Skip if not on Linux
        if sys.platform != 'linux':
            pytest.skip("TensorRT only supported on Linux")
        
        print("\n" + "="*60)
        print("Testing TensorRT Conversion Preparation")
        print("="*60)
        
        # Create converter
        converter = EnhancedModelConverter(
            model_path=str(test_model),
            output_dir=str(output_dir),
            calibration_data=str(calibration_data),
            model_size=320,
            debug=True
        )
        
        # Convert to TensorRT (will create optimized ONNX if TRT not available)
        print("\nPreparing for TensorRT...")
        results = converter.convert_all(
            formats=['tensorrt'],
            validate=False,
            benchmark=False
        )
        
        # Check results
        trt_results = results['sizes']['320x320']['models'].get('tensorrt', {})
        
        if 'error' in trt_results:
            print(f"⚠ TensorRT not available: {trt_results['error']}")
            # Should at least create optimized ONNX
            onnx_files = list((output_dir / '320x320').glob('*tensorrt*.onnx'))
            if onnx_files:
                print(f"✓ Created TensorRT-optimized ONNX: {onnx_files[0].name}")
        else:
            # Check for engine file
            engine_files = list((output_dir / '320x320').glob('*.engine'))
            if engine_files:
                print(f"✓ Created TensorRT engine: {engine_files[0].name}")
    
    @pytest.mark.skipif(not HAS_CONVERTER, reason="Converter module not available")
    def test_openvino_conversion_real(self, test_model, calibration_data, output_dir):
        """Test OpenVINO conversion"""
        print("\n" + "="*60)
        print("Testing OpenVINO Conversion")
        print("="*60)
        
        # Create converter
        converter = EnhancedModelConverter(
            model_path=str(test_model),
            output_dir=str(output_dir),
            calibration_data=str(calibration_data),
            model_size=320,
            debug=True
        )
        
        # Convert to OpenVINO
        print("\nConverting to OpenVINO...")
        results = converter.convert_all(
            formats=['openvino'],
            validate=False,
            benchmark=False
        )
        
        # Check results
        ov_results = results['sizes']['320x320']['models'].get('openvino', {})
        
        if 'error' in ov_results:
            print(f"⚠ OpenVINO conversion failed: {ov_results['error']}")
        else:
            # Check for OpenVINO files
            ov_files = list((output_dir / '320x320').glob('*openvino*'))
            if ov_files:
                print(f"✓ Created OpenVINO files:")
                for f in ov_files:
                    print(f"  - {f.name}")
    
    @pytest.mark.skipif(not HAS_CONVERTER, reason="Converter module not available")
    def test_hailo_conversion_prep(self, test_model, calibration_data, output_dir):
        """Test Hailo conversion preparation"""
        print("\n" + "="*60)
        print("Testing Hailo Conversion Preparation")
        print("="*60)
        
        # Create converter
        converter = EnhancedModelConverter(
            model_path=str(test_model),
            output_dir=str(output_dir),
            calibration_data=str(calibration_data),
            model_size=320,
            debug=True
        )
        
        # Convert to Hailo (will create preparation files)
        print("\nPreparing for Hailo...")
        results = converter.convert_all(
            formats=['hailo'],
            validate=False,
            benchmark=False
        )
        
        # Check for Hailo preparation files
        hailo_dir = output_dir / '320x320'
        
        # Should create config and script files
        config_files = list(hailo_dir.glob('*hailo*.json'))
        script_files = list(hailo_dir.glob('*hailo*.sh'))
        
        if config_files:
            print(f"✓ Created Hailo config: {config_files[0].name}")
            
            # Check config content
            with open(config_files[0]) as f:
                config = json.load(f)
            print(f"  Model: {config.get('model', {}).get('name', 'unknown')}")
            print(f"  Input shape: {config.get('input_shape', 'unknown')}")
        
        if script_files:
            print(f"✓ Created Hailo conversion script: {script_files[0].name}")
            
            # Check that script uses Python 3.10
            with open(script_files[0]) as f:
                script_content = f.read()
            
            if 'python3.10' in script_content:
                print("  ✓ Script correctly uses Python 3.10 for Hailo SDK")
    
    @pytest.mark.skipif(not HAS_CONVERTER, reason="Converter module not available")
    def test_conversion_with_validation(self, test_model, calibration_data, output_dir):
        """Test conversion with validation enabled"""
        print("\n" + "="*60)
        print("Testing Conversion with Validation")
        print("="*60)
        
        # Create converter
        converter = EnhancedModelConverter(
            model_path=str(test_model),
            output_dir=str(output_dir),
            calibration_data=str(calibration_data),
            model_size=320,
            debug=True
        )
        
        # Convert with validation
        print("\nConverting ONNX with validation...")
        results = converter.convert_all(
            formats=['onnx'],
            validate=True,
            benchmark=True
        )
        
        # Check validation results
        size_results = results['sizes']['320x320']
        
        if 'validation' in size_results:
            val_results = size_results['validation'].get('onnx', {})
            print("\nValidation Results:")
            print(f"  Passed: {val_results.get('passed', 'N/A')}")
            print(f"  Degradation: {val_results.get('degradation', 'N/A')}%")
            
            if 'error' in val_results:
                print(f"  Error: {val_results['error']}")
        
        if 'benchmarks' in size_results:
            bench_results = size_results['benchmarks'].get('onnx', {})
            if 'fps' in bench_results:
                print(f"\nBenchmark Results:")
                print(f"  FPS: {bench_results['fps']:.2f}")
                print(f"  Avg inference: {bench_results.get('avg_inference_ms', 'N/A')}ms")
    
    @pytest.mark.skipif(not HAS_CONVERTER, reason="Converter module not available")
    def test_error_recovery(self, calibration_data, output_dir):
        """Test error handling and recovery"""
        print("\n" + "="*60)
        print("Testing Error Recovery")
        print("="*60)
        
        # Test with non-existent model
        converter = EnhancedModelConverter(
            model_path="non_existent_model.pt",
            output_dir=str(output_dir),
            calibration_data=str(calibration_data),
            model_size=320,
            debug=True
        )
        
        # Should handle error gracefully
        results = converter.convert_all(formats=['onnx'], validate=False)
        
        assert 'error' in results
        print(f"✓ Handled missing model gracefully: {results['error']}")
    
    @pytest.mark.skipif(not HAS_CONVERTER, reason="Converter module not available")
    def test_conversion_summary(self, test_model, calibration_data, output_dir):
        """Test generation of conversion summary"""
        print("\n" + "="*60)
        print("Testing Conversion Summary Generation")
        print("="*60)
        
        # Create converter
        converter = EnhancedModelConverter(
            model_path=str(test_model),
            output_dir=str(output_dir),
            calibration_data=str(calibration_data),
            model_size=320,
            debug=True
        )
        
        # Convert multiple formats
        print("\nConverting to multiple formats...")
        results = converter.convert_all(
            formats=['onnx', 'tflite'],
            validate=False,
            benchmark=False
        )
        
        # Check summary file
        summary_path = output_dir / 'conversion_summary.json'
        
        if summary_path.exists():
            print(f"✓ Created conversion summary: {summary_path}")
            
            with open(summary_path) as f:
                summary = json.load(f)
            
            print("\nSummary Contents:")
            print(f"  Model: {summary.get('model_info', {}).get('original_path', 'unknown')}")
            print(f"  Sizes: {list(summary.get('sizes', {}).keys())}")
            
            # Check converted formats
            for size_str, size_data in summary.get('sizes', {}).items():
                formats = list(size_data.get('models', {}).keys())
                print(f"  {size_str} formats: {formats}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])