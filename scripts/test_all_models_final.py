#!/usr/bin/env python3.12
"""
Final test of all models with proper validation
Including INT8, QAT, TensorRT and multiple sizes
"""
import sys
import os
import logging
import time
from pathlib import Path
import urllib.request
import tarfile
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'converted_models'))

from convert_model import EnhancedModelConverter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test models
TEST_MODELS = [
    'yolov8n',      # Should work
    'yolov9t',      # Testing improved handler
    'yolo_nas_s',   # Should work with ONNX
    'rtdetrv2_s',   # Will likely fail (state dict only)
]

# Test multiple sizes
TEST_SIZES = ["640x640", "416x416", "320x320"]

# Test all formats including INT8 and TensorRT
TEST_FORMATS = ['onnx', 'tflite', 'tensorrt', 'openvino', 'hailo']

# Download calibration data for QAT/INT8
CALIBRATION_DATA_URL = "https://huggingface.co/datasets/mailseth/wildfire-watch/resolve/main/wildfire_calibration_data.tar.gz?download=true"

def download_calibration_data():
    """Download and extract calibration data"""
    tar_path = Path("wildfire_calibration_data.tar.gz")
    
    if not tar_path.exists():
        logger.info(f"Downloading calibration data...")
        urllib.request.urlretrieve(CALIBRATION_DATA_URL, tar_path)
    
    extract_dir = Path(tempfile.mkdtemp(prefix="wildfire_calib_"))
    logger.info(f"Extracting calibration data to {extract_dir}")
    
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(extract_dir)
    
    # Find the actual calibration images directory
    for root, dirs, files in os.walk(extract_dir):
        jpg_files = [f for f in files if f.endswith('.jpg')]
        if len(jpg_files) > 10:
            logger.info(f"Found {len(jpg_files)} calibration images")
            return root
    
    return str(extract_dir)

def test_model(model_name: str, output_base_dir: str = "output/test_final"):
    """Test a single model conversion with all formats and sizes"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing model: {model_name}")
    logger.info(f"{'='*60}")
    
    output_dir = Path(output_base_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'model': model_name,
        'success': False,
        'formats': {},
        'sizes': {},
        'errors': [],
        'validation_passed': False,
        'qat_used': False,
        'int8_success': False,
        'tensorrt_success': False
    }
    
    try:
        # Get calibration data for INT8
        calibration_dir = download_calibration_data()
        
        # Create converter with QAT enabled
        converter = EnhancedModelConverter(
            model_path=model_name,
            output_dir=str(output_dir),
            model_name=model_name,
            model_size=TEST_SIZES,  # Multiple sizes
            calibration_data=calibration_dir,  # For INT8 quantization
            qat_enabled=True,  # Enable QAT for better INT8
            debug=False,  # Less verbose
            auto_download=True
        )
        
        # Convert with validation
        start_time = time.time()
        conversion_results = converter.convert_all(
            formats=TEST_FORMATS,
            validate=True,
            benchmark=False
        )
        elapsed = time.time() - start_time
        
        results['conversion_time'] = elapsed
        
        # Check for conversion errors
        if 'error' in conversion_results:
            results['errors'].append(conversion_results['error'])
            results['success'] = False
        else:
            # Check conversions for each size
            if 'sizes' in conversion_results:
                for size_str in TEST_SIZES:
                    if size_str in conversion_results['sizes']:
                        size_results = conversion_results['sizes'][size_str]
                        results['sizes'][size_str] = {
                            'outputs': {},
                            'errors': [],
                            'validation': {}
                        }
                        
                        # Check outputs
                        outputs = size_results.get('outputs', {})
                        errors = size_results.get('errors', [])
                        validation = size_results.get('validation', {})
                        
                        # Track successful formats for this size
                        for fmt in TEST_FORMATS:
                            found = False
                            for key in outputs:
                                if key.startswith(fmt):
                                    results['sizes'][size_str]['outputs'][fmt] = 'converted'
                                    found = True
                                    
                                    # Check specific format successes
                                    if 'tflite_quantized' in key or 'tflite_qat' in key:
                                        results['int8_success'] = True
                                        results['qat_used'] = True
                                    elif 'tensorrt' in key:
                                        results['tensorrt_success'] = True
                                    elif 'edge_tpu' in key:
                                        results['formats']['edge_tpu'] = 'converted'
                            
                            if not found:
                                # Check if there was an error
                                fmt_error = next((e for e in errors if e['format'] == fmt), None)
                                if fmt_error:
                                    error_msg = str(fmt_error['error'])
                                    results['sizes'][size_str]['outputs'][fmt] = f"failed: {error_msg}"
                                    results['errors'].append(f"{fmt} ({size_str}): {error_msg}")
                                    
                                    # Check for specific errors
                                    if "name 'e' is not defined" in error_msg:
                                        results['errors'].append(f"Code error in {fmt}: name 'e' is not defined")
                                        results['success'] = False
                                        return results
                                    
                                    # Check for Edge TPU error
                                    if "edgetpu-custom-op" in error_msg:
                                        results['formats']['edge_tpu'] = 'failed: custom op error'
                                        results['errors'].append("Edge TPU: Encountered unresolved custom op")
                        
                        # Check validation for this size
                        for fmt, val_result in validation.items():
                            if isinstance(val_result, dict):
                                if val_result.get('acceptable', False):
                                    results['sizes'][size_str]['validation'][fmt] = 'passed'
                                    if size_str == "640x640" and fmt == "onnx":
                                        results['validation_passed'] = True
                                else:
                                    results['sizes'][size_str]['validation'][fmt] = 'failed'
                                    results['errors'].append(f"{fmt} validation failed ({size_str}): {val_result.get('degradation', 'unknown')}")
                
                # Determine overall success
                results['success'] = (
                    len(results['sizes']) > 0 and
                    any(
                        any('converted' in str(v) for v in size_data['outputs'].values())
                        for size_data in results['sizes'].values()
                    ) and
                    len([e for e in results['errors'] if "name 'e' is not defined" in e]) == 0
                )
                
    except Exception as e:
        results['errors'].append(f"Exception: {str(e)}")
        results['success'] = False
    
    return results


def main():
    """Run all model tests"""
    logger.info("Starting comprehensive model conversion tests")
    logger.info(f"Testing formats: {TEST_FORMATS}")
    logger.info(f"Testing sizes: {TEST_SIZES}")
    logger.info("QAT: Enabled for INT8 quantization")
    
    all_results = []
    
    for model in TEST_MODELS:
        result = test_model(model)
        all_results.append(result)
        
        # Summary for this model
        logger.info(f"\n{model} Results:")
        logger.info(f"  Success: {result['success']}")
        logger.info(f"  QAT Used: {result['qat_used']}")
        logger.info(f"  INT8 Success: {result['int8_success']}")
        logger.info(f"  TensorRT Success: {result['tensorrt_success']}")
        logger.info(f"  Validation: {'PASSED' if result['validation_passed'] else 'FAILED'}")
        
        # Show size-specific results
        for size, size_data in result['sizes'].items():
            successful_formats = [fmt for fmt, status in size_data['outputs'].items() if status == 'converted']
            if successful_formats:
                logger.info(f"  {size}: {', '.join(successful_formats)}")
        
        if result['errors']:
            logger.info(f"  First Error: {result['errors'][0]}")
    
    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'='*60}")
    
    successful = sum(1 for r in all_results if r['success'])
    total = len(all_results)
    
    logger.info(f"Total models: {total}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {total - successful}")
    
    # Detailed summary
    for result in all_results:
        status = "✅" if result['success'] else "❌"
        val_status = "✓" if result['validation_passed'] else "✗"
        qat_status = "✓" if result['qat_used'] else "✗"
        int8_status = "✓" if result['int8_success'] else "✗"
        trt_status = "✓" if result['tensorrt_success'] else "✗"
        
        logger.info(f"\n  {status} {result['model']}:")
        logger.info(f"     Validation: {val_status}, QAT: {qat_status}, INT8: {int8_status}, TensorRT: {trt_status}")
        
        # Count successful conversions
        total_conversions = 0
        successful_conversions = 0
        for size_data in result['sizes'].values():
            for fmt, status in size_data['outputs'].items():
                total_conversions += 1
                if status == 'converted':
                    successful_conversions += 1
        
        if successful_conversions > 0:
            logger.info(f"     Conversions: {successful_conversions}/{total_conversions} successful")
        
        if not result['success'] and result['errors']:
            logger.info(f"     First Error: {result['errors'][0]}")
    
    # Check for specific issues
    code_errors = []
    validation_failures = []
    edge_tpu_failures = []
    
    for result in all_results:
        for error in result['errors']:
            if "name 'e' is not defined" in error:
                code_errors.append(result['model'])
            if "validation failed" in error.lower():
                validation_failures.append(result['model'])
            if "edgetpu-custom-op" in error:
                edge_tpu_failures.append(result['model'])
    
    if code_errors:
        logger.error(f"\nModels with code errors: {code_errors}")
    
    if validation_failures:
        logger.warning(f"\nModels with validation failures: {validation_failures}")
        
    if edge_tpu_failures:
        logger.warning(f"\nModels with Edge TPU failures: {edge_tpu_failures}")
    
    # QAT/INT8 summary
    qat_models = [r['model'] for r in all_results if r['qat_used']]
    int8_models = [r['model'] for r in all_results if r['int8_success']]
    trt_models = [r['model'] for r in all_results if r['tensorrt_success']]
    
    logger.info(f"\nQAT-enabled models: {qat_models}")
    logger.info(f"INT8 successful models: {int8_models}")
    logger.info(f"TensorRT successful models: {trt_models}")
    
    return all_results


if __name__ == "__main__":
    results = main()