#!/usr/bin/env python3.12
"""
Test Model Conversions with Accuracy Validation
Tests multiple YOLO variants with QAT preference for INT8
"""
import sys
import os
import logging
import time
from pathlib import Path
import urllib.request
import tarfile
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'converted_models'))

from convert_model import EnhancedModelConverter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test models as available in convert_model.py
TEST_MODELS = [
    'yolov8n',      # YOLOv8 nano
    # 'yolov8s',      # YOLOv8 small - takes ~24 minutes
    'yolov9t',      # YOLOv9 tiny - converted version
    'yolo_nas_s',   # YOLO-NAS small - URL updated  
    'rtdetrv2_s',   # RT-DETRv2 small - URL updated
]

# Test configurations
TEST_SIZES = ["640x640"]  # Start with one size to test new models
# Test all available formats
TEST_FORMATS = ['onnx', 'tflite']  # Start with basic formats for testing

# Calibration data URL
CALIBRATION_DATA_URL = "https://huggingface.co/datasets/mailseth/wildfire-watch/resolve/main/wildfire_calibration_data.tar.gz?download=true"

# Global variable to store calibration data path
CALIBRATION_DATA_PATH = None


def download_calibration_data():
    """Download and extract calibration data from HuggingFace"""
    global CALIBRATION_DATA_PATH
    
    if CALIBRATION_DATA_PATH and Path(CALIBRATION_DATA_PATH).exists():
        return CALIBRATION_DATA_PATH
    
    # Check if already downloaded
    tar_path = Path("wildfire_calibration_data.tar.gz")
    
    if not tar_path.exists():
        logger.info(f"Downloading calibration data from HuggingFace...")
        try:
            urllib.request.urlretrieve(CALIBRATION_DATA_URL, tar_path)
            logger.info(f"Downloaded calibration data to {tar_path}")
        except Exception as e:
            logger.error(f"Failed to download calibration data: {e}")
            return None
    
    # Extract to temp directory
    extract_dir = Path(tempfile.mkdtemp(prefix="wildfire_calib_"))
    logger.info(f"Extracting calibration data to {extract_dir}")
    
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(extract_dir)
    
    # Find the actual calibration images directory
    for root, dirs, files in os.walk(extract_dir):
        jpg_files = [f for f in files if f.endswith('.jpg')]
        if len(jpg_files) > 10:  # Found calibration images
            CALIBRATION_DATA_PATH = root
            logger.info(f"Found {len(jpg_files)} calibration images in {root}")
            return root
    
    # If not found in subdirs, use extract_dir
    CALIBRATION_DATA_PATH = str(extract_dir)
    return str(extract_dir)


def test_model_conversion(model_name: str, output_base_dir: str = "test_conversions"):
    """Test conversion of a single model with accuracy validation"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing model: {model_name}")
    logger.info(f"{'='*60}")
    
    output_dir = Path(output_base_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Get calibration data
        calibration_dir = download_calibration_data()
        if not calibration_dir:
            logger.warning("No calibration data available, quantization will be limited")
        
        # Create converter with auto_download enabled
        logger.info(f"Creating converter with QAT enabled and calibration data: {calibration_dir}")
        converter = EnhancedModelConverter(
            model_path=model_name,  # Just pass the model name
            output_dir=str(output_dir),
            model_name=model_name,
            model_size=TEST_SIZES,
            calibration_data=str(calibration_dir) if calibration_dir else None,
            qat_enabled=True,  # Always enable QAT for INT8 formats
            debug=True,
            auto_download=True  # Enable auto-download
        )
        logger.info(f"Converter initialized. QAT enabled: {converter.qat_enabled}, QAT compatible: {getattr(converter.model_info, 'qat_compatible', 'Unknown')}")
        
        # Convert with accuracy validation
        start_time = time.time()
        results = converter.convert_all(
            formats=TEST_FORMATS,
            validate=True,  # Enable validation
            benchmark=False  # Skip benchmarking for faster testing
        )
        elapsed = time.time() - start_time
        
        # Check for errors in conversion
        if 'error' in results:
            raise RuntimeError(f"Conversion failed: {results['error']}")
        
        # Check if any conversions actually happened
        if not results.get('sizes'):
            raise RuntimeError("No conversions were performed")
        
        # Check for calibration errors
        calibration_errors = []
        qat_errors = []
        
        # Summarize results
        logger.info(f"\n{model_name} Conversion Summary:")
        logger.info(f"  - Conversion time: {elapsed:.1f}s")
        
        if 'baseline_metrics' in results:
            baseline = results['baseline_metrics']
            logger.info(f"  - Baseline mAP@50: {baseline.get('mAP50', 0):.3f}")
            logger.info(f"  - Baseline mAP@50-95: {baseline.get('mAP50_95', 0):.3f}")
        
        # Check each size
        total_conversions = 0
        successful_conversions = 0
        failed_conversions = []
        failed_validations = []
        
        for size_str, size_results in results.get('sizes', {}).items():
            logger.info(f"\n  Size {size_str}:")
            
            # Check for errors first
            errors = size_results.get('errors', [])
            if errors:
                for error in errors:
                    error_msg = str(error.get('error', ''))
                    logger.error(f"    ❌ {error['format']}: {error_msg}")
                    failed_conversions.append(f"{error['format']} ({size_str})")
                    
                    # Check for specific error types
                    if 'No calibration directory available' in error_msg:
                        calibration_errors.append(f"{error['format']} ({size_str})")
                    if '_check_qat_compatibility' in error_msg or 'QAT compatibility check failed' in error_msg:
                        qat_errors.append(f"{error['format']} ({size_str})")
            
            # Count conversions and check models
            outputs = size_results.get('outputs', {})
            models = size_results.get('models', {})
            validations = size_results.get('validation', {})
            
            # Check expected formats
            for expected_format in TEST_FORMATS:
                total_conversions += 1
                
                # Check if format had an error
                format_error = next((e for e in errors if e['format'] == expected_format), None)
                if format_error:
                    continue  # Already counted as failed
                
                # Check if format was converted
                format_found = False
                for output_key in outputs:
                    if output_key.startswith(expected_format):
                        format_found = True
                        
                        # Check validation if available
                        if output_key in validations:
                            if validations[output_key].get('acceptable', False):
                                successful_conversions += 1
                                logger.info(f"    ✅ {output_key}: PASSED validation")
                            else:
                                failed_validations.append(f"{output_key} ({size_str})")
                                logger.warning(f"    ❌ {output_key}: FAILED validation")
                                if 'degradation' in validations[output_key]:
                                    deg = validations[output_key]['degradation']
                                    if isinstance(deg, dict):
                                        logger.warning(f"       mAP@50 degradation: {deg.get('mAP50_degradation', 0):.1f}%")
                                    else:
                                        logger.warning(f"       Degradation: {deg}")
                        else:
                            # No validation data but converted
                            successful_conversions += 1
                            logger.info(f"    ✓ {output_key}: Converted (no validation)")
                
                if not format_found and not format_error:
                    # Format was expected but not found and no error recorded
                    logger.error(f"    ❌ {expected_format}: Not found in outputs")
                    failed_conversions.append(f"{expected_format} ({size_str})")
        
        # Overall summary
        logger.info(f"\n  Overall: {successful_conversions}/{total_conversions} successful")
        if failed_conversions:
            logger.error(f"  Failed conversions: {', '.join(failed_conversions)}")
        if failed_validations:
            logger.warning(f"  Failed validations: {', '.join(failed_validations)}")
        if calibration_errors:
            logger.error(f"  Calibration errors: {', '.join(calibration_errors)}")
        if qat_errors:
            logger.error(f"  QAT compatibility errors: {', '.join(qat_errors)}")
        
        # Check if accuracy report was generated
        if 'accuracy_report' in results:
            logger.info(f"  Accuracy report: {results['accuracy_report']}")
        
        # Determine overall success - fail if any conversions failed or critical errors occurred
        overall_success = (len(failed_conversions) == 0 and 
                          len(calibration_errors) == 0 and 
                          len(qat_errors) == 0)
        
        if not overall_success:
            error_summary = []
            if failed_conversions:
                error_summary.append(f"{len(failed_conversions)} conversion failures")
            if calibration_errors:
                error_summary.append(f"{len(calibration_errors)} calibration errors")
            if qat_errors:
                error_summary.append(f"{len(qat_errors)} QAT errors")
            logger.error(f"Model {model_name} had: {', '.join(error_summary)}")
        
        return overall_success, results
        
    except Exception as e:
        logger.error(f"Error converting {model_name}: {e}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return False, {'error': str(e)}


def main():
    """Run all model conversion tests"""
    logger.info("Starting model conversion tests with accuracy validation")
    logger.info(f"Models to test: {', '.join(TEST_MODELS)}")
    logger.info(f"Sizes: {', '.join(TEST_SIZES)}")
    logger.info(f"Formats: {', '.join(TEST_FORMATS)}")
    logger.info("Note: INT8 formats will automatically use QAT when available\n")
    
    results_summary = {}
    start_time = time.time()
    
    # Test each model
    for model_name in TEST_MODELS:
        success, results = test_model_conversion(model_name)
        results_summary[model_name] = {
            'success': success,
            'results': results
        }
        
        # Small delay between models
        time.sleep(2)
    
    total_time = time.time() - start_time
    
    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total test time: {total_time:.1f}s")
    
    successful_models = sum(1 for r in results_summary.values() if r['success'])
    failed_models = len(TEST_MODELS) - successful_models
    
    logger.info(f"Models tested: {len(TEST_MODELS)}")
    logger.info(f"Successful: {successful_models}")
    logger.info(f"Failed: {failed_models}")
    
    # Detailed results
    for model_name, summary in results_summary.items():
        if summary['success']:
            logger.info(f"  ✅ {model_name}: Conversion successful")
        else:
            logger.info(f"  ❌ {model_name}: Conversion failed")
            if 'results' in summary and 'error' in summary['results']:
                logger.error(f"     Error: {summary['results']['error']}")
    
    # Check QAT usage
    logger.info("\nQAT Usage Summary:")
    for model_name, summary in results_summary.items():
        if summary['success'] and summary['results']:
            # Check if QAT was used for INT8
            qat_used = False
            for size_results in summary['results'].get('sizes', {}).values():
                outputs = size_results.get('outputs', {})
                # Check for QAT indicators in output files
                for output_key, output_path in outputs.items():
                    if '_qat' in str(output_path) or 'qat' in output_key.lower() or (output_key == 'quantized' and '_qat' in str(output_path)):
                        qat_used = True
                        logger.debug(f"Found QAT indicator: {output_key} -> {output_path}")
                        break
                # Also check models dict
                models = size_results.get('models', {})
                for model_key in models:
                    if '_qat' in model_key or 'qat' in model_key.lower():
                        qat_used = True
                        break
                if qat_used:
                    break
            
            if qat_used:
                logger.info(f"  {model_name}: QAT used for INT8 formats ✓")
            else:
                logger.info(f"  {model_name}: Standard INT8 (no QAT)")
    
    logger.info("\n✅ Test completed! Check individual model directories for converted files.")
    
    # Exit with error code if any models failed
    if failed_models > 0:
        logger.error(f"\n❌ {failed_models} model(s) failed to convert!")
        sys.exit(1)
    else:
        logger.info(f"\n✅ All {successful_models} models converted successfully!")
        sys.exit(0)


if __name__ == '__main__':
    main()