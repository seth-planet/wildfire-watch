#!/usr/bin/env python3.12
"""
Demo: Model Conversion with Accuracy Validation
Shows how the enhanced model converter validates accuracy across formats
"""
import sys
import logging
from pathlib import Path

sys.path.insert(0, 'converted_models')

from convert_model import EnhancedModelConverter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run demo of model conversion with accuracy validation"""
    
    # Check if model exists
    model_path = Path("yolov8n.pt")
    if not model_path.exists():
        # Try other locations
        alt_paths = [
            Path("converted_models/yolov8n.pt"),
            Path("/tmp/yolov8n.pt")
        ]
        for path in alt_paths:
            if path.exists():
                model_path = path
                break
        else:
            logger.error("No YOLOv8n model found. Please download it first.")
            logger.info("Run: wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt")
            return
    
    logger.info(f"Using model: {model_path}")
    
    # Create converter with multiple sizes
    converter = EnhancedModelConverter(
        model_path=str(model_path),
        output_dir="demo_output",
        model_name="yolov8n_demo",
        model_size=["640x640", "416x416"],  # Test two sizes
        qat_enabled=False,
        debug=True
    )
    
    logger.info("\n" + "="*60)
    logger.info("Starting model conversion with accuracy validation")
    logger.info("="*60)
    
    # Convert to ONNX and TFLite with validation
    results = converter.convert_all(
        formats=['onnx', 'tflite'],
        validate=True,  # Enable accuracy validation
        benchmark=False
    )
    
    # Display results
    logger.info("\n" + "="*60)
    logger.info("CONVERSION RESULTS")
    logger.info("="*60)
    
    if 'baseline_metrics' in results:
        baseline = results['baseline_metrics']
        logger.info(f"\nBaseline PyTorch Model:")
        logger.info(f"  - mAP@50: {baseline['mAP50']:.3f}")
        logger.info(f"  - mAP@50-95: {baseline['mAP50_95']:.3f}")
        logger.info(f"  - Inference time: {baseline['inference_time_ms']:.1f}ms")
        logger.info(f"  - Model size: {baseline['model_size_mb']:.1f}MB")
    
    # Show results for each size
    for size_str, size_results in results['sizes'].items():
        logger.info(f"\n{size_str} Results:")
        
        # Show conversions
        if 'outputs' in size_results:
            logger.info("  Converted formats:")
            for fmt, path in size_results['outputs'].items():
                logger.info(f"    - {fmt}: {Path(path).name}")
        
        # Show validation results
        if 'validation' in size_results:
            logger.info("  Accuracy validation:")
            for fmt, val in size_results['validation'].items():
                if 'metrics' in val and 'acceptable' in val:
                    metrics = val['metrics']
                    status = "✅ PASSED" if val['acceptable'] else "❌ FAILED"
                    logger.info(f"    - {fmt}: {status}")
                    logger.info(f"        mAP@50: {metrics['mAP50']:.3f}")
                    if 'degradation' in val and 'mAP50_degradation' in val['degradation']:
                        logger.info(f"        Degradation: {val['degradation']['mAP50_degradation']:.1f}%")
    
    # Show accuracy report location
    if 'accuracy_report' in results:
        logger.info(f"\n📊 Full accuracy report saved to: {results['accuracy_report']}")
        
        # Display report content
        with open(results['accuracy_report']) as f:
            report = f.read()
            logger.info("\n" + "="*60)
            logger.info("ACCURACY REPORT PREVIEW")
            logger.info("="*60)
            # Show first 30 lines
            lines = report.split('\n')[:30]
            logger.info('\n'.join(lines))
            if len(report.split('\n')) > 30:
                logger.info("... (truncated, see full report for details)")
    
    logger.info("\n✅ Demo completed! Check 'demo_output' directory for converted models.")


if __name__ == '__main__':
    main()