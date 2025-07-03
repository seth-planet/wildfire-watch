#!/usr/bin/env python3.10
"""Direct Hailo conversion script using basic CLI commands.

This script provides a simpler approach to converting ONNX models to HEF
format using the basic Hailo CLI commands without complex options.
"""

import os
import sys
import subprocess
import shutil
import logging
from pathlib import Path
import tempfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def convert_to_hailo(onnx_path: str, output_dir: str = "./hailo_output", target: str = "hailo8l"):
    """Convert ONNX model to Hailo HEF format.
    
    Args:
        onnx_path: Path to ONNX model
        output_dir: Output directory for HEF files
        target: Target hardware (hailo8 or hailo8l)
    """
    onnx_path = Path(onnx_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    
    model_name = onnx_path.stem
    work_dir = Path(tempfile.mkdtemp(prefix=f"hailo_{model_name}_"))
    
    try:
        logger.info(f"Converting {onnx_path} to Hailo {target} format...")
        logger.info(f"Working directory: {work_dir}")
        
        # Step 1: Parse ONNX to HAR
        har_path = work_dir / f"{model_name}.har"
        
        logger.info("Step 1: Parsing ONNX to HAR...")
        parse_cmd = [
            'hailo', 'parser', 'onnx',
            str(onnx_path),
            '--har-path', str(har_path),
            '--start-node-names', 'images',
            '--end-node-names', 'output0',
            '--hw-arch', target,
            '-y'
        ]
        
        logger.debug(f"Parse command: {' '.join(parse_cmd)}")
        result = subprocess.run(parse_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Parse failed: {result.stderr}")
            return False
        
        if not har_path.exists():
            logger.error("HAR file was not created")
            return False
        
        logger.info(f"HAR created: {har_path}")
        
        # Step 2: Optimize (quantize) the HAR
        optimized_har = work_dir / f"{model_name}_optimized.har"
        
        logger.info("Step 2: Optimizing HAR (quantization)...")
        
        # First, let's check if we have calibration data
        calib_dir = Path("./calibration_data")
        calib_npy = None
        
        if calib_dir.exists():
            # Create a simple calibration numpy file from images
            logger.info("Creating calibration data from images...")
            calib_script = f"""
import numpy as np
import cv2
from pathlib import Path

calib_dir = Path("{calib_dir}")
images = list(calib_dir.glob("*.jpg")) + list(calib_dir.glob("*.png"))
images = images[:100]  # Use first 100 images

calib_data = []
for img_path in images:
    img = cv2.imread(str(img_path))
    if img is not None:
        img = cv2.resize(img, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        calib_data.append(img)

calib_array = np.array(calib_data)
np.save("{work_dir}/calibration.npy", calib_array)
print(f"Saved calibration data: shape={{calib_array.shape}}")
"""
            
            result = subprocess.run(['python3.10', '-c', calib_script], capture_output=True, text=True)
            if result.returncode == 0:
                calib_npy = work_dir / "calibration.npy"
                logger.info("Calibration data created")
        
        # Run optimization
        if calib_npy and calib_npy.exists():
            optimize_cmd = [
                'hailo', 'optimize',
                str(har_path),
                '--hw-arch', target,
                '--calib-set-path', str(calib_npy),
                '--output-har-path', str(optimized_har)
            ]
        else:
            logger.warning("No calibration data, using random calibration")
            optimize_cmd = [
                'hailo', 'optimize',
                str(har_path),
                '--hw-arch', target,
                '--use-random-calib-set',
                '--output-har-path', str(optimized_har)
            ]
        
        logger.debug(f"Optimize command: {' '.join(optimize_cmd)}")
        result = subprocess.run(optimize_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Optimize failed: {result.stderr}")
            # Try without output path
            logger.info("Retrying optimization with default output...")
            optimize_cmd = [
                'hailo', 'optimize',
                str(har_path),
                '--hw-arch', target,
                '--use-random-calib-set'
            ]
            result = subprocess.run(optimize_cmd, capture_output=True, text=True, cwd=work_dir)
            
            if result.returncode != 0:
                logger.error(f"Optimization retry failed: {result.stderr}")
                return False
            
            # Look for the output HAR
            optimized_files = list(work_dir.glob("*_quantized.har"))
            if optimized_files:
                optimized_har = optimized_files[0]
                logger.info(f"Found optimized HAR: {optimized_har}")
            else:
                logger.error("No optimized HAR found")
                return False
        
        logger.info(f"HAR optimized: {optimized_har}")
        
        # Step 3: Compile HAR to HEF
        logger.info("Step 3: Compiling HAR to HEF...")
        
        compile_cmd = [
            'hailo', 'compiler',
            str(optimized_har),
            '--hw-arch', target,
            '--output-dir', str(work_dir)
        ]
        
        logger.debug(f"Compile command: {' '.join(compile_cmd)}")
        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Compile failed: {result.stderr}")
            return False
        
        # Find the output HEF
        hef_files = list(work_dir.glob("*.hef"))
        if not hef_files:
            logger.error("No HEF file created")
            return False
        
        hef_path = hef_files[0]
        logger.info(f"HEF created: {hef_path}")
        
        # Copy to output directory
        output_hef = output_dir / f"{model_name}_{target}.hef"
        shutil.copy2(hef_path, output_hef)
        logger.info(f"HEF copied to: {output_hef}")
        
        # Create metadata
        metadata = {
            'source_model': str(onnx_path),
            'target': target,
            'optimization': 'random_calib' if not calib_npy else 'calibration_data',
            'file_size_mb': output_hef.stat().st_size / 1024 / 1024
        }
        
        import json
        metadata_path = output_hef.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Conversion complete! Output: {output_hef}")
        return True
        
    except Exception as e:
        logger.error(f"Conversion error: {e}")
        return False
    finally:
        # Cleanup
        if work_dir.exists():
            logger.debug(f"Cleaning up: {work_dir}")
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3.10 convert_to_hailo_direct.py <onnx_model>")
        sys.exit(1)
    
    onnx_model = sys.argv[1]
    
    # Convert for both targets
    for target in ['hailo8', 'hailo8l']:
        success = convert_to_hailo(onnx_model, target=target)
        if success:
            print(f"✓ Successfully converted to {target}")
        else:
            print(f"✗ Failed to convert to {target}")