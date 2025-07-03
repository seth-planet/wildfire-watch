#!/usr/bin/env python3.10
"""Updated Hailo conversion method that performs actual conversion.

This file contains the updated convert_to_hailo_optimized method that should
replace the existing one in convert_model.py. It uses the hailo_converter
module to perform the actual conversion instead of just creating config files.
"""

import logging
from pathlib import Path
from typing import Dict, Optional
import subprocess
import sys

logger = logging.getLogger(__name__)


def convert_to_hailo_optimized(self) -> Dict[str, Path]:
    """Convert to Hailo HEF format with actual conversion.
    
    This method performs the complete Hailo conversion pipeline:
    1. Ensures ONNX model exists
    2. Runs Hailo conversion for both targets (hailo8 and hailo8l)
    3. Returns paths to converted HEF files
    
    Returns:
        Dictionary with paths to converted HEF files
    """
    logger.info("Starting Hailo HEF conversion...")
    
    results = {}
    
    # Ensure we have an ONNX model first
    onnx_suffix = "_qat" if self.qat_enabled else ""
    onnx_path = self.output_dir / f"{self.model_name}{onnx_suffix}.onnx"
    
    if not onnx_path.exists():
        logger.info("ONNX model not found, converting from PyTorch...")
        onnx_path = self.convert_to_onnx_optimized(qat_prepare=self.qat_enabled)
        if not onnx_path or not onnx_path.exists():
            logger.error("Failed to create ONNX model for Hailo conversion")
            return results
    
    # Import hailo_converter module
    hailo_converter_path = Path(__file__).parent / 'hailo_converter.py'
    if not hailo_converter_path.exists():
        logger.error(f"hailo_converter.py not found at {hailo_converter_path}")
        logger.info("Creating conversion scripts instead...")
        # Fall back to original behavior
        return self._convert_to_hailo_scripts_only(onnx_path)
    
    # Check if we're running with Python 3.10 (required by Hailo SDK)
    if sys.version_info[:2] != (3, 10):
        logger.warning(f"Hailo SDK requires Python 3.10, current version is {sys.version}")
        logger.info("Attempting conversion with subprocess...")
        
        # Convert for both Hailo targets
        for target in ['hailo8', 'hailo8l']:
            try:
                output_path = self.output_dir / f"{self.model_name}_{target}.hef"
                
                cmd = [
                    'python3.10',
                    str(hailo_converter_path),
                    str(onnx_path),
                    '--output', str(output_path),
                    '--target', target,
                    '--batch-size', '1'  # Frigate always uses batch size 1
                ]
                
                # Add calibration data if available
                if self.calibration_dir:
                    cmd.extend(['--calibration-data', str(self.calibration_dir)])
                
                logger.info(f"Running Hailo conversion for {target}...")
                logger.debug(f"Command: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=3600  # 60 minute timeout
                )
                
                if result.returncode == 0 and output_path.exists():
                    logger.info(f"Successfully converted to {target}: {output_path}")
                    results[target] = output_path
                    
                    # Also create size-specific variants
                    size_str = f"{self.model_size[0]}x{self.model_size[1]}"
                    size_specific_path = self.output_dir / f"{self.model_name}_{size_str}_{target}.hef"
                    if output_path != size_specific_path:
                        import shutil
                        shutil.copy2(output_path, size_specific_path)
                        results[f"{target}_{size_str}"] = size_specific_path
                else:
                    logger.error(f"Hailo conversion failed for {target}")
                    if result.stderr:
                        logger.error(f"Error: {result.stderr}")
                    if result.stdout:
                        logger.debug(f"Output: {result.stdout}")
                        
            except subprocess.TimeoutExpired:
                logger.error(f"Hailo conversion timed out for {target}")
            except Exception as e:
                logger.error(f"Hailo conversion error for {target}: {e}")
    
    else:
        # We're running with Python 3.10, can import directly
        try:
            from hailo_converter import HailoConverter
            
            # Convert for both targets
            for target in ['hailo8', 'hailo8l']:
                try:
                    output_path = self.output_dir / f"{self.model_name}_{target}.hef"
                    
                    logger.info(f"Converting to {target}...")
                    converter = HailoConverter(
                        model_path=str(onnx_path),
                        output_path=str(output_path),
                        target=target,
                        calibration_data=self.calibration_dir,
                        batch_size=1
                    )
                    
                    hef_path = converter.convert()
                    if hef_path and hef_path.exists():
                        results[target] = hef_path
                        
                        # Validate if requested
                        if converter.validate_conversion():
                            logger.info(f"Validation passed for {target}")
                        else:
                            logger.warning(f"Validation failed for {target}")
                        
                        # Create size-specific variant
                        size_str = f"{self.model_size[0]}x{self.model_size[1]}"
                        size_specific_path = self.output_dir / f"{self.model_name}_{size_str}_{target}.hef"
                        if hef_path != size_specific_path:
                            import shutil
                            shutil.copy2(hef_path, size_specific_path)
                            results[f"{target}_{size_str}"] = size_specific_path
                            
                except Exception as e:
                    logger.error(f"Failed to convert to {target}: {e}")
                    
        except ImportError as e:
            logger.error(f"Failed to import hailo_converter: {e}")
            logger.info("Falling back to script generation...")
            return self._convert_to_hailo_scripts_only(onnx_path)
    
    # If we successfully converted any models, also create the config files
    if results:
        # Create optimized Hailo configuration for reference
        config = self._create_optimized_hailo_config()
        config_path = self.output_dir / f"{self.model_name}_hailo_config.json"
        with open(config_path, 'w') as f:
            import json
            json.dump(config, f, indent=2)
        results['config'] = config_path
        
        logger.info(f"Hailo conversion complete. Converted {len([k for k in results if k.endswith('.hef')])} models")
    else:
        logger.warning("No Hailo models were successfully converted")
        # Fall back to creating scripts
        return self._convert_to_hailo_scripts_only(onnx_path)
    
    return results


def _convert_to_hailo_scripts_only(self, onnx_path: Path) -> Dict[str, Path]:
    """Original method that only creates scripts without conversion.
    
    This is the fallback when actual conversion isn't possible.
    """
    logger.info("Creating Hailo conversion scripts (no actual conversion)...")
    
    results = {}
    
    # Create optimized Hailo configuration
    config = self._create_optimized_hailo_config()
    config_path = self.output_dir / f"{self.model_name}_hailo_config.json"
    with open(config_path, 'w') as f:
        import json
        json.dump(config, f, indent=2)
    results['config'] = config_path
    
    # Create conversion script with QAT optimizations
    script_path = self._create_hailo_qat_script(onnx_path, config_path)
    results['script'] = script_path
    
    # Create Docker compose with optimization
    compose_path = self._create_hailo_docker_optimized()
    results['docker_compose'] = compose_path
    
    # Create post-training optimization script
    pto_script = self._create_hailo_pto_script(onnx_path)
    results['pto_script'] = pto_script
    
    return results


# Instructions for integration:
# 1. Replace the existing convert_to_hailo_optimized method in convert_model.py
#    with the one above (starting at line 1579)
# 2. Add the _convert_to_hailo_scripts_only method after it
# 3. The new method will attempt actual conversion and fall back to script
#    generation if the converter isn't available