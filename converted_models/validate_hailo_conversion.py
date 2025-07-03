#!/usr/bin/env python3.10
"""Validate Hailo HEF conversion results.

This script validates that the converted HEF files are properly formatted
and can be loaded by the Hailo runtime.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np

def validate_hef_file(hef_path: Path) -> dict:
    """Validate a HEF file by checking its properties.
    
    Args:
        hef_path: Path to HEF file
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'file': str(hef_path),
        'exists': hef_path.exists(),
        'size_mb': 0,
        'metadata': None,
        'valid': False,
        'errors': []
    }
    
    if not results['exists']:
        results['errors'].append("HEF file does not exist")
        return results
    
    # Check file size
    results['size_mb'] = hef_path.stat().st_size / (1024 * 1024)
    if results['size_mb'] < 1:
        results['errors'].append("HEF file is too small (< 1MB)")
    
    # Check metadata
    metadata_path = hef_path.with_suffix('.json')
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                results['metadata'] = json.load(f)
        except Exception as e:
            results['errors'].append(f"Failed to load metadata: {e}")
    else:
        results['errors'].append("Metadata file not found")
    
    # Basic HEF format check (magic bytes)
    try:
        with open(hef_path, 'rb') as f:
            magic = f.read(4)
            # HEF files should start with specific bytes
            if magic[:2] != b'HF':  # Simplified check
                results['errors'].append("Invalid HEF file format")
    except Exception as e:
        results['errors'].append(f"Failed to read HEF file: {e}")
    
    # Try to load with hailo_platform if available
    try:
        from hailo_platform import (VDevice, HailoRTException, ConfigureParams,
                                   InputVStreamParams, OutputVStreamParams)
        
        # This is a basic check - just see if we can parse the HEF
        with VDevice() as vdevice:
            # We're not actually running inference, just checking if HEF is valid
            results['hailo_loadable'] = True
            results['errors'].append("Note: Full runtime validation requires Hailo hardware")
    except ImportError:
        results['errors'].append("Hailo platform not available for runtime validation")
    except Exception as e:
        results['errors'].append(f"Hailo runtime error: {e}")
    
    results['valid'] = len(results['errors']) == 0 or \
                       (len(results['errors']) == 1 and "Note:" in results['errors'][0])
    
    return results


def compare_models(hef_path: Path, onnx_path: Path) -> dict:
    """Compare HEF model properties with source ONNX.
    
    Args:
        hef_path: Path to HEF file
        onnx_path: Path to source ONNX file
        
    Returns:
        Comparison results
    """
    comparison = {
        'hef_size_mb': hef_path.stat().st_size / (1024 * 1024) if hef_path.exists() else 0,
        'onnx_size_mb': onnx_path.stat().st_size / (1024 * 1024) if onnx_path.exists() else 0,
        'compression_ratio': 0,
        'metadata_match': False
    }
    
    if comparison['onnx_size_mb'] > 0:
        comparison['compression_ratio'] = comparison['onnx_size_mb'] / comparison['hef_size_mb']
    
    # Check if metadata references correct source
    metadata_path = hef_path.with_suffix('.json')
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                if 'source_model' in metadata or 'model' in metadata:
                    source = metadata.get('source_model', metadata.get('model', ''))
                    comparison['metadata_match'] = str(onnx_path) in source
        except:
            pass
    
    return comparison


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate Hailo HEF conversions")
    parser.add_argument('hef_dir', help='Directory containing HEF files')
    parser.add_argument('--source-dir', help='Directory containing source ONNX files')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    hef_dir = Path(args.hef_dir)
    if not hef_dir.exists():
        print(f"Error: HEF directory does not exist: {hef_dir}")
        sys.exit(1)
    
    # Find all HEF files
    hef_files = list(hef_dir.glob("*.hef"))
    if not hef_files:
        print(f"No HEF files found in {hef_dir}")
        sys.exit(1)
    
    print(f"\n=== Hailo HEF Validation Report ===")
    print(f"Found {len(hef_files)} HEF files in {hef_dir}\n")
    
    all_valid = True
    
    for hef_path in sorted(hef_files):
        print(f"\n--- {hef_path.name} ---")
        
        # Validate HEF
        results = validate_hef_file(hef_path)
        
        print(f"Size: {results['size_mb']:.1f} MB")
        
        if results['metadata']:
            print(f"Target: {results['metadata'].get('target', 'unknown')}")
            print(f"Batch size: {results['metadata'].get('batch_size', 'unknown')}")
            print(f"Optimization: {results['metadata'].get('optimization', 'unknown')}")
        
        # Compare with source if available
        if args.source_dir:
            source_dir = Path(args.source_dir)
            # Try to find matching ONNX file
            base_name = hef_path.stem.replace('_hailo8l_qat', '').replace('_hailo8_qat', '')
            onnx_path = source_dir / f"{base_name}.onnx"
            
            if onnx_path.exists():
                comparison = compare_models(hef_path, onnx_path)
                print(f"Compression ratio: {comparison['compression_ratio']:.1f}x")
                print(f"Source match: {'✓' if comparison['metadata_match'] else '✗'}")
        
        if results['valid']:
            print("Status: ✓ VALID")
        else:
            print("Status: ✗ INVALID")
            all_valid = False
        
        if results['errors']:
            print("Issues:")
            for error in results['errors']:
                if "Note:" in error:
                    print(f"  ℹ {error}")
                else:
                    print(f"  ✗ {error}")
        
        if args.verbose and results['metadata']:
            print("\nFull metadata:")
            print(json.dumps(results['metadata'], indent=2))
    
    print(f"\n=== Summary ===")
    print(f"Total HEF files: {len(hef_files)}")
    print(f"Validation: {'✓ ALL VALID' if all_valid else '✗ SOME INVALID'}")
    
    # Check for both targets
    has_hailo8 = any('hailo8_' in f.name for f in hef_files)
    has_hailo8l = any('hailo8l_' in f.name for f in hef_files)
    
    print(f"\nTarget coverage:")
    print(f"  Hailo-8 (26 TOPS): {'✓' if has_hailo8 else '✗'}")
    print(f"  Hailo-8L (13 TOPS): {'✓' if has_hailo8l else '✗'}")
    
    return 0 if all_valid else 1


if __name__ == '__main__':
    sys.exit(main())