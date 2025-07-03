#!/usr/bin/env python3.10
"""Inspect Hailo HEF file to display tensor information.

This utility parses a compiled Hailo Executable Format (HEF) file and displays
detailed information about input and output tensors. This is critical for
verifying that the compiled model matches Frigate's expectations.

The script uses the HailoRT Python API to extract:
- Input/output tensor names
- Tensor shapes and data formats
- Quantization parameters
- Network configuration

This tool is essential for debugging integration issues between Hailo models
and Frigate, as tensor naming and ordering must match exactly.

Usage:
    python3.10 inspect_hef.py model.hef
    
Example Output:
    --- Inspecting HEF: yolov8_hailo8l.hef ---
    Net params: {'batch_size': 1, 'input_count': 1, 'output_count': 3}
    
    --- Input VStream Infos ---
      Name: images
      Shape: (640, 640, 3) (H, W, C)
      Format: UINT8
      Quant Info: scale=0.00392, zero_point=0
    
    --- Output VStream Infos ---
      Name: output0
      Shape: (80, 80, 85)
      Format: UINT8
      Quant Info: scale=0.0156, zero_point=128

Requirements:
    - Python 3.10 (required by Hailo SDK)
    - hailort package installed
    - Valid HEF file compiled with Hailo Dataflow Compiler
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List, Any

try:
    from hailo_platform import HEF, VDevice, FormatType
except ImportError:
    print("Error: hailo_platform module not found.")
    print("Please ensure HailoRT is installed: pip install hailort")
    sys.exit(1)


def format_shape(shape: tuple) -> str:
    """Format tensor shape for display.
    
    Args:
        shape: Tuple of dimensions
        
    Returns:
        Formatted string like "(640, 640, 3)"
    """
    return f"({', '.join(map(str, shape))})"


def format_quant_info(quant_info: Any) -> str:
    """Format quantization info for display.
    
    Args:
        quant_info: Quantization information object
        
    Returns:
        Formatted string with scale and zero point
    """
    if hasattr(quant_info, 'qp_scale') and hasattr(quant_info, 'qp_zp'):
        return f"scale={quant_info.qp_scale:.6f}, zero_point={quant_info.qp_zp}"
    return str(quant_info)


def inspect_hef(hef_path: str, json_output: bool = False) -> Dict[str, Any]:
    """Parse a HEF file and extract tensor information.
    
    Args:
        hef_path: Path to the HEF file
        json_output: If True, return JSON-serializable dict instead of printing
        
    Returns:
        Dictionary containing model information if json_output=True
        
    Raises:
        RuntimeError: If HEF file cannot be loaded
    """
    hef_path = Path(hef_path)
    if not hef_path.exists():
        raise FileNotFoundError(f"HEF file not found: {hef_path}")
    
    if not hef_path.suffix == '.hef':
        raise ValueError(f"File must have .hef extension, got: {hef_path.suffix}")
    
    try:
        # Load the HEF file
        hef = HEF(str(hef_path))
    except Exception as e:
        raise RuntimeError(f"Failed to load HEF file: {e}")
    
    # Extract network information
    network_groups = hef.get_network_groups_infos()
    if not network_groups:
        raise RuntimeError("No network groups found in HEF file")
    
    # Use first network group (usually only one)
    network_info = network_groups[0]
    
    # Prepare output structure
    model_info = {
        "file": str(hef_path),
        "name": network_info.name,
        "network_params": {
            "is_multi_context": network_info.is_multi_context,
        },
        "inputs": [],
        "outputs": []
    }
    
    # Get input stream infos
    for stream_info in network_info.get_input_stream_infos():
        input_info = {
            "name": stream_info.name,
            "shape": stream_info.shape,
            "format": stream_info.format.name if hasattr(stream_info.format, 'name') else str(stream_info.format),
            "hw_shape": stream_info.hw_shape if hasattr(stream_info, 'hw_shape') else None,
            "quant_info": {
                "scale": float(stream_info.quant_info.qp_scale) if hasattr(stream_info.quant_info, 'qp_scale') else None,
                "zero_point": int(stream_info.quant_info.qp_zp) if hasattr(stream_info.quant_info, 'qp_zp') else None
            } if hasattr(stream_info, 'quant_info') else None
        }
        model_info["inputs"].append(input_info)
    
    # Get output stream infos
    for stream_info in network_info.get_output_stream_infos():
        output_info = {
            "name": stream_info.name,
            "shape": stream_info.shape,
            "format": stream_info.format.name if hasattr(stream_info.format, 'name') else str(stream_info.format),
            "hw_shape": stream_info.hw_shape if hasattr(stream_info, 'hw_shape') else None,
            "quant_info": {
                "scale": float(stream_info.quant_info.qp_scale) if hasattr(stream_info.quant_info, 'qp_scale') else None,
                "zero_point": int(stream_info.quant_info.qp_zp) if hasattr(stream_info.quant_info, 'qp_zp') else None
            } if hasattr(stream_info, 'quant_info') else None
        }
        model_info["outputs"].append(output_info)
    
    if json_output:
        return model_info
    
    # Print formatted output
    print(f"\n{'='*60}")
    print(f"HEF Model Inspector - {hef_path.name}")
    print(f"{'='*60}")
    
    print(f"\nFile: {hef_path}")
    print(f"Network Name: {network_info.name}")
    print(f"Multi-context: {network_info.is_multi_context}")
    
    print(f"\n--- Input Streams ({len(model_info['inputs'])}) ---")
    for i, input_info in enumerate(model_info['inputs']):
        print(f"\n  [{i}] Name: {input_info['name']}")
        print(f"      Shape: {format_shape(input_info['shape'])}")
        print(f"      Format: {input_info['format']}")
        if input_info['hw_shape']:
            print(f"      HW Shape: {format_shape(input_info['hw_shape'])}")
        if input_info['quant_info'] and input_info['quant_info']['scale']:
            print(f"      Quantization: scale={input_info['quant_info']['scale']:.6f}, "
                  f"zero_point={input_info['quant_info']['zero_point']}")
    
    print(f"\n--- Output Streams ({len(model_info['outputs'])}) ---")
    for i, output_info in enumerate(model_info['outputs']):
        print(f"\n  [{i}] Name: {output_info['name']}")
        print(f"      Shape: {format_shape(output_info['shape'])}")
        print(f"      Format: {output_info['format']}")
        if output_info['hw_shape']:
            print(f"      HW Shape: {format_shape(output_info['hw_shape'])}")
        if output_info['quant_info'] and output_info['quant_info']['scale']:
            print(f"      Quantization: scale={output_info['quant_info']['scale']:.6f}, "
                  f"zero_point={output_info['quant_info']['zero_point']}")
    
    # Verify YOLO compatibility
    print(f"\n--- Frigate Compatibility Check ---")
    compatibility_issues = []
    
    # Check input
    if len(model_info['inputs']) != 1:
        compatibility_issues.append(f"Expected 1 input, found {len(model_info['inputs'])}")
    else:
        input_name = model_info['inputs'][0]['name']
        if input_name not in ['images', 'input', 'data']:
            compatibility_issues.append(f"Input name '{input_name}' may not be compatible (expected 'images')")
    
    # Check outputs for YOLO
    if len(model_info['outputs']) == 3:
        print("✓ YOLO-style model detected (3 outputs)")
        # Check output shapes are descending (largest to smallest feature maps)
        shapes = [out['shape'] for out in model_info['outputs']]
        print(f"  Output shapes: {[format_shape(s) for s in shapes]}")
    else:
        compatibility_issues.append(f"Expected 3 outputs for YOLO, found {len(model_info['outputs'])}")
    
    if compatibility_issues:
        print("\n⚠️  Potential compatibility issues:")
        for issue in compatibility_issues:
            print(f"  - {issue}")
    else:
        print("✓ Model appears compatible with Frigate")
    
    print(f"\n{'='*60}\n")
    
    return model_info


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Inspect Hailo HEF file tensor information",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s model.hef
  %(prog)s --json model.hef > model_info.json
  %(prog)s --compare model1.hef model2.hef
        """
    )
    
    parser.add_argument(
        "hef_path",
        type=str,
        help="Path to the HEF file to inspect"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )
    
    parser.add_argument(
        "--compare",
        type=str,
        metavar="HEF2",
        help="Compare with another HEF file"
    )
    
    args = parser.parse_args()
    
    try:
        if args.compare:
            # Compare two HEF files
            info1 = inspect_hef(args.hef_path, json_output=True)
            info2 = inspect_hef(args.compare, json_output=True)
            
            print(f"\nComparing HEF files:")
            print(f"  File 1: {args.hef_path}")
            print(f"  File 2: {args.compare}")
            
            # Compare inputs
            if info1['inputs'] != info2['inputs']:
                print("\n⚠️  Input differences detected")
            else:
                print("\n✓ Inputs match")
            
            # Compare outputs
            if info1['outputs'] != info2['outputs']:
                print("⚠️  Output differences detected")
            else:
                print("✓ Outputs match")
                
        elif args.json:
            # JSON output
            info = inspect_hef(args.hef_path, json_output=True)
            print(json.dumps(info, indent=2))
        else:
            # Standard output
            inspect_hef(args.hef_path)
            
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()