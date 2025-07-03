#!/usr/bin/env python3
"""
Check if Python 3.8 is available for Coral TPU usage
"""

import sys
import subprocess
import shutil

def check_python38():
    """Check if Python 3.8 is available"""
    print("Checking for Python 3.8 (required for Coral TPU)...")
    
    # Check current Python version
    current_version = sys.version_info
    print(f"Current Python version: {current_version.major}.{current_version.minor}.{current_version.micro}")
    
    # Check if python3.8 executable exists
    python38_path = shutil.which("python3.8")
    if python38_path:
        print(f"✓ Python 3.8 found at: {python38_path}")
        
        # Check if tflite_runtime is installed for Python 3.8
        try:
            result = subprocess.run(
                [python38_path, "-c", "import tflite_runtime.interpreter as tflite; print('✓ tflite_runtime is installed')"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print(result.stdout.strip())
            else:
                print("✗ tflite_runtime is NOT installed for Python 3.8")
                print("  Install with: python3.8 -m pip install tflite-runtime")
        except Exception as e:
            print(f"Error checking tflite_runtime: {e}")
    else:
        print("✗ Python 3.8 NOT found")
        print("  Coral TPU requires Python 3.8 for tflite_runtime compatibility")
        print("  Install with: sudo apt install python3.8 python3.8-pip python3.8-dev")
    
    return python38_path is not None

if __name__ == "__main__":
    if check_python38():
        sys.exit(0)
    else:
        sys.exit(1)