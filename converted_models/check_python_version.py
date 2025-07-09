#!/usr/bin/env python3
"""Python version checker for hardware-specific model conversion."""

import sys

def check_coral_version():
    """Check if Python version is compatible with Coral TPU."""
    if sys.version_info[:2] != (3, 8):
        print(f"ERROR: Coral TPU requires Python 3.8")
        print(f"Current version: {sys.version_info.major}.{sys.version_info.minor}")
        print("Please install and use Python 3.8:")
        print("  sudo apt install python3.8 python3.8-dev python3.8-venv")
        print("  python3.8 -m pip install tflite-runtime pycoral")
        return False
    return True

def check_hailo_version():
    """Check if Python version is compatible with Hailo."""
    if sys.version_info[:2] != (3, 10):
        print(f"ERROR: Hailo SDK requires Python 3.10")
        print(f"Current version: {sys.version_info.major}.{sys.version_info.minor}")
        print("Please install and use Python 3.10:")
        print("  sudo apt install python3.10 python3.10-dev python3.10-venv")
        print("  python3.10 -m pip install hailo-python")
        return False
    return True

def check_yolo_nas_version():
    """Check if Python version is compatible with YOLO-NAS training."""
    if sys.version_info[:2] != (3, 10):
        print(f"ERROR: YOLO-NAS/super-gradients requires Python 3.10")
        print(f"Current version: {sys.version_info.major}.{sys.version_info.minor}")
        print("Please install and use Python 3.10:")
        print("  python3.10 -m pip install super-gradients")
        return False
    return True

def check_tensorrt_version():
    """Check if Python version is compatible with TensorRT."""
    if sys.version_info[:2] not in [(3, 10), (3, 12)]:
        print(f"WARNING: TensorRT works best with Python 3.10 or 3.12")
        print(f"Current version: {sys.version_info.major}.{sys.version_info.minor}")
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check Python version compatibility")
    parser.add_argument("--hardware", choices=["coral", "hailo", "yolo-nas", "tensorrt"],
                       required=True, help="Hardware/framework to check")
    args = parser.parse_args()
    
    if args.hardware == "coral":
        sys.exit(0 if check_coral_version() else 1)
    elif args.hardware == "hailo":
        sys.exit(0 if check_hailo_version() else 1)
    elif args.hardware == "yolo-nas":
        sys.exit(0 if check_yolo_nas_version() else 1)
    elif args.hardware == "tensorrt":
        sys.exit(0 if check_tensorrt_version() else 1)