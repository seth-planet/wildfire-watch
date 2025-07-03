#!/usr/bin/env python3.12
"""
Generate TensorRT models for tests
Creates symlinks to existing models or generates new ones as needed
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def ensure_tensorrt_models():
    """Ensure TensorRT models exist for tests"""
    
    # Create models directory if it doesn't exist
    models_dir = Path("/models")
    if not models_dir.exists():
        try:
            models_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            # If we can't create /models, use a local directory
            models_dir = Path("models")
            models_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing TensorRT models
    existing_models = {
        "wildfire_640_tensorrt_int8.trt": [
            "converted_models/output_int8/640x640/yolo8l_fire_int8_tensorrt.engine",
            "converted_models/output/640x640/yolo8l_fire_tensorrt.engine",
        ],
        "yolov8n_tensorrt.engine": [
            "converted_models/output_test/640x640/yolo8l_fire_test_tensorrt.engine",
        ]
    }
    
    created_models = []
    missing_models = []
    
    for target_name, possible_sources in existing_models.items():
        target_path = models_dir / target_name
        
        if target_path.exists():
            print(f"✓ {target_name} already exists")
            created_models.append(target_name)
            continue
        
        # Look for existing model to link
        source_found = False
        for source in possible_sources:
            source_path = Path(source)
            if source_path.exists():
                try:
                    # Try to create symlink first
                    target_path.symlink_to(source_path.absolute())
                    print(f"✓ Created symlink: {target_name} -> {source}")
                    created_models.append(target_name)
                    source_found = True
                    break
                except (OSError, PermissionError):
                    # If symlink fails, try to copy
                    try:
                        shutil.copy2(source_path, target_path)
                        print(f"✓ Copied: {source} -> {target_name}")
                        created_models.append(target_name)
                        source_found = True
                        break
                    except Exception as e:
                        print(f"✗ Failed to copy {source}: {e}")
        
        if not source_found:
            missing_models.append(target_name)
    
    # Generate missing models if needed
    if missing_models:
        print(f"\n⚠ Missing models: {missing_models}")
        print("You can generate them using:")
        print("  cd converted_models")
        print("  python3.12 convert_model.py --model yolov8l --format tensorrt --size 640")
        
        # Try to generate if convert_model.py exists
        convert_script = Path("converted_models/convert_model.py")
        if convert_script.exists():
            print("\nAttempting to generate missing models...")
            for model in missing_models:
                if "int8" in model:
                    cmd = [
                        "python3.12", str(convert_script),
                        "--model", "yolov8l",
                        "--format", "tensorrt",
                        "--size", "640",
                        "--precision", "int8"
                    ]
                else:
                    cmd = [
                        "python3.12", str(convert_script),
                        "--model", "yolov8l",
                        "--format", "tensorrt",
                        "--size", "640"
                    ]
                
                print(f"Running: {' '.join(cmd)}")
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        print(f"✓ Generated {model}")
                    else:
                        print(f"✗ Failed to generate {model}: {result.stderr}")
                except Exception as e:
                    print(f"✗ Error running converter: {e}")
    
    return created_models, missing_models


def main():
    """Main function"""
    print("=== TensorRT Model Setup ===")
    
    created, missing = ensure_tensorrt_models()
    
    print(f"\nSummary:")
    print(f"  Created: {len(created)} models")
    print(f"  Missing: {len(missing)} models")
    
    if missing:
        print("\n⚠ Some models are missing. Tests requiring these models may be skipped.")
        sys.exit(1)
    else:
        print("\n✓ All TensorRT models are ready for testing!")
        sys.exit(0)


if __name__ == "__main__":
    main()