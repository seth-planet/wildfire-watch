#!/usr/bin/env python3.12
"""Rename existing models to follow the new naming convention."""

import os
import sys
import shutil
from pathlib import Path

# Add parent directory to path for utils
sys.path.append(str(Path(__file__).parent.parent))
from utils.model_naming import get_model_filename

def rename_existing_models():
    """Rename existing models to the new naming convention."""
    model_dir = Path("/home/seth/wildfire-watch/models")
    
    # Mapping of old names to new parameters
    # Assuming current models are 'large' size based on yolo8l
    rename_map = {
        "yolo8l_fire_tensorrt.engine": {
            "size": "large",
            "accelerator": "tensorrt", 
            "precision": "fp16"
        },
        "yolo8l_fire_tensorrt_int8.engine": {
            "size": "large",
            "accelerator": "tensorrt",
            "precision": "int8"
        },
    }
    
    renamed = []
    for old_name, params in rename_map.items():
        old_path = model_dir / old_name
        if old_path.exists():
            new_name = get_model_filename(
                params["size"], 
                params["accelerator"], 
                params["precision"]
            )
            new_path = model_dir / new_name
            
            if not new_path.exists():
                print(f"Renaming: {old_name} -> {new_name}")
                shutil.move(str(old_path), str(new_path))
                renamed.append((old_name, new_name))
            else:
                print(f"Target already exists: {new_name}")
        else:
            print(f"Source not found: {old_name}")
    
    # List all models after renaming
    print("\nModels in directory after renaming:")
    for model in sorted(model_dir.glob("wildfire_*")):
        size = os.path.getsize(model) / (1024 * 1024)  # MB
        print(f"  {model.name} ({size:.1f} MB)")
    
    # Also list any remaining non-standard models
    print("\nNon-standard models (may need manual renaming):")
    for model in model_dir.glob("*"):
        if not model.name.startswith("wildfire_") and model.is_file():
            size = os.path.getsize(model) / (1024 * 1024)  # MB
            print(f"  {model.name} ({size:.1f} MB)")
    
    return renamed

if __name__ == "__main__":
    renamed = rename_existing_models()
    if renamed:
        print(f"\nSuccessfully renamed {len(renamed)} models")
    else:
        print("\nNo models were renamed")