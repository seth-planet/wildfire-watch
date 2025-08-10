#!/usr/bin/env python3
"""
Patch Frigate to add YOLO EdgeTPU detector support.
This script modifies Frigate's detector system to support YOLO models.
"""

import os
import sys
import shutil

def patch_detector_types():
    """Add yolo_edgetpu to the list of supported detector types."""
    
    # Find Frigate installation
    frigate_paths = [
        "/opt/frigate/frigate",
        "/usr/local/lib/python3.9/dist-packages/frigate",
        "/app/frigate"
    ]
    
    frigate_path = None
    for path in frigate_paths:
        if os.path.exists(path):
            frigate_path = path
            break
    
    if not frigate_path:
        print("ERROR: Could not find Frigate installation")
        return False
    
    print(f"Found Frigate at: {frigate_path}")
    
    # Check if plugins directory exists (indicates dev version)
    plugins_dir = os.path.join(frigate_path, "detectors", "plugins")
    if os.path.exists(plugins_dir):
        print("✓ Frigate has plugin support - using plugin system")
        # Copy our plugin - try simple version first
        plugin_src = "/patch/yolo_edgetpu_simple.py"
        if not os.path.exists(plugin_src):
            plugin_src = "/patch/yolo_edgetpu_v2.py"
            if not os.path.exists(plugin_src):
                plugin_src = "/patch/yolo_edgetpu.py"
        plugin_dst = os.path.join(plugins_dir, "yolo_edgetpu.py")
        if os.path.exists(plugin_src):
            shutil.copy2(plugin_src, plugin_dst)
            print(f"✓ Copied YOLO plugin to {plugin_dst}")
            return True
        else:
            print(f"ERROR: Plugin source not found: {plugin_src}")
            return False
    
    # Fallback: Patch detector_types.py directly
    print("⚠ No plugin support - patching detector_types.py directly")
    
    # Find detector_types.py
    detector_types_path = None
    for root, dirs, files in os.walk(frigate_path):
        if "detector_types.py" in files:
            detector_types_path = os.path.join(root, "detector_types.py")
            break
    
    if not detector_types_path:
        print("ERROR: Could not find detector_types.py")
        return False
    
    print(f"Found detector_types.py at: {detector_types_path}")
    
    # Read the file
    with open(detector_types_path, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "yolo_edgetpu" in content:
        print("✓ Frigate already patched for YOLO support")
        return True
    
    # Add yolo_edgetpu to DETECTOR_TYPES
    if "DETECTOR_TYPES = [" in content:
        # Find the list and add our detector
        lines = content.split('\n')
        new_lines = []
        in_detector_types = False
        
        for line in lines:
            if "DETECTOR_TYPES = [" in line:
                in_detector_types = True
                new_lines.append(line)
            elif in_detector_types and "]" in line:
                # Add our detector before closing
                new_lines.append('    "yolo_edgetpu",')
                new_lines.append(line)
                in_detector_types = False
            else:
                new_lines.append(line)
        
        # Write back
        with open(detector_types_path, 'w') as f:
            f.write('\n'.join(new_lines))
        
        print("✓ Patched detector_types.py to include yolo_edgetpu")
    
    # Copy our detector implementation
    detectors_dir = os.path.join(os.path.dirname(detector_types_path), "plugins")
    if not os.path.exists(detectors_dir):
        os.makedirs(detectors_dir)
    
    # Try simple first, then v2, then v1
    detector_src = "/patch/yolo_edgetpu_simple.py"
    if not os.path.exists(detector_src):
        detector_src = "/patch/yolo_edgetpu_v2.py"
        if not os.path.exists(detector_src):
            detector_src = "/patch/yolo_edgetpu.py"
    detector_dst = os.path.join(detectors_dir, "yolo_edgetpu.py")
    
    if os.path.exists(detector_src):
        shutil.copy2(detector_src, detector_dst)
        print(f"✓ Copied YOLO detector to {detector_dst}")
        
        # Create __init__.py if needed
        init_path = os.path.join(detectors_dir, "__init__.py")
        if not os.path.exists(init_path):
            with open(init_path, 'w') as f:
                f.write("# Detector plugins\n")
        
        return True
    else:
        print(f"ERROR: Detector source not found: {detector_src}")
        return False

if __name__ == "__main__":
    print("Patching Frigate for YOLO EdgeTPU support...")
    print("=" * 50)
    
    if patch_detector_types():
        print("\n✓ Patching completed successfully!")
        print("Frigate now supports yolo_edgetpu detector type")
    else:
        print("\n✗ Patching failed!")
        sys.exit(1)