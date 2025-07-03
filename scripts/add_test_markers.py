#!/usr/bin/env python3.12
"""Add pytest markers to test files for proper Python version selection."""

import os
import re
from pathlib import Path

def add_marker_to_file(filepath, marker):
    """Add pytest marker to a test file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check if marker already exists
    if f"pytest.mark.{marker}" in content or f"pytestmark = pytest.mark.{marker}" in content:
        print(f"  ✓ {filepath.name} already has {marker} marker")
        return False
    
    # Find import section
    lines = content.split('\n')
    import_end = 0
    has_pytest_import = False
    
    for i, line in enumerate(lines):
        if line.strip().startswith('import ') or line.strip().startswith('from '):
            import_end = i + 1
            if 'pytest' in line:
                has_pytest_import = True
        elif line.strip() and not line.strip().startswith('#') and not line.strip().startswith('"""'):
            # Found first non-import, non-comment line
            break
    
    # Add pytest import if needed
    if not has_pytest_import:
        lines.insert(import_end, 'import pytest')
        import_end += 1
    
    # Add marker after imports
    lines.insert(import_end, '')
    lines.insert(import_end + 1, f'pytestmark = pytest.mark.{marker}')
    lines.insert(import_end + 2, '')
    
    # Write back
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"  ✓ Added {marker} marker to {filepath.name}")
    return True

def main():
    """Add markers to test files."""
    root = Path(__file__).parent.parent
    tests_dir = root / 'tests'
    
    # Define which files need which markers
    file_markers = {
        'test_yolo_nas_training.py': 'yolo_nas',
        'test_yolo_nas_training_updated.py': 'yolo_nas',
        'test_api_usage.py': 'api_usage',
        'test_api_usage_fixed.py': 'api_usage',
        'test_qat_functionality.py': 'qat',
        # Coral TPU tests for Python 3.8
        'test_coral_camera_integration.py': 'coral_tpu',
        'test_coral_fire_video_e2e.py': 'coral_tpu',
        'test_coral_frigate_integration.py': 'coral_tpu',
        'test_e2e_coral_frigate.py': 'coral_tpu',
        'test_model_converter.py': 'model_conversion',
        'test_model_converter_e2e.py': 'model_conversion',
        'test_model_converter_hardware.py': 'model_conversion',
        'test_deployment.py': 'deployment',
    }
    
    print("Adding pytest markers to test files...")
    
    # Add Python 3.10 markers
    print("\nPython 3.10 test markers:")
    for filename, marker in file_markers.items():
        if marker in ['yolo_nas', 'api_usage', 'qat']:
            filepath = tests_dir / filename
            if filepath.exists():
                add_marker_to_file(filepath, marker)
            else:
                print(f"  ⚠️  {filename} not found")
    
    # Add Python 3.8 markers
    print("\nPython 3.8 test markers:")
    for filename, marker in file_markers.items():
        if marker in ['coral_tpu', 'model_conversion', 'deployment']:
            filepath = tests_dir / filename
            if filepath.exists():
                add_marker_to_file(filepath, marker)
            else:
                print(f"  ⚠️  {filename} not found")
    
    print("\nDone! Test files have been marked for proper Python version selection.")

if __name__ == "__main__":
    main()