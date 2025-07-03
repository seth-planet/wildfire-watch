#!/usr/bin/env python3
"""Apply mechanical fixes to tests for refactored code patterns.

This script automatically fixes common patterns in tests:
1. Config imports: Config → FireConsensusConfig/CameraDetectorConfig
2. Config access: config['key'] → config.key
3. Camera ID: camera.id → camera.mac
4. Method renames: _publish_health() → health reporter pattern
"""

import re
import os
from pathlib import Path
from typing import List, Tuple

# Define replacement patterns
REPLACEMENTS = [
    # Config imports
    (r'from fire_consensus\.consensus import Config\b', 
     'from fire_consensus.consensus import FireConsensusConfig'),
    (r'from camera_detector\.detect import Config\b', 
     'from camera_detector.detect import CameraDetectorConfig'),
    
    # Config instantiation
    (r'\bConfig\(\)', 'FireConsensusConfig()'),  # Default to FireConsensusConfig
    
    # Config dict access to attribute access
    (r'config\[[\'"]([\w_]+)[\'"]\]', r'config.\1'),
    (r'self\.config\[[\'"]([\w_]+)[\'"]\]', r'self.config.\1'),
    
    # Camera ID to MAC
    (r'camera\.id\b', 'camera.mac'),
    (r'cam\.id\b', 'cam.mac'),
    
    # Method renames
    (r'detector\._publish_health\(\)', 
     'detector.health_reporter.report_health()'),
    (r'self\._publish_health\(\)', 
     'self.health_reporter.report_health()'),
    
    # Direct health publishing
    (r'publish_message\([\'"]system/health/[\w_]+[\'"],', 
     'health_reporter.report_health('),
]

# Files to skip (already manually fixed or special cases)
SKIP_FILES = {
    'conftest.py',  # Has special adapter logic
    'test_model_converter.py',  # Different Config class
}

def apply_replacements(content: str, filename: str) -> Tuple[str, List[str]]:
    """Apply all replacement patterns to content.
    
    Returns:
        Tuple of (modified_content, list_of_changes)
    """
    changes = []
    modified = content
    
    for pattern, replacement in REPLACEMENTS:
        # Skip certain replacements for specific files
        if 'FireConsensusConfig' in replacement and 'camera_detector' in filename:
            # Use CameraDetectorConfig for camera detector tests
            replacement = replacement.replace('FireConsensusConfig', 'CameraDetectorConfig')
        
        # Find all matches before replacement
        matches = list(re.finditer(pattern, modified))
        if matches:
            # Apply replacement
            modified = re.sub(pattern, replacement, modified)
            changes.append(f"  - Replaced {len(matches)} instances of '{pattern}' with '{replacement}'")
    
    return modified, changes

def fix_test_file(filepath: Path) -> bool:
    """Fix a single test file. Returns True if changes were made."""
    try:
        content = filepath.read_text()
        original = content
        
        # Apply replacements
        modified, changes = apply_replacements(content, str(filepath))
        
        # Special case: Fix Config class context
        if 'test_consensus' in filepath.name:
            # Ensure FireConsensusConfig is used
            modified = re.sub(r'\bCameraDetectorConfig\b', 'FireConsensusConfig', modified)
        elif 'test_camera' in filepath.name or 'test_detect' in filepath.name:
            # Ensure CameraDetectorConfig is used  
            modified = re.sub(r'\bFireConsensusConfig\b', 'CameraDetectorConfig', modified)
        
        # Write back if changed
        if modified != original:
            filepath.write_text(modified)
            print(f"\n✓ Fixed {filepath}")
            for change in changes:
                print(change)
            return True
        return False
        
    except Exception as e:
        print(f"\n✗ Error processing {filepath}: {e}")
        return False

def main():
    """Run mechanical fixes on all test files."""
    tests_dir = Path(__file__).parent.parent / "tests"
    
    if not tests_dir.exists():
        print(f"Tests directory not found: {tests_dir}")
        return
    
    print("Applying mechanical fixes to test files...")
    print(f"Looking in: {tests_dir}")
    
    # Find all test files
    test_files = list(tests_dir.glob("test_*.py"))
    print(f"\nFound {len(test_files)} test files")
    
    # Process each file
    fixed_count = 0
    for test_file in sorted(test_files):
        if test_file.name in SKIP_FILES:
            print(f"\n⏭️  Skipping {test_file.name} (in skip list)")
            continue
            
        if fix_test_file(test_file):
            fixed_count += 1
    
    print(f"\n{'='*60}")
    print(f"Summary: Fixed {fixed_count} out of {len(test_files)} test files")
    print(f"{'='*60}")
    
    # Additional manual fixes needed
    print("\n⚠️  Manual fixes still needed for:")
    print("  - Tests using camera.id where adapter isn't available")
    print("  - Tests expecting old health publishing patterns")
    print("  - Integration tests with complex service interactions")
    print("  - E2E tests (currently skipped)")

if __name__ == "__main__":
    main()