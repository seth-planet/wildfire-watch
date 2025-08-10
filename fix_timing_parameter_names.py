#!/usr/bin/env python3
"""Fix ENGINE_START_DURATION to IGNITION_START_DURATION in test files."""

import os
import re
import glob

# Find all test files
test_files = glob.glob("tests/**/*.py", recursive=True)

def fix_timing_parameter(filepath):
    """Replace ENGINE_START_DURATION with IGNITION_START_DURATION."""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return False
        
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Replace the incorrect parameter name
    original_content = content
    content = re.sub(r'ENGINE_START_DURATION', 'IGNITION_START_DURATION', content)
    
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Fixed timing parameter in {filepath}")
        return True
    else:
        return False

# Fix all test files
fixed_count = 0
for test_file in test_files:
    if fix_timing_parameter(test_file):
        fixed_count += 1

print(f"\nFixed {fixed_count} files")