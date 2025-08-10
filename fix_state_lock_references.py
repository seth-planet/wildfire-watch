#!/usr/bin/env python3
"""Fix _state_lock references in GPIO tests - should be _lock."""

import os
import re

# Find all test files that might have this issue
test_files = [
    "tests/test_gpio_edge_cases.py",
    "tests/test_gpio_state_machine_integrity.py",
]

def fix_state_lock_references(filepath):
    """Replace _state_lock with _lock in the file."""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return False
        
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Replace controller._state_lock with controller._lock
    original_content = content
    content = re.sub(r'controller\._state_lock', 'controller._lock', content)
    content = re.sub(r'self\._state_lock', 'self._lock', content)
    content = re.sub(r'with\s+controller\._state_lock:', 'with controller._lock:', content)
    
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Fixed _state_lock references in {filepath}")
        return True
    else:
        print(f"No _state_lock references found in {filepath}")
        return False

# Fix all test files
fixed_count = 0
for test_file in test_files:
    if fix_state_lock_references(test_file):
        fixed_count += 1

print(f"\nFixed {fixed_count} files")