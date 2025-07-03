#!/usr/bin/env python3
"""Fix private attribute access in tests for refactored services.

This script updates tests to use the correct private attribute names
that are used in the refactored base classes.
"""

import re
import os
from pathlib import Path
from typing import List, Tuple

# Define replacement patterns for private attributes
ATTRIBUTE_REPLACEMENTS = [
    # MQTT connection state
    (r'\.mqtt_connected\b', '._mqtt_connected'),
    (r'hasattr\((\w+),\s*[\'"]mqtt_connected[\'"]\)', r'hasattr(\1, "_mqtt_connected")'),
    
    # MQTT client
    (r'\.mqtt_client\b(?!\.)', '._mqtt_client'),
    (r'hasattr\((\w+),\s*[\'"]mqtt_client[\'"]\)', r'hasattr(\1, "_mqtt_client")'),
    
    # Shutdown flag
    (r'\.shutdown\b(?!\()', '._shutdown'),
    (r'hasattr\((\w+),\s*[\'"]shutdown[\'"]\)', r'hasattr(\1, "_shutdown")'),
    
    # Running flag
    (r'\.running\b(?!\.)', '._running'),
    (r'hasattr\((\w+),\s*[\'"]running[\'"]\)', r'hasattr(\1, "_running")'),
    
    # Background threads
    (r'\.background_threads\b', '._background_threads'),
    
    # Service state
    (r'\.is_running\b(?!\()', '._is_running'),
    
    # Thread pool
    (r'\.thread_pool\b', '._thread_pool'),
    (r'\.executor\b', '._executor'),
    
    # Lock objects
    (r'\.lock\b(?!\.)', '._lock'),
    (r'\.mqtt_lock\b', '._mqtt_lock'),
    
    # Offline queue
    (r'\.offline_queue\b', '._offline_queue'),
    (r'\.offline_queue_enabled\b', '._offline_queue_enabled'),
]

# Method name updates
METHOD_REPLACEMENTS = [
    # Health reporting
    (r'detector\._publish_health\(\)', 'detector.health_reporter.report_health()'),
    (r'service\._publish_health\(\)', 'service.health_reporter.report_health()'),
    (r'self\._publish_health\(\)', 'self.health_reporter.report_health()'),
    
    # Background tasks - these are now managed differently
    (r'_start_background_tasks\(\)', '_setup_background_tasks()'),
    
    # MQTT methods that might be private
    (r'\.connect_mqtt\(\)', '._connect_with_retry()'),
    (r'\.disconnect_mqtt\(\)', '.shutdown()'),
]

# Files to skip (already manually fixed or special cases)
SKIP_FILES = {
    'conftest.py',
    'test_model_converter.py',
    'fix_private_attributes.py',  # Don't modify this script
}

def apply_replacements(content: str, filename: str) -> Tuple[str, List[str]]:
    """Apply all replacement patterns to content.
    
    Returns:
        Tuple of (modified_content, list_of_changes)
    """
    changes = []
    modified = content
    
    # Apply attribute replacements
    for pattern, replacement in ATTRIBUTE_REPLACEMENTS:
        matches = list(re.finditer(pattern, modified))
        if matches:
            modified = re.sub(pattern, replacement, modified)
            changes.append(f"  - Updated {len(matches)} instances of '{pattern}' to '{replacement}'")
    
    # Apply method replacements
    for pattern, replacement in METHOD_REPLACEMENTS:
        matches = list(re.finditer(pattern, modified))
        if matches:
            modified = re.sub(pattern, replacement, modified)
            changes.append(f"  - Updated {len(matches)} instances of '{pattern}' to '{replacement}'")
    
    return modified, changes

def should_process_file(filepath: Path) -> bool:
    """Check if file should be processed."""
    # Skip non-test files
    if not filepath.name.startswith('test_'):
        return False
    
    # Skip files in skip list
    if filepath.name in SKIP_FILES:
        return False
    
    # Skip if file doesn't contain service/detector references
    try:
        content = filepath.read_text()
        # Look for indicators this is a service/detector test
        indicators = ['CameraDetector', 'FireConsensus', 'TelemetryService', 
                     'GPIOTrigger', 'MQTTService', 'mqtt_connected', 'mqtt_client']
        return any(indicator in content for indicator in indicators)
    except:
        return False

def fix_test_file(filepath: Path) -> bool:
    """Fix a single test file. Returns True if changes were made."""
    try:
        content = filepath.read_text()
        original = content
        
        # Apply replacements
        modified, changes = apply_replacements(content, str(filepath))
        
        # Additional specific fixes for common patterns
        # Fix is_connected() calls on mqtt client
        modified = re.sub(
            r'(\w+)\.mqtt_client\.is_connected\(\)',
            r'\1._mqtt_connected',
            modified
        )
        
        # Fix _mqtt_client.is_connected() pattern
        modified = re.sub(
            r'(\w+)\._mqtt_client\.is_connected\(\)',
            r'\1._mqtt_connected',
            modified
        )
        
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
    """Run private attribute fixes on all test files."""
    tests_dir = Path(__file__).parent.parent / "tests"
    
    if not tests_dir.exists():
        print(f"Tests directory not found: {tests_dir}")
        return
    
    print("Fixing private attribute access in test files...")
    print(f"Looking in: {tests_dir}")
    
    # Find all test files
    test_files = list(tests_dir.glob("test_*.py"))
    print(f"\nFound {len(test_files)} test files")
    
    # Filter to only files that need processing
    files_to_process = [f for f in test_files if should_process_file(f)]
    print(f"Processing {len(files_to_process)} relevant test files")
    
    # Process each file
    fixed_count = 0
    for test_file in sorted(files_to_process):
        if fix_test_file(test_file):
            fixed_count += 1
    
    print(f"\n{'='*60}")
    print(f"Summary: Fixed {fixed_count} out of {len(files_to_process)} relevant test files")
    print(f"{'='*60}")
    
    # List common patterns that might need manual attention
    print("\n⚠️  Manual review needed for:")
    print("  - Tests that mock internal attributes")
    print("  - Tests checking connection state in complex ways")
    print("  - Integration tests with custom MQTT handling")
    print("  - Any remaining timeout issues")

if __name__ == "__main__":
    main()