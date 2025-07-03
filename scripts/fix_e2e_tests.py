#!/usr/bin/env python3.12
"""
Fix E2E tests to use refactored API patterns.
Updates tests to match the new service interfaces and configurations.
"""
import re
import sys
from pathlib import Path

def fix_e2e_tests(file_path):
    """Fix E2E test patterns to match refactored APIs"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Fix patterns
    replacements = [
        # Fix config object access to dict access
        (r'config\.cameras\b', "config['cameras']"),
        (r'assert len\(config\.cameras\)', "assert len(config['cameras'])"),
        
        # Fix camera object attributes to dict access
        (r'camera\.mac\b', "camera['mac']"),
        (r'camera\.id\b', "camera['mac']"),
        (r'camera\.ip\b', "camera['ip']"),
        (r'camera\.manufacturer\b', "camera['manufacturer']"),
        (r'camera\.model\b', "camera['model']"),
        (r'camera\.name\b', "camera['name']"),
        (r'camera\.rtsp_urls\b', "camera['rtsp_urls']"),
        (r'camera\.online\b', "camera['online']"),
        
        # Fix service attribute access to private names
        (r'service\.mqtt_connected\b', 'service._mqtt_connected'),
        (r'service\.mqtt_client\b', 'service._mqtt_client'),
        (r'detector\.cameras\b', 'detector.cameras'),  # This remains public
        (r'consensus\.detections\b', 'consensus.detections'),  # This remains public
        
        # Fix environment variable names
        (r'MQTT_TOPIC_PREFIX', 'TOPIC_PREFIX'),
        
        # Fix detection payload format
        (r"'camera_id': camera\.mac", "'camera_id': camera['mac']"),
        (r"'camera_ip': camera\.ip", "'camera_ip': camera['ip']"),
        (r"'camera_name': camera\.name", "'camera_name': camera['name']"),
        
        # Fix Frigate config access
        (r"config\['cameras'\]\[camera\.mac\.replace\(':', ''\)\]", "config['cameras'][camera['mac'].replace(':', '')]"),
    ]
    
    # Apply replacements
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)
    
    # Fix multi-line patterns
    # Fix camera object creation to dict
    content = re.sub(
        r'Camera\(\s*mac=([^,]+),\s*ip=([^,]+),\s*manufacturer=([^,]+),\s*model=([^,]+),\s*name=([^,]+),\s*rtsp_urls=([^,]+),\s*online=([^)]+)\)',
        r"{'mac': \1, 'ip': \2, 'manufacturer': \3, 'model': \4, 'name': \5, 'rtsp_urls': \6, 'online': \7}",
        content
    )
    
    # Fix detection creation
    content = re.sub(
        r"Detection\(\s*camera_id=([^,]+),\s*confidence=([^,]+),\s*timestamp=([^,]+),\s*object_type=([^)]+)\)",
        r"{'camera_id': \1, 'confidence': \2, 'timestamp': \3, 'object_type': \4}",
        content
    )
    
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    """Main function to fix E2E tests"""
    test_dir = Path('/home/seth/wildfire-watch/tests')
    
    # Find all E2E test files
    e2e_test_files = [
        'test_integration_e2e_improved.py',
        'test_integration_docker.py',
        'test_integration_docker_sdk.py',
        'test_e2e_coral_frigate.py',
        'test_e2e_hardware_docker.py',
        'test_e2e_hardware_integration.py',
        'test_frigate_hailo_docker_e2e.py',
        'test_hailo_e2e_fire_detection.py',
    ]
    
    fixed_count = 0
    
    for test_file in e2e_test_files:
        file_path = test_dir / test_file
        if file_path.exists():
            print(f"Processing {test_file}...")
            if fix_e2e_tests(file_path):
                print(f"  ✓ Fixed {test_file}")
                fixed_count += 1
            else:
                print(f"  - No changes needed for {test_file}")
        else:
            print(f"  ⚠ {test_file} not found")
    
    print(f"\nFixed {fixed_count} E2E test files")
    return 0

if __name__ == '__main__':
    sys.exit(main())