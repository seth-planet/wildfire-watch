#!/usr/bin/env python3
"""
Add test tier markers (smoke, integration, hardware, e2e, unit) to test files.
This complements the Python version markers by adding test organization markers.
"""

import re
import os
from pathlib import Path

def determine_tier_markers(filepath):
    """Determine appropriate tier markers based on file name and content."""
    filename = os.path.basename(filepath)
    markers = []
    
    # Read file content for better classification
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check if already has tier markers (not Python version markers)
    tier_markers = ['smoke', 'integration', 'hardware', 'e2e', 'unit']
    existing_markers = []
    for marker in tier_markers:
        if f'pytest.mark.{marker}' in content:
            existing_markers.append(marker)
    
    if existing_markers:
        print(f"  Already has tier markers: {', '.join(existing_markers)}")
        return None
    
    # E2E tests
    if 'e2e' in filename.lower() or 'end_to_end' in filename.lower():
        markers.append('e2e')
        markers.append('integration')
        markers.append('slow')
    
    # Integration tests
    elif 'integration' in filename.lower():
        markers.append('integration')
    
    # Hardware tests
    elif any(hw in filename.lower() for hw in ['gpio', 'coral', 'hailo', 'tensorrt', 'hardware']):
        markers.append('hardware')
        markers.append('integration')
    
    # Docker tests
    elif 'docker' in filename.lower():
        markers.append('integration')
        markers.append('docker')
    
    # Camera tests
    elif 'camera' in filename.lower():
        markers.append('integration')
        markers.append('cameras')
    
    # MQTT tests
    elif 'mqtt' in filename.lower():
        markers.append('integration')
        markers.append('mqtt')
    
    # Model/AI tests
    elif any(m in filename.lower() for m in ['model', 'yolo', 'accuracy', 'training']):
        markers.append('integration')
    
    # API tests
    elif 'api' in filename.lower():
        markers.append('integration')
    
    # Web interface tests
    elif 'web' in filename.lower():
        markers.append('integration')
    
    # Consensus/Detection tests - check content
    elif 'consensus' in filename.lower() or 'detect' in filename.lower():
        if 'test_mqtt_broker' in content or 'MQTTTestBroker' in content or 'real MQTT' in content:
            markers.append('integration')
            markers.append('mqtt')
        else:
            markers.append('unit')
    
    # Trigger tests
    elif 'trigger' in filename.lower():
        markers.append('integration')
    
    # Core logic / Config tests - check content
    elif any(c in filename.lower() for c in ['core', 'config', 'util', 'helper', 'refill']):
        if 'test_mqtt_broker' in content or 'MQTTTestBroker' in content:
            markers.append('integration')
        else:
            markers.append('unit')
    
    # Default classification based on content
    else:
        # Check content for integration indicators
        integration_indicators = [
            'test_mqtt_broker', 'MQTTTestBroker', 'DockerContainerManager',
            'real MQTT', 'mqtt_client', 'docker', 'GPIO'
        ]
        if any(ind in content for ind in integration_indicators):
            markers.append('integration')
        else:
            markers.append('unit')
    
    # Add smoke marker to key test files
    smoke_files = [
        'test_consensus.py', 'test_detect.py', 'test_trigger.py', 
        'test_camera_detector.py', 'test_configuration_system.py'
    ]
    if filename in smoke_files and 'smoke' not in markers:
        markers.append('smoke')
    
    # Add slow marker if needed
    if any(slow in content for slow in ['timeout=300', 'timeout=600', 'timeout=1800']):
        if 'slow' not in markers:
            markers.append('slow')
    
    return markers

def add_tier_markers_to_file(filepath, markers):
    """Add test tier markers to a file that may already have other markers."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Check if file already has pytestmark
    has_pytestmark = False
    pytestmark_line = -1
    pytestmark_end = -1
    
    for i, line in enumerate(lines):
        if 'pytestmark' in line:
            has_pytestmark = True
            pytestmark_line = i
            # Find the end of pytestmark list
            for j in range(i, len(lines)):
                if ']' in lines[j]:
                    pytestmark_end = j
                    break
            break
    
    if has_pytestmark:
        # Add to existing pytestmark
        print(f"  Adding tier markers to existing pytestmark in {os.path.basename(filepath)}")
        
        # Parse existing pytestmark to avoid duplicates
        existing_content = ''.join(lines[pytestmark_line:pytestmark_end+1])
        
        # Build new marker lines
        new_marker_lines = []
        for marker in markers:
            marker_str = f'    pytest.mark.{marker}'
            if marker_str not in existing_content:
                new_marker_lines.append(f'{marker_str},\n')
        
        if new_marker_lines:
            # Insert before the closing bracket
            lines[pytestmark_end:pytestmark_end] = new_marker_lines
            
            with open(filepath, 'w') as f:
                f.writelines(lines)
            print(f"  Added markers: {', '.join(markers)}")
        else:
            print(f"  All markers already present")
    else:
        # Add new pytestmark section
        # Find where to insert (after imports)
        import_end = 0
        has_pytest_import = False
        
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                import_end = i + 1
                if 'pytest' in line:
                    has_pytest_import = True
            elif line.strip() and not line.strip().startswith('#') and not line.strip().startswith('"""'):
                break
        
        # Build the marker block
        marker_block = []
        if not has_pytest_import:
            marker_block.append('import pytest\n')
        
        marker_block.append('\n')
        marker_block.append('# Test tier markers for organization\n')
        marker_block.append('pytestmark = [\n')
        for marker in markers:
            marker_block.append(f'    pytest.mark.{marker},\n')
        marker_block.append(']\n')
        marker_block.append('\n')
        
        # Insert the marker block
        lines[import_end:import_end] = marker_block
        
        with open(filepath, 'w') as f:
            f.writelines(lines)
        
        print(f"  Added new pytestmark with markers: {', '.join(markers)}")

def main():
    """Process all test files to add tier markers."""
    test_dir = Path('/home/seth/wildfire-watch/tests')
    test_files = sorted(test_dir.glob('test_*.py'))
    
    print(f"Processing {len(test_files)} test files for tier markers...")
    print("="*60)
    
    stats = {
        'added': 0,
        'skipped': 0,
        'already_marked': 0
    }
    
    for filepath in test_files:
        print(f"\n{filepath.name}:")
        markers = determine_tier_markers(filepath)
        
        if markers is None:
            stats['already_marked'] += 1
        elif markers:
            add_tier_markers_to_file(filepath, markers)
            stats['added'] += 1
        else:
            print(f"  No markers determined")
            stats['skipped'] += 1
    
    print("\n" + "="*60)
    print("Summary:")
    print(f"  Files with markers added: {stats['added']}")
    print(f"  Files already marked: {stats['already_marked']}")
    print(f"  Files skipped: {stats['skipped']}")
    print("\nTest tiers added:")
    print("  - unit: Fast tests with no external dependencies")
    print("  - smoke: Quick basic functionality tests")
    print("  - integration: Tests requiring external services")
    print("  - hardware: Tests requiring specific hardware")
    print("  - e2e: End-to-end system tests")

if __name__ == '__main__':
    main()