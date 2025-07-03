#!/usr/bin/env python3.12
"""
Fix CameraState constructor calls in test files.
"""
import re
import os

# Files to fix
test_files = [
    'tests/test_consensus_debug.py',
    'tests/test_consensus_enhanced.py',
    'tests/test_core_logic.py'
]

def fix_camera_state_calls(filepath):
    """Fix CameraState constructor calls to include config parameter."""
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Pattern to match CameraState constructor without config
    # Matches: CameraState('id') or CameraState("id")
    pattern = r'CameraState\(([\'""][^\'""]+[\'""])\)'
    
    # Check if file imports Config
    has_config_import = 'from consensus import' in content and 'Config' in content
    
    if not has_config_import:
        # Add Config import if missing
        if 'from consensus import' in content:
            # Add Config to existing import
            content = re.sub(
                r'from consensus import ([^\\n]+)',
                r'from consensus import \1, Config',
                content
            )
        else:
            # Add new import after other imports
            import_pos = content.find('\nimport')
            if import_pos == -1:
                import_pos = content.find('\nfrom')
            if import_pos != -1:
                # Find end of imports
                lines = content[:import_pos].split('\n')
                last_import_line = len(lines)
                content_lines = content.split('\n')
                content_lines.insert(last_import_line, 'from consensus import Config')
                content = '\n'.join(content_lines)
    
    # Add mock config creation before first test or class
    if 'mock_config = Mock' not in content:
        # Find first test function or class
        test_pos = content.find('\ndef test_')
        class_pos = content.find('\nclass Test')
        
        insert_pos = min(pos for pos in [test_pos, class_pos] if pos > 0)
        
        if insert_pos > 0:
            mock_config_code = '''
# Create mock config for CameraState
mock_config = Mock(spec=Config)
mock_config.CONSENSUS_THRESHOLD = 2
mock_config.TIME_WINDOW = 30.0
mock_config.MIN_CONFIDENCE = 0.7
mock_config.MIN_AREA_RATIO = 0.0001
mock_config.MAX_AREA_RATIO = 0.5
mock_config.COOLDOWN_PERIOD = 60.0
mock_config.SINGLE_CAMERA_TRIGGER = False
mock_config.DETECTION_WINDOW = 30.0
mock_config.MOVING_AVERAGE_WINDOW = 3
mock_config.AREA_INCREASE_RATIO = 1.2
'''
            content = content[:insert_pos] + mock_config_code + content[insert_pos:]
    
    # Replace CameraState calls
    def replacement(match):
        camera_id = match.group(1)
        return f'CameraState({camera_id}, mock_config)'
    
    content = re.sub(pattern, replacement, content)
    
    # Write back
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"Fixed {filepath}")

# Fix all files
for filepath in test_files:
    if os.path.exists(filepath):
        fix_camera_state_calls(filepath)

print("\nDone! CameraState constructor calls have been fixed.")