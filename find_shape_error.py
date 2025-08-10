#!/usr/bin/env python3.10
"""Find where dict.shape error occurs in YOLO-NAS test"""

import sys
import os
import ast
import re

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def find_shape_accesses(directory):
    """Find all .shape accesses in Python files"""
    shape_pattern = re.compile(r'(\w+)\.shape')
    results = []
    
    for root, dirs, files in os.walk(directory):
        # Skip __pycache__ directories
        dirs[:] = [d for d in dirs if d != '__pycache__']
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        lines = f.readlines()
                    
                    for i, line in enumerate(lines, 1):
                        matches = shape_pattern.finditer(line)
                        for match in matches:
                            var_name = match.group(1)
                            # Skip if it's clearly checking with hasattr
                            if 'hasattr' in line and 'shape' in line:
                                continue
                            # Skip comments
                            if line.strip().startswith('#'):
                                continue
                            
                            results.append({
                                'file': filepath,
                                'line': i,
                                'var': var_name,
                                'code': line.strip()
                            })
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
    
    return results

# Search for shape accesses in converted_models
print("Searching for .shape accesses in converted_models/...")
shape_accesses = find_shape_accesses('converted_models')

# Filter for potentially problematic ones
print("\nPotentially problematic .shape accesses (not checking with hasattr):")
for access in shape_accesses:
    # Skip if the line already has protection
    if 'if ' in access['code'] or 'hasattr' in access['code']:
        continue
    print(f"\n{access['file']}:{access['line']}")
    print(f"  Variable: {access['var']}")
    print(f"  Code: {access['code']}")

# Now let's specifically check the files mentioned in the test
print("\n\nChecking specific files from the test...")

# Check if there's a specific model output handling
try:
    from converted_models.unified_yolo_trainer import UnifiedYOLOTrainer
    print("✓ UnifiedYOLOTrainer imports successfully")
except Exception as e:
    print(f"✗ UnifiedYOLOTrainer import error: {e}")

try:
    from converted_models.model_exporter import ModelExporter
    print("✓ ModelExporter imports successfully")
except Exception as e:
    print(f"✗ ModelExporter import error: {e}")

# Look for specific patterns that might cause dict output
print("\n\nSearching for model output handling...")
output_pattern = re.compile(r'(output|result|pred\w*|detect\w*)\s*=.*model')

for root, dirs, files in os.walk('converted_models'):
    dirs[:] = [d for d in dirs if d != '__pycache__']
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                
                for i, line in enumerate(lines, 1):
                    if output_pattern.search(line):
                        # Check next few lines for .shape access
                        for j in range(i, min(i+5, len(lines))):
                            if '.shape' in lines[j]:
                                print(f"\n{filepath}:{i}-{j+1}")
                                print(f"  Model output: {line.strip()}")
                                print(f"  Shape access: {lines[j].strip()}")
            except:
                pass