#!/usr/bin/env python3.12
"""Remove hardcoded credentials from the codebase."""

import os
import re
import sys
from pathlib import Path

# Patterns to search for
CREDENTIAL_PATTERNS = [
    r'',
    r'',
    r'',
    r'',
    r'',
]

# Replacement patterns
REPLACEMENTS = {
    # In code files
    r"''": "''",
    r'""': '""',
    r"''": "''",
    r'""': '""',
    r"''": "''", 
    r'""': '""',
    r"''": "''",
    r'""': '""',
    r"''": "''",
    r'""': '""',
    
    # In YAML/config files
    r'': 'username:password',
    r'': 'username:password',
    r'': 'username:password',
    r'': 'username:password',
    r'': 'username:password',
    
    # In lists
    r'admin:,username:password,username:password,username:password': '',
    r'admin:,username:password,username:password': '',
}

def find_files_with_credentials(root_dir):
    """Find all files containing hardcoded credentials."""
    files_with_creds = []
    
    for pattern in CREDENTIAL_PATTERNS:
        cmd = f"grep -r '{pattern}' '{root_dir}' --include='*.py' --include='*.yml' --include='*.yaml' --include='*.md' | grep -v '.pyc' | cut -d':' -f1 | sort | uniq"
        result = os.popen(cmd).read().strip()
        if result:
            files_with_creds.extend(result.split('\n'))
    
    # Remove duplicates
    return list(set(files_with_creds))

def fix_file(filepath):
    """Remove hardcoded credentials from a file."""
    print(f"Processing: {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Apply replacements
        for pattern, replacement in REPLACEMENTS.items():
            content = re.sub(pattern, replacement, content)
        
        if content != original_content:
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"  ✓ Fixed credentials in {filepath}")
            return True
        else:
            print(f"  - No changes needed in {filepath}")
            return False
            
    except Exception as e:
        print(f"  ✗ Error processing {filepath}: {e}")
        return False

def main():
    """Main function."""
    root_dir = Path(__file__).parent.parent
    
    print("Searching for files with hardcoded credentials...")
    files = find_files_with_credentials(root_dir)
    
    if not files:
        print("No files with hardcoded credentials found!")
        return
    
    print(f"\nFound {len(files)} files with hardcoded credentials:")
    for f in sorted(files):
        print(f"  - {f}")
    
    print("\nFixing files...")
    fixed_count = 0
    
    for filepath in files:
        if fix_file(filepath):
            fixed_count += 1
    
    print(f"\nFixed {fixed_count} files")
    
    # Verify
    print("\nVerifying...")
    remaining = find_files_with_credentials(root_dir)
    if remaining:
        print(f"⚠️  Still found credentials in {len(remaining)} files:")
        for f in remaining:
            print(f"  - {f}")
    else:
        print("✓ All hardcoded credentials have been removed!")

if __name__ == "__main__":
    main()