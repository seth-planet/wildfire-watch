#!/usr/bin/env python3
"""Script to update hardware_detector.py to use new command runner"""

import re

# Read the file
with open('security_nvr/hardware_detector.py', 'r') as f:
    content = f.read()

# Replace all self._run_command calls with run_command
# Pattern: self._run_command(cmd) -> run_command(cmd, check=False)[1] or ""
pattern = r'self\._run_command\(([^)]+)\)'

def replacement(match):
    args = match.group(1)
    # Check if there are additional arguments
    if ',' in args:
        # Extract just the command argument
        parts = args.split(',', 1)
        cmd = parts[0].strip()
        # Try to extract timeout if present
        timeout_match = re.search(r'timeout\s*=\s*(\d+)', parts[1])
        if timeout_match:
            timeout = timeout_match.group(1)
            return f'run_command({cmd}, timeout={timeout}, check=False)[1] or ""'
    else:
        cmd = args.strip()
    return f'run_command({cmd}, check=False)[1] or ""'

# Apply replacements
updated_content = re.sub(pattern, replacement, content)

# Handle special cases where the result is used differently
# For lines where the result is checked directly
updated_content = re.sub(
    r'(\w+) = run_command\(([^)]+), check=False\)\[1\] or ""',
    lambda m: f'try:\n            _, {m.group(1)}, _ = run_command({m.group(2)}, check=False)\n        except (CommandError, FileNotFoundError, PermissionError):\n            {m.group(1)} = ""',
    updated_content
)

# Write the updated file
with open('security_nvr/hardware_detector.py', 'w') as f:
    f.write(updated_content)

print("Updated hardware_detector.py to use centralized command runner")