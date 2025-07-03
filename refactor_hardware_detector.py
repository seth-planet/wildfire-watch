#!/usr/bin/env python3
"""Refactor hardware_detector.py to use centralized command runner"""

import re
import sys

def refactor_command_calls(filepath):
    """
    Refactors self._run_command calls to use run_command with try/except blocks.
    """
    print(f"Processing file: {filepath}")
    with open(filepath, 'r') as f:
        lines = f.readlines()

    new_lines = []
    # Regex to find self._run_command calls, capturing:
    # Group 1: leading whitespace for indentation
    # Group 2: optional assignment part (e.g., 'variable = ')
    # Group 3: arguments inside run_command(...)
    pattern = re.compile(r"(\s*)([a-zA-Z0-9_]+\s*=\s*)?self\._run_command\((.*?)\)")

    i = 0
    while i < len(lines):
        line = lines[i]
        match = pattern.search(line)
        if match:
            indent = match.group(1)
            assignment_prefix = match.group(2)
            args_inside_call = match.group(3).strip() # Strip to handle leading/trailing spaces

            # Skip lines that are already refactored (in try blocks)
            if i > 0 and 'try:' in lines[i-1]:
                new_lines.append(line)
                i += 1
                continue

            # Determine the variable name for the stdout output
            # If there's an assignment, use that variable name. Otherwise, use 'output'
            output_var = "output"
            if assignment_prefix:
                output_var = assignment_prefix.split('=')[0].strip()

            # Add check=False to the arguments, ensuring it's not duplicated
            # This logic assumes check=False/True is not already deeply nested in args.
            if 'check=False' not in args_inside_call and 'check=True' not in args_inside_call:
                # Append check=False, ensuring it's the last argument before the closing parenthesis.
                # This simple append works for common cases like `['cmd']` or `['cmd'], timeout=X`.
                updated_args = f"{args_inside_call}, check=False"
            else:
                updated_args = args_inside_call

            # Check if the line continues (has if condition after)
            rest_of_line = line[match.end():]
            
            # Construct the new multi-line block
            new_block = []
            new_block.append(f"{indent}try:\n")
            new_block.append(f"{indent}    _, {output_var}, _ = run_command({updated_args})\n")
            new_block.append(f"{indent}except (FileNotFoundError, PermissionError, CommandError):\n")
            new_block.append(f"{indent}    {output_var} = \"\"\n") # Set to empty string on error as per requirement
            
            # Handle if conditions that were on the same line
            if rest_of_line.strip():
                new_block.append(f"{indent}{rest_of_line}")

            # Add the new block
            new_lines.extend(new_block)
            print(f"  Refactored line {i+1}: '{line.strip()}'")
        else:
            new_lines.append(line)
        i += 1

    # Write the modified content back to the file
    with open(filepath, 'w') as f:
        f.writelines(new_lines)
    print(f"Refactoring complete for {filepath}. Please review the changes carefully.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python refactor_script.py <path_to_hardware_detector.py>")
        sys.exit(1)
    filepath = sys.argv[1]
    refactor_command_calls(filepath)