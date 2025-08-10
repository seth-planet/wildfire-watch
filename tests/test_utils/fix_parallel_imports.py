"""Fix for parallel test execution import issues.

This module ensures that the wildfire-watch modules can be imported
correctly when using multiprocessing in 'spawn' mode.
"""
import os
import sys

def setup_python_path():
    """Add wildfire-watch root to Python path for imports."""
    # Get the wildfire-watch root directory
    wildfire_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Add to Python path if not already there
    if wildfire_root not in sys.path:
        sys.path.insert(0, wildfire_root)
    
    # Also add the parent directory for utils imports
    parent_dir = os.path.dirname(wildfire_root)
    if parent_dir not in sys.path and os.path.exists(os.path.join(parent_dir, 'utils')):
        sys.path.insert(0, parent_dir)

# Call setup when module is imported
setup_python_path()