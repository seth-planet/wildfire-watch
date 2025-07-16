"""Site customization for wildfire-watch.

This file is automatically imported by Python and ensures that
the project paths are set up correctly for multiprocessing.
"""
import os
import sys

# Add wildfire-watch root to Python path
wildfire_root = os.path.dirname(os.path.abspath(__file__))
if wildfire_root not in sys.path:
    sys.path.insert(0, wildfire_root)