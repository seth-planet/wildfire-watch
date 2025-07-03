#!/usr/bin/env python3.12
"""Direct test execution to see errors"""
import os
os.environ['PYTEST_CURRENT_TEST'] = 'test'

# Import test module directly
try:
    import sys
    sys.path.insert(0, '.')
    
    # Try importing the test
    print("Importing test_consensus...")
    from tests import test_consensus
    print("Import successful!")
    
    # Try to get fixtures
    print("\nChecking fixtures...")
    import inspect
    fixtures = [name for name, obj in inspect.getmembers(test_consensus) 
                if hasattr(obj, '__name__') and 'fixture' in str(obj)]
    print(f"Found fixtures: {fixtures}")
    
    # Try running a simple test function
    print("\nLooking for test functions...")
    test_funcs = [name for name, obj in inspect.getmembers(test_consensus)
                  if name.startswith('test_') and callable(obj)]
    print(f"Found test functions: {test_funcs[:5]}...")  # First 5
    
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()