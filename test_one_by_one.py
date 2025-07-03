#!/usr/bin/env python3.12
"""Run tests one by one to identify issues"""
import os
import sys
import time

# Ensure tests directory is in path
sys.path.insert(0, 'tests')

def run_test_module(module_name):
    """Import and check a test module"""
    print(f"\n{'='*60}")
    print(f"Testing: {module_name}")
    print(f"{'='*60}")
    
    try:
        # Import the module
        print(f"Importing {module_name}...")
        module = __import__(module_name)
        print("✓ Import successful")
        
        # Check for test functions
        test_funcs = [name for name in dir(module) if name.startswith('test_') and callable(getattr(module, name))]
        print(f"✓ Found {len(test_funcs)} test functions")
        
        # Check for fixtures
        fixtures = [name for name in dir(module) if hasattr(getattr(module, name), '__wrapped__')]
        print(f"✓ Found {len(fixtures)} fixtures")
        
        # Check for classes
        test_classes = [name for name in dir(module) if name.startswith('Test') and isinstance(getattr(module, name), type)]
        print(f"✓ Found {len(test_classes)} test classes")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Check all test modules"""
    print("Wildfire Watch Test Module Checker")
    print(f"Python: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    test_modules = [
        'test_mqtt_broker',
        'test_consensus', 
        'test_detect',
        'test_trigger',
        'test_telemetry',
        'test_simplified_integration',
    ]
    
    results = {}
    
    for module in test_modules:
        success = run_test_module(module)
        results[module] = success
        time.sleep(0.5)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    
    print(f"Modules checked: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed modules:")
        for module, success in results.items():
            if not success:
                print(f"  - {module}")

if __name__ == "__main__":
    main()