#!/usr/bin/env python3.12
"""
Script to verify test isolation fixes
"""
import subprocess
import sys
import time

def run_test_suite(test_files, runs=3):
    """Run test suite multiple times and check for failures"""
    print(f"Running test suite {runs} times to verify isolation...")
    
    results = []
    for i in range(runs):
        print(f"\nRun {i+1}/{runs}:")
        cmd = ["python3.12", "-m", "pytest"] + test_files + ["-v", "--tb=short", "-x"]
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - start_time
        
        # Count passed/failed
        output = result.stdout + result.stderr
        passed = output.count(" PASSED")
        failed = output.count(" FAILED")
        
        results.append({
            'run': i+1,
            'returncode': result.returncode,
            'passed': passed,
            'failed': failed,
            'duration': duration
        })
        
        print(f"  Passed: {passed}, Failed: {failed}, Duration: {duration:.2f}s")
        
        if result.returncode != 0:
            print("  FAILURE DETECTED!")
            print("  Last 50 lines of output:")
            print("-" * 80)
            lines = output.split('\n')
            for line in lines[-50:]:
                print(f"  {line}")
            print("-" * 80)
    
    return results

def main():
    # Test the most problematic files
    test_files = [
        "tests/test_trigger.py",
        "tests/test_simplified_integration.py"
    ]
    
    print("=" * 80)
    print("TEST ISOLATION VERIFICATION")
    print("=" * 80)
    
    # Run the test suite
    results = run_test_suite(test_files, runs=3)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    all_passed = all(r['returncode'] == 0 for r in results)
    
    if all_passed:
        print("✅ ALL RUNS PASSED - Test isolation is working!")
    else:
        print("❌ SOME RUNS FAILED - Test isolation issues remain")
    
    for r in results:
        status = "✅ PASSED" if r['returncode'] == 0 else "❌ FAILED"
        print(f"Run {r['run']}: {status} ({r['passed']} passed, {r['failed']} failed)")
    
    # Test individual files
    print("\n" + "=" * 80)
    print("TESTING INDIVIDUAL FILES")
    print("=" * 80)
    
    for test_file in test_files:
        print(f"\nTesting {test_file} individually:")
        cmd = ["python3.12", "-m", "pytest", test_file, "-v", "--tb=short"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"  ✅ PASSED when run alone")
        else:
            print(f"  ❌ FAILED when run alone")
    
    # Test all together
    print("\n" + "=" * 80)
    print("TESTING ALL TOGETHER")
    print("=" * 80)
    
    cmd = ["python3.12", "-m", "pytest", "tests/", "-v", "--tb=short", "-k", "test_trigger or test_simplified"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    output = result.stdout + result.stderr
    passed = output.count(" PASSED")
    failed = output.count(" FAILED")
    
    print(f"All tests together: {passed} passed, {failed} failed")
    
    if result.returncode != 0:
        print("\nFailed tests:")
        lines = output.split('\n')
        for line in lines:
            if "FAILED" in line and "::" in line:
                print(f"  - {line.strip()}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())