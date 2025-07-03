#!/usr/bin/env python3.12
"""
Find and run all failed tests
"""
import subprocess
import sys
import re

def get_all_tests():
    """Get all test methods from the test files"""
    test_files = [
        "tests/test_consensus.py",
        "tests/test_detect.py", 
        "tests/test_model_converter_e2e.py",
        "tests/test_integration_e2e.py",
        "tests/test_model_converter.py",
        "tests/test_frigate_integration.py",
        "tests/test_hardware_integration.py"
    ]
    
    all_tests = []
    
    for test_file in test_files:
        try:
            # Get test classes and methods
            result = subprocess.run(
                ["grep", "-E", "^class Test|^    def test_", test_file],
                capture_output=True,
                text=True
            )
            
            lines = result.stdout.strip().split('\n')
            current_class = None
            
            for line in lines:
                if line.startswith('class Test'):
                    # Extract class name
                    match = re.match(r'class (Test\w+)', line)
                    if match:
                        current_class = match.group(1)
                elif line.strip().startswith('def test_') and current_class:
                    # Extract method name
                    match = re.match(r'\s*def (test_\w+)', line.strip())
                    if match:
                        method = match.group(1)
                        all_tests.append(f"{test_file}::{current_class}::{method}")
                        
        except Exception as e:
            print(f"Error processing {test_file}: {e}")
            
    return all_tests

def run_test(test_path):
    """Run a single test and return result"""
    cmd = [
        sys.executable, "-m", "pytest", 
        test_path, 
        "-xvs",
        "--tb=short",
        "--timeout=120"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        # Check if test passed
        passed = result.returncode == 0 and "PASSED" in result.stdout
        # Check if test was skipped
        skipped = "SKIPPED" in result.stdout or "no tests ran" in result.stdout
        return passed, skipped, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, False, "", "Test timed out after 180 seconds"
    except Exception as e:
        return False, False, "", str(e)

def main():
    print("Finding all tests...")
    all_tests = get_all_tests()
    print(f"Found {len(all_tests)} total tests")
    
    # Run a subset first to check status
    print("\nRunning sample of tests to check status...")
    
    passed = []
    failed = []
    skipped = []
    
    # Test a representative sample
    sample_tests = [
        t for t in all_tests 
        if any(x in t for x in ["consensus", "detect", "integration", "model_converter"])
    ][:20]  # Test first 20
    
    for i, test in enumerate(sample_tests, 1):
        print(f"\n[{i}/{len(sample_tests)}] Running: {test}")
        success, skip, stdout, stderr = run_test(test)
        
        if skip:
            print("⊘ SKIPPED")
            skipped.append(test)
        elif success:
            print("✓ PASSED")
            passed.append(test)
        else:
            print("✗ FAILED")
            failed.append(test)
            if stderr and "FAILED" in stderr:
                # Extract failure reason
                for line in stderr.split('\n'):
                    if "FAILED" in line or "AssertionError" in line:
                        print(f"  Reason: {line.strip()[:100]}...")
                        break
    
    print("\n" + "=" * 80)
    print(f"\nSummary of sample:")
    print(f"  Total tested: {len(sample_tests)}")
    print(f"  Passed: {len(passed)} ({len(passed)/len(sample_tests)*100:.1f}%)")
    print(f"  Failed: {len(failed)} ({len(failed)/len(sample_tests)*100:.1f}%)")
    print(f"  Skipped: {len(skipped)} ({len(skipped)/len(sample_tests)*100:.1f}%)")
    
    if failed:
        print(f"\nFailed tests:")
        for test in failed[:10]:  # Show first 10
            print(f"  - {test}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")

if __name__ == "__main__":
    main()