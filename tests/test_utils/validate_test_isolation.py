#!/usr/bin/env python3.12
"""
Validate test isolation improvements.
Run this to verify that tests now pass consistently.
"""
import subprocess
import sys
import time
import os
from pathlib import Path

class TestIsolationValidator:
    """Validate test isolation improvements"""
    
    def __init__(self):
        self.results = {}
        self.test_dir = Path(__file__).parent
        
    def run_validation(self):
        """Run validation suite"""
        print("=" * 60)
        print("Test Isolation Validation")
        print("=" * 60)
        
        # Step 1: Test individual test files
        print("\n1. Testing individual test files...")
        self.test_individual_files()
        
        # Step 2: Test small groups
        print("\n2. Testing small groups...")
        self.test_small_groups()
        
        # Step 3: Test all together
        print("\n3. Testing all together...")
        self.test_all_together()
        
        # Step 4: Test parallel execution
        print("\n4. Testing parallel execution...")
        self.test_parallel_execution()
        
        # Report results
        self.report_results()
    
    def run_pytest(self, args, description):
        """Run pytest with given arguments"""
        cmd = [sys.executable, "-m", "pytest"] + args
        print(f"\n{description}")
        print(f"Command: {' '.join(cmd)}")
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - start_time
        
        # Parse output
        passed = failed = errors = 0
        for line in result.stdout.split('\n'):
            if 'passed' in line and 'failed' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'passed':
                        passed = int(parts[i-1])
                    elif part == 'failed':
                        failed = int(parts[i-1])
                    elif part == 'error' in part:
                        errors = int(parts[i-1])
        
        success = result.returncode == 0
        
        self.results[description] = {
            'success': success,
            'passed': passed,
            'failed': failed,
            'errors': errors,
            'duration': duration,
            'returncode': result.returncode
        }
        
        if success:
            print(f"✓ PASSED ({passed} tests in {duration:.1f}s)")
        else:
            print(f"✗ FAILED ({failed} failures, {errors} errors in {duration:.1f}s)")
            if result.stderr:
                print(f"Stderr: {result.stderr[:200]}...")
        
        return success
    
    def test_individual_files(self):
        """Test each file individually"""
        test_files = [
            "test_consensus.py",
            "test_detect.py", 
            "test_trigger.py",
            "test_telemetry.py"
        ]
        
        for test_file in test_files:
            file_path = self.test_dir / test_file
            if file_path.exists():
                self.run_pytest(
                    [str(file_path), "-v", "-x"],
                    f"Individual: {test_file}"
                )
    
    def test_small_groups(self):
        """Test small groups of related tests"""
        # Test MQTT-dependent services together
        self.run_pytest(
            ["tests/test_consensus.py", "tests/test_telemetry.py", "-v"],
            "Group: MQTT services"
        )
        
        # Test detection services together
        self.run_pytest(
            ["tests/test_detect.py", "tests/test_detect_optimized.py", "-v"],
            "Group: Detection services"
        )
    
    def test_all_together(self):
        """Test all tests together"""
        self.run_pytest(
            ["tests/", "-v", "--tb=short"],
            "All tests together"
        )
    
    def test_parallel_execution(self):
        """Test parallel execution with pytest-xdist"""
        # Check if xdist is available
        try:
            import xdist
            self.run_pytest(
                ["tests/", "-n", "4", "-v"],
                "Parallel execution (4 workers)"
            )
        except ImportError:
            print("pytest-xdist not installed, skipping parallel test")
    
    def report_results(self):
        """Report validation results"""
        print("\n" + "=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)
        
        total_success = True
        
        for description, result in self.results.items():
            status = "✓ PASS" if result['success'] else "✗ FAIL"
            print(f"\n{status} {description}")
            print(f"     Tests: {result['passed']} passed, {result['failed']} failed, {result['errors']} errors")
            print(f"     Duration: {result['duration']:.1f}s")
            
            if not result['success']:
                total_success = False
        
        print("\n" + "=" * 60)
        if total_success:
            print("✓ ALL VALIDATIONS PASSED - Test isolation is working!")
        else:
            print("✗ SOME VALIDATIONS FAILED - Test isolation needs more work")
            print("\nRecommended next steps:")
            print("1. Check enhanced_mqtt_broker.py is being used")
            print("2. Verify test_isolation_fixtures.py is imported in conftest.py")
            print("3. Update test files to use clean fixtures")
            print("4. Check for remaining resource leaks")
        print("=" * 60)


def main():
    """Run validation"""
    # Set test environment
    os.environ['PYTEST_TIMEOUT'] = '60'  # 1 minute timeout per test
    os.environ['LOG_LEVEL'] = 'WARNING'  # Reduce noise
    
    validator = TestIsolationValidator()
    validator.run_validation()


if __name__ == "__main__":
    main()