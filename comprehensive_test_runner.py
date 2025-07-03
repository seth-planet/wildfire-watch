#!/usr/bin/env python3.12
"""
Comprehensive test runner for Wildfire Watch
Runs all tests and generates detailed reports
"""
import pytest
import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple

class TestRunner:
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        
    def run_test_file(self, test_file: str) -> Tuple[int, str]:
        """Run all tests in a file and return results"""
        print(f"\n{'='*60}")
        print(f"Running: {test_file}")
        print(f"{'='*60}")
        
        # Run pytest programmatically
        args = [
            test_file,
            '-v',
            '--tb=short',
            '--timeout=300',
            '-q'
        ]
        
        start = time.time()
        exit_code = pytest.main(args)
        duration = time.time() - start
        
        status = "PASSED" if exit_code == 0 else "FAILED"
        print(f"Status: {status} (took {duration:.2f}s)")
        
        return exit_code, status
    
    def check_for_mocking_violations(self, test_file: str) -> List[str]:
        """Check if test file violates integration testing philosophy"""
        violations = []
        
        try:
            with open(test_file, 'r') as f:
                content = f.read()
                
            # Check for MockMQTTClient
            if 'MockMQTTClient' in content:
                violations.append("Uses MockMQTTClient instead of real MQTT broker")
            
            # Check for mocking internal modules
            internal_modules = ['consensus', 'trigger', 'detect', 'telemetry', 'camera_detector']
            for module in internal_modules:
                if f"@patch('{module}" in content or f'@patch("{module}' in content:
                    violations.append(f"Mocks internal module: {module}")
                    
            # Check for paho.mqtt mocking
            if "@patch('paho.mqtt" in content or '@patch("paho.mqtt' in content:
                violations.append("Mocks paho.mqtt.client - should use real MQTT broker")
                
        except Exception as e:
            violations.append(f"Error checking file: {e}")
            
        return violations
    
    def run_all_tests(self):
        """Run all tests and generate report"""
        # Test files to run
        test_files = [
            # Core service tests
            "tests/test_consensus.py",
            "tests/test_detect.py", 
            "tests/test_trigger.py",
            "tests/test_telemetry.py",
            "tests/test_mqtt_broker.py",
            
            # Integration tests
            "tests/test_integration_e2e.py",
            "tests/test_simplified_integration.py",
            "tests/test_tls_integration.py",
            
            # Hardware tests (if hardware available)
            "tests/test_hardware_integration.py",
            
            # Model tests (Python 3.12 compatible)
            "tests/test_model_converter.py",
        ]
        
        for test_file in test_files:
            if not os.path.exists(test_file):
                print(f"Skipping {test_file} - file not found")
                continue
                
            # Check for violations first
            violations = self.check_for_mocking_violations(test_file)
            
            # Run the test
            exit_code, status = self.run_test_file(test_file)
            
            # Store results
            self.results[test_file] = {
                'status': status,
                'exit_code': exit_code,
                'violations': violations
            }
            
            # Brief pause between test files
            time.sleep(1)
    
    def generate_report(self):
        """Generate final test report"""
        print(f"\n{'='*60}")
        print("FINAL TEST REPORT")
        print(f"{'='*60}")
        
        total_duration = time.time() - self.start_time
        passed = sum(1 for r in self.results.values() if r['status'] == 'PASSED')
        failed = sum(1 for r in self.results.values() if r['status'] == 'FAILED')
        
        print(f"\nTotal tests files: {len(self.results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Total duration: {total_duration:.2f}s")
        
        if failed > 0:
            print("\nFailed tests:")
            for test_file, result in self.results.items():
                if result['status'] == 'FAILED':
                    print(f"  - {test_file}")
        
        # Check for violations
        tests_with_violations = {
            f: v['violations'] 
            for f, v in self.results.items() 
            if v['violations']
        }
        
        if tests_with_violations:
            print("\nIntegration Testing Violations:")
            for test_file, violations in tests_with_violations.items():
                print(f"\n{test_file}:")
                for violation in violations:
                    print(f"  - {violation}")
        
        # Save detailed results
        with open("test_results_detailed.json", "w") as f:
            json.dump(self.results, f, indent=2)
            
        print("\nDetailed results saved to: test_results_detailed.json")
        
        return failed == 0

def main():
    """Main entry point"""
    print("Wildfire Watch Comprehensive Test Runner")
    print(f"Python: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    runner = TestRunner()
    runner.run_all_tests()
    success = runner.generate_report()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()