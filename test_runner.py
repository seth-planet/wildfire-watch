#!/usr/bin/env python3.12
"""
Test runner script to analyze and fix tests systematically
"""
import subprocess
import sys
import os
import time
import json

def run_single_test(test_file, test_name=None):
    """Run a single test and return results"""
    cmd = ["python3.12", "-m", "pytest", "-v", "--timeout=300", "--tb=short"]
    
    if test_name:
        cmd.append(f"{test_file}::{test_name}")
    else:
        cmd.append(test_file)
    
    print(f"Running: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=320)
        elapsed = time.time() - start_time
        
        return {
            "file": test_file,
            "test": test_name,
            "returncode": result.returncode,
            "elapsed": elapsed,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0
        }
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return {
            "file": test_file,
            "test": test_name,
            "returncode": -1,
            "elapsed": elapsed,
            "stdout": "",
            "stderr": "TIMEOUT",
            "success": False
        }

def analyze_test_output(result):
    """Analyze test output for common issues"""
    issues = []
    
    # Check for mocking issues
    if "Mock" in result["stdout"] or "Mock" in result["stderr"]:
        if "paho.mqtt" in result["stdout"] or "consensus" in result["stdout"] or "trigger" in result["stdout"]:
            issues.append("MOCKING_INTERNAL_MODULE")
    
    # Check for import errors
    if "ImportError" in result["stderr"] or "ModuleNotFoundError" in result["stderr"]:
        issues.append("IMPORT_ERROR")
    
    # Check for timeout
    if result["stderr"] == "TIMEOUT":
        issues.append("TIMEOUT")
    
    # Check for fixture errors
    if "fixture" in result["stderr"] and "not found" in result["stderr"]:
        issues.append("FIXTURE_NOT_FOUND")
    
    return issues

def main():
    # List of tests to analyze
    test_files = [
        "tests/test_consensus.py",
        "tests/test_detect.py",
        "tests/test_trigger.py",
        "tests/test_mqtt_broker.py",
        "tests/test_telemetry.py"
    ]
    
    results = []
    
    for test_file in test_files:
        print(f"\n{'='*60}")
        print(f"Testing: {test_file}")
        print(f"{'='*60}")
        
        result = run_single_test(test_file)
        issues = analyze_test_output(result)
        
        result["issues"] = issues
        results.append(result)
        
        print(f"Result: {'PASS' if result['success'] else 'FAIL'}")
        print(f"Time: {result['elapsed']:.2f}s")
        if issues:
            print(f"Issues: {', '.join(issues)}")
        
        # Save partial results
        with open("test_results_partial.json", "w") as f:
            json.dump(results, f, indent=2)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for r in results if r["success"])
    total = len(results)
    
    print(f"Tests run: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    
    # Save final results
    with open("test_results_final.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()