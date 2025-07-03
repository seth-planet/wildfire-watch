#!/usr/bin/env python3
"""Run tests with appropriate Python versions based on requirements.

This script routes different test files to the correct Python version:
- Python 3.12: Main application tests (consensus, detect, trigger, etc.)
- Python 3.10: YOLO-NAS and super-gradients tests
- Python 3.8: Coral TPU and tflite_runtime tests
"""
import subprocess
import sys
import os
import json
from pathlib import Path

# Test groups by Python version based on README.md requirements
PYTHON_312_TESTS = {
    "description": "Main application tests",
    "tests": [
        "tests/test_consensus.py",
        "tests/test_detect.py", 
        "tests/test_trigger.py",
        "tests/test_telemetry.py",
        "tests/test_mqtt_broker.py",
        "tests/test_simplified_integration.py",
        "tests/test_integration_e2e.py",
        "tests/test_frigate_integration.py",
        "tests/test_security_nvr.py",
    ],
    "markers": "-k 'not (api_usage or yolo_nas or qat_functionality or int8_quantization or model_converter or hardware_integration or deployment)'"
}

PYTHON_310_TESTS = {
    "description": "YOLO-NAS and super-gradients tests", 
    "tests": [
        "tests/test_api_usage.py",
        "tests/test_yolo_nas_training.py",
        "tests/test_qat_functionality.py",
    ],
    "markers": "-m 'api_usage or yolo_nas or qat_functionality'"
}

PYTHON_38_TESTS = {
    "description": "Coral TPU and tflite_runtime tests",
    "tests": [
        "tests/test_model_converter.py",
        "tests/test_hardware_integration.py", 
        "tests/test_deployment.py",
        "tests/test_int8_quantization.py",
    ],
    "markers": "-m 'model_converter or hardware_integration or deployment or int8_quantization'"
}


def check_python_version(version):
    """Check if Python version is available."""
    python_cmd = f"python{version}"
    try:
        result = subprocess.run(
            [python_cmd, "--version"], 
            check=True, 
            capture_output=True, 
            text=True
        )
        print(f"✓ Found {result.stdout.strip()}")
        return python_cmd
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"✗ Python {version} not found")
        return None


def run_tests(python_cmd, test_group, version):
    """Run tests with specific Python version."""
    if not python_cmd:
        print(f"\nSkipping Python {version} tests - Python not available")
        return 0
    
    print(f"\n{'='*60}")
    print(f"Running {test_group['description']} with Python {version}")
    print(f"{'='*60}")
    
    # Filter existing test files
    existing_tests = [t for t in test_group['tests'] if os.path.exists(t)]
    if not existing_tests:
        print(f"No test files found for Python {version}")
        return 0
    
    # Build pytest command
    cmd = [
        python_cmd, "-m", "pytest",
        "-v",
        "--timeout=300",
        "--tb=short",
        "--no-header"
    ]
    
    # Add test files or markers
    if 'markers' in test_group:
        cmd.append(test_group['markers'])
        cmd.append("tests/")  # Search in tests directory
    else:
        cmd.extend(existing_tests)
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Run tests
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def main():
    """Main entry point."""
    print("Wildfire Watch Test Runner")
    print("Running tests with appropriate Python versions...")
    print()
    
    # Check available Python versions
    python312 = check_python_version("3.12")
    python310 = check_python_version("3.10") 
    python38 = check_python_version("3.8")
    
    # Track results
    results = []
    
    # Run Python 3.12 tests (main tests)
    if python312:
        exit_code = run_tests(python312, PYTHON_312_TESTS, "3.12")
        results.append(("Python 3.12", exit_code))
    
    # Run Python 3.10 tests (YOLO-NAS)
    if python310:
        exit_code = run_tests(python310, PYTHON_310_TESTS, "3.10")
        results.append(("Python 3.10", exit_code))
    
    # Run Python 3.8 tests (Coral TPU)
    if python38:
        exit_code = run_tests(python38, PYTHON_38_TESTS, "3.8")
        results.append(("Python 3.8", exit_code))
    
    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    
    total_failures = 0
    for version, exit_code in results:
        status = "PASSED" if exit_code == 0 else "FAILED"
        print(f"{version}: {status}")
        if exit_code != 0:
            total_failures += 1
    
    print()
    
    # Exit with error if any tests failed
    if total_failures > 0:
        print(f"Total test suites with failures: {total_failures}")
        sys.exit(1)
    else:
        print("All test suites passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()