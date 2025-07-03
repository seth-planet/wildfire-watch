#!/usr/bin/env python3.12
"""Simple test to diagnose pytest issues"""
import subprocess
import sys

print("Python version:", sys.version)
print("Working directory:", subprocess.run(['pwd'], capture_output=True, text=True).stdout.strip())

# Try running a simple pytest command
cmd = [sys.executable, "-m", "pytest", "--version"]
print(f"Running: {' '.join(cmd)}")
result = subprocess.run(cmd, capture_output=True, text=True)
print("Return code:", result.returncode)
print("Stdout:", result.stdout)
print("Stderr:", result.stderr)

# Try collecting tests
cmd = [sys.executable, "-m", "pytest", "tests/test_mqtt_broker.py", "--collect-only", "-q"]
print(f"\nRunning: {' '.join(cmd)}")
result = subprocess.run(cmd, capture_output=True, text=True)
print("Return code:", result.returncode)
print("Stdout:", result.stdout)
print("Stderr:", result.stderr)

# Try running a single test with minimal options
cmd = [sys.executable, "-m", "pytest", "tests/test_mqtt_broker.py::test_base_config_loading", "-v", "--no-header", "--tb=no"]
print(f"\nRunning: {' '.join(cmd)}")
result = subprocess.run(cmd, capture_output=True, text=True)
print("Return code:", result.returncode)
print("Stdout:", result.stdout)
print("Stderr:", result.stderr)