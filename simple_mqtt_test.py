#!/usr/bin/env python3.12
"""Run a simple MQTT broker test"""
import pytest
import sys

# Run a single simple test
exit_code = pytest.main([
    'tests/test_mqtt_broker.py::TestConfigurationFiles::test_main_mosquitto_config_exists',
    '-v',
    '--tb=short',
    '--capture=no'
])

print(f"\nTest exit code: {exit_code}")
sys.exit(exit_code)