#!/usr/bin/env python3.12
import sys
import pytest

# Run pytest directly
sys.exit(pytest.main([
    'tests/test_mqtt_broker.py::test_base_config_loading',
    '-v',
    '--tb=short'
]))