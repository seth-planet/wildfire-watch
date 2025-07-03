#!/usr/bin/env python3.12
"""Example of using shared fixtures with refactored services."""

import pytest
from tests.fixtures.refactored_fixtures import *
from tests.fixtures.test_helpers import ServiceTestHelper, MockCamera, MockDetection


def test_example_with_fixtures(mock_service_config, mock_mqtt_client):
    """Example test using shared fixtures."""
    # Your test code here
    assert mock_service_config.mqtt_broker == 'localhost'
    assert mock_mqtt_client.is_connected()


def test_example_with_helper(refactored_service_factory):
    """Example using the service factory."""
    # This would create a real service instance with mocked MQTT
    # service, mqtt_client = refactored_service_factory(YourServiceClass)
    pass
