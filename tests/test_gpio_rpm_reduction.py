#!/usr/bin/env python3.12
"""Test RPM reduction functionality for safe motor shutdown with REAL components.

This module tests the REDUCING_RPM state and ensures the motor properly
slows down before stopping to prevent damage to the pump.

BEST PRACTICES FOLLOWED:
1. NO mocking of PumpController or internal components
2. Uses real MQTT broker for all tests
3. Uses real GPIO module or built-in simulation
4. Tests actual RPM reduction behavior
"""

import pytest
import time
import threading
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import test fixtures from test_trigger
from tests.test_trigger import gpio_test_setup, wait_for_state

from gpio_trigger.trigger import PumpController, PumpState, CONFIG


class TestRPMReduction:
    """Test motor RPM reduction before shutdown with real components."""
    
    @pytest.fixture
    def rpm_controller(self, gpio_test_setup, monkeypatch, test_mqtt_broker, mqtt_topic_factory):
        """Create controller configured for RPM reduction testing.
        
        BEST PRACTICE: Real PumpController with real MQTT and GPIO.
        """
        # Get connection parameters from the test broker
        conn_params = test_mqtt_broker.get_connection_params()
        
        # Get unique topic prefix for test isolation
        full_topic = mqtt_topic_factory("dummy")
        topic_prefix = full_topic.rsplit('/', 1)[0]
        
        # Configure for RPM reduction testing
        monkeypatch.setenv("FIRE_OFF_DELAY", "5.0")  # 5 seconds
        monkeypatch.setenv("RPM_REDUCTION_LEAD", "3.0")  # 3 seconds before shutdown
        monkeypatch.setenv("MAX_ENGINE_RUNTIME", "30")  # 30 seconds
        monkeypatch.setenv("RPM_REDUCE_PIN", "27")
        monkeypatch.setenv("RPM_REDUCTION_DURATION", "3.0")
        monkeypatch.setenv("PRIMING_DURATION", "0.5")
        monkeypatch.setenv("IGNITION_START_DURATION", "0.1")
        monkeypatch.setenv("HEALTH_INTERVAL", "10")
        
        # Configure MQTT for testing
        monkeypatch.setenv("MQTT_BROKER", conn_params['host'])
        monkeypatch.setenv("MQTT_PORT", str(conn_params['port']))
        monkeypatch.setenv("MQTT_TLS", "false")
        monkeypatch.setenv("TOPIC_PREFIX", topic_prefix)
        
        # Reload module to pick up new environment
        import importlib
        if 'gpio_trigger.trigger' in sys.modules:
            del sys.modules['gpio_trigger.trigger']
        if 'trigger' in sys.modules:
            del sys.modules['trigger']
        
        # Re-import to get fresh config
        import gpio_trigger.trigger as trigger
        from gpio_trigger.trigger import PumpController, CONFIG, PumpState
        
        # Update globals
        globals()['trigger'] = trigger
        globals()['PumpController'] = PumpController
        globals()['CONFIG'] = CONFIG
        globals()['PumpState'] = PumpState
        
        # Create controller
        controller = PumpController()
        
        # Wait for MQTT connection
        time.sleep(1.0)
        
        yield controller
        
        # Cleanup
        controller._shutdown = True
        controller.cleanup()
    
    @pytest.mark.timeout(30)
    def test_rpm_reduction_before_fire_off_delay(self, rpm_controller, gpio_test_setup):
        """Test RPM reduction occurs before fire_off_delay expires."""
        if gpio_test_setup is None:
            pytest.skip("GPIO simulation not available")
            
        # Start pump
        rpm_controller.handle_fire_trigger()
        
        # Wait for pump to be fully running
        assert wait_for_state(rpm_controller, PumpState.RUNNING, timeout=2)
        
        # RPM reduce pin should be LOW initially
        assert gpio_test_setup.input(CONFIG['RPM_REDUCE_PIN']) is False
        
        # Wait for RPM reduction to start (fire_off_delay - rpm_lead = 5.0 - 3.0 = 2.0s)
        time.sleep(2.5)
        
        # Should be in RPM reduction state
        assert rpm_controller._state == PumpState.REDUCING_RPM
        assert gpio_test_setup.input(CONFIG['RPM_REDUCE_PIN']) is True
        
        # Engine should still be running during RPM reduction
        assert gpio_test_setup.input(CONFIG['IGN_ON_PIN']) is True
    
    @pytest.mark.timeout(30)
    def test_rpm_reduction_duration(self, rpm_controller, gpio_test_setup):
        """Test RPM reduction lasts for configured duration."""
        if gpio_test_setup is None:
            pytest.skip("GPIO simulation not available")
            
        # Start pump
        rpm_controller.handle_fire_trigger()
        assert wait_for_state(rpm_controller, PumpState.RUNNING, timeout=2)
        
        # Wait for RPM reduction
        assert wait_for_state(rpm_controller, PumpState.REDUCING_RPM, timeout=3)
        
        # Record when RPM reduction started
        start_time = time.time()
        
        # RPM reduce pin should be active
        assert gpio_test_setup.input(CONFIG['RPM_REDUCE_PIN']) is True
        
        # Wait for RPM reduction to complete
        assert wait_for_state(rpm_controller, PumpState.STOPPING, timeout=4)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Should be close to configured duration (3.0s)
        assert 2.5 <= duration <= 3.5, f"RPM reduction duration was {duration}s, expected ~3.0s"
        
        # RPM reduce pin should be deactivated
        assert gpio_test_setup.input(CONFIG['RPM_REDUCE_PIN']) is False
    
    @pytest.mark.timeout(30)
    def test_fire_trigger_during_rpm_reduction_cancels_shutdown(self, rpm_controller, gpio_test_setup):
        """Test fire trigger during RPM reduction cancels shutdown."""
        if gpio_test_setup is None:
            pytest.skip("GPIO simulation not available")
            
        # Start pump
        rpm_controller.handle_fire_trigger()
        assert wait_for_state(rpm_controller, PumpState.RUNNING, timeout=2)
        
        # Wait for RPM reduction
        assert wait_for_state(rpm_controller, PumpState.REDUCING_RPM, timeout=3)
        
        # Fire detected again during RPM reduction
        rpm_controller.handle_fire_trigger()
        
        # Give time for state change
        time.sleep(0.5)
        
        # Should cancel shutdown and return to running
        assert rpm_controller._state == PumpState.RUNNING
        assert gpio_test_setup.input(CONFIG['RPM_REDUCE_PIN']) is False
        assert gpio_test_setup.input(CONFIG['IGN_ON_PIN']) is True
    
    @pytest.mark.timeout(30)
    def test_max_runtime_with_rpm_reduction(self, rpm_controller, gpio_test_setup, monkeypatch):
        """Test RPM reduction occurs even at max runtime."""
        if gpio_test_setup is None:
            pytest.skip("GPIO simulation not available")
            
        # Set very short max runtime for testing
        monkeypatch.setenv("MAX_ENGINE_RUNTIME", "4.0")
        monkeypatch.setenv("RPM_REDUCTION_LEAD", "1.0")
        CONFIG['MAX_ENGINE_RUNTIME'] = 4.0
        CONFIG['RPM_REDUCTION_LEAD'] = 1.0
        
        # Start pump
        rpm_controller.handle_fire_trigger()
        assert wait_for_state(rpm_controller, PumpState.RUNNING, timeout=2)
        
        # Keep sending fire triggers to maintain operation
        for _ in range(3):
            time.sleep(0.5)
            rpm_controller.handle_fire_trigger()
        
        # Wait for RPM reduction (should start at max_runtime - rpm_lead = 4.0 - 1.0 = 3.0s)
        assert wait_for_state(rpm_controller, PumpState.REDUCING_RPM, timeout=2)
        
        # Verify RPM reduction is active
        assert gpio_test_setup.input(CONFIG['RPM_REDUCE_PIN']) is True
        
        # Wait for shutdown
        assert wait_for_state(rpm_controller, PumpState.STOPPING, timeout=2)
        
        # Engine should stop after max runtime
        time.sleep(0.5)
        assert gpio_test_setup.input(CONFIG['IGN_ON_PIN']) is False
    
    @pytest.mark.timeout(30)
    def test_rpm_reduction_pin_configuration(self, rpm_controller, gpio_test_setup):
        """Test RPM reduction pin is properly configured."""
        if gpio_test_setup is None:
            pytest.skip("GPIO simulation not available")
            
        # Verify RPM reduce pin is configured
        rpm_pin = CONFIG.get('RPM_REDUCE_PIN', 27)
        
        # Pin should be set up as output and initially LOW
        assert gpio_test_setup.input(rpm_pin) is False
        
        # Start pump and wait for RPM reduction
        rpm_controller.handle_fire_trigger()
        assert wait_for_state(rpm_controller, PumpState.RUNNING, timeout=2)
        assert wait_for_state(rpm_controller, PumpState.REDUCING_RPM, timeout=3)
        
        # Pin should be HIGH during RPM reduction
        assert gpio_test_setup.input(rpm_pin) is True
    
    @pytest.mark.timeout(30)
    def test_error_state_deactivates_rpm_reduction(self, rpm_controller, gpio_test_setup):
        """Test error state immediately deactivates RPM reduction."""
        if gpio_test_setup is None:
            pytest.skip("GPIO simulation not available")
            
        # Start pump
        rpm_controller.handle_fire_trigger()
        assert wait_for_state(rpm_controller, PumpState.RUNNING, timeout=2)
        
        # Wait for RPM reduction
        assert wait_for_state(rpm_controller, PumpState.REDUCING_RPM, timeout=3)
        assert gpio_test_setup.input(CONFIG['RPM_REDUCE_PIN']) is True
        
        # Force error state
        rpm_controller._enter_error_state("Test error during RPM reduction")
        
        # Should immediately deactivate all outputs
        assert rpm_controller._state == PumpState.ERROR
        assert gpio_test_setup.input(CONFIG['RPM_REDUCE_PIN']) is False
        assert gpio_test_setup.input(CONFIG['IGN_ON_PIN']) is False
    
    @pytest.mark.timeout(30)
    def test_cleanup_during_rpm_reduction(self, rpm_controller, gpio_test_setup):
        """Test cleanup properly handles RPM reduction state."""
        if gpio_test_setup is None:
            pytest.skip("GPIO simulation not available")
            
        # Start pump
        rpm_controller.handle_fire_trigger()
        assert wait_for_state(rpm_controller, PumpState.RUNNING, timeout=2)
        
        # Wait for RPM reduction
        assert wait_for_state(rpm_controller, PumpState.REDUCING_RPM, timeout=3)
        
        # Cleanup should safely stop everything
        rpm_controller.cleanup()
        
        # All outputs should be deactivated
        assert gpio_test_setup.input(CONFIG['RPM_REDUCE_PIN']) is False
        assert gpio_test_setup.input(CONFIG['IGN_ON_PIN']) is False
        assert gpio_test_setup.input(CONFIG['MAIN_VALVE_PIN']) is False


class TestRPMReductionIntegration:
    """Test RPM reduction integration with other features."""
    
    @pytest.mark.timeout(30)
    def test_rpm_reduction_with_refill(self, rpm_controller, gpio_test_setup):
        """Test RPM reduction followed by refill sequence."""
        if gpio_test_setup is None:
            pytest.skip("GPIO simulation not available")
            
        # Start pump
        rpm_controller.handle_fire_trigger()
        assert wait_for_state(rpm_controller, PumpState.RUNNING, timeout=2)
        
        # Wait for RPM reduction
        assert wait_for_state(rpm_controller, PumpState.REDUCING_RPM, timeout=3)
        
        # Wait for stopping
        assert wait_for_state(rpm_controller, PumpState.STOPPING, timeout=4)
        
        # Should transition to refilling
        assert wait_for_state(rpm_controller, PumpState.REFILLING, timeout=2)
        
        # Refill valve should be open
        assert gpio_test_setup.input(CONFIG['REFILL_VALVE_PIN']) is True
        
        # Engine and RPM reduction should be off
        assert gpio_test_setup.input(CONFIG['IGN_ON_PIN']) is False
        assert gpio_test_setup.input(CONFIG['RPM_REDUCE_PIN']) is False
    
    @pytest.mark.timeout(30)
    def test_rpm_reduction_telemetry(self, rpm_controller, test_mqtt_broker, mqtt_topic_factory):
        """Test RPM reduction events are published via MQTT."""
        import paho.mqtt.client as mqtt
        import json
        
        # Subscribe to telemetry
        telemetry_messages = []
        
        def on_message(client, userdata, msg):
            try:
                payload = json.loads(msg.payload.decode())
                telemetry_messages.append(payload)
            except:
                pass
        
        conn_params = test_mqtt_broker.get_connection_params()
        subscriber = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="test_rpm_telemetry")
        subscriber.on_message = on_message
        subscriber.connect(conn_params['host'], conn_params['port'])
        
        # Subscribe to telemetry topic
        telemetry_topic = mqtt_topic_factory("telemetry/#")
        subscriber.subscribe(telemetry_topic)
        subscriber.loop_start()
        
        # Start pump and wait for RPM reduction
        rpm_controller.handle_fire_trigger()
        assert wait_for_state(rpm_controller, PumpState.RUNNING, timeout=2)
        assert wait_for_state(rpm_controller, PumpState.REDUCING_RPM, timeout=3)
        
        # Wait for telemetry
        time.sleep(1.0)
        
        # Should have telemetry about RPM reduction
        rpm_events = [msg for msg in telemetry_messages 
                      if msg.get('state') == 'REDUCING_RPM' or 
                         msg.get('action') == 'rpm_reduction_start']
        
        assert len(rpm_events) > 0, "Should publish RPM reduction telemetry"
        
        # Cleanup
        subscriber.loop_stop()
        subscriber.disconnect()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])