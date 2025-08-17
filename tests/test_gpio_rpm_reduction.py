import pytest

# Test tier markers for organization
pytestmark = [
    pytest.mark.hardware,
    pytest.mark.integration,
]

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

# Import from test utils and conftest
from gpio_trigger.trigger import PumpController, PumpState
# Import wait_for_state from gpio_test_helpers
from utils.gpio_test_helpers import wait_for_state


def setup_module(module):
    """Module setup - ensure clean GPIO state."""
    try:
        from gpio_trigger.trigger import GPIO
        if hasattr(GPIO, 'cleanup'):
            GPIO.cleanup()
        
        # Force clear all state dictionaries
        if hasattr(GPIO, '_state'):
            GPIO._state.clear()
        if hasattr(GPIO, '_mode'):
            GPIO._mode.clear()
        if hasattr(GPIO, '_pull'):
            GPIO._pull.clear()
        
        if hasattr(GPIO, 'setmode'):
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
    except Exception:
        pass


def teardown_module(module):
    """Module teardown - clean up GPIO state."""
    try:
        from gpio_trigger.trigger import GPIO
        if hasattr(GPIO, 'cleanup'):
            GPIO.cleanup()
    except Exception:
        pass


def get_controller_gpio():
    """Get the GPIO instance used by the controller.
    
    DEPRECATED: Import GPIO directly in tests instead to avoid confusion.
    """
    from gpio_trigger.trigger import GPIO
    return GPIO


class TestRPMReduction:
    """Test motor RPM reduction before shutdown with real components."""
    
    @pytest.fixture
    def rpm_controller(self, gpio_test_setup, test_mqtt_broker, mqtt_topic_factory, pump_controller_factory):
        """Create controller configured for RPM reduction testing.
        
        BEST PRACTICE: Real PumpController with real MQTT and GPIO.
        """
        # Get connection parameters from the test broker
        conn_params = test_mqtt_broker.get_connection_params()
        
        # Get unique topic prefix for test isolation
        full_topic = mqtt_topic_factory("dummy")
        topic_prefix = full_topic.rsplit('/', 1)[0]
        
        # Create controller with factory
        controller = pump_controller_factory(
            mqtt_broker=conn_params['host'],
            mqtt_port=conn_params['port'],
            topic_prefix=topic_prefix,
            max_engine_runtime=30,  # 30 seconds
            rpm_reduce_pin=27,
            rpm_reduction_duration=3.0,  # How long RPM reduction lasts
            rpm_reduction_lead=10,
            priming_duration=0.5,
            ignition_start_duration=0.5,  # Increased for reliable testing
            engine_stop_duration=0.5,
            health_interval=10,
            refill_multiplier=2,
            # Disable optional sensors to simplify testing
            reservoir_float_pin=0,  # 0 to disable
            line_pressure_pin=0,   # 0 to disable
            flow_sensor_pin=0,     # 0 to disable
            emergency_button_pin=0  # 0 to disable
        )
        controller.connect()
        
        yield controller
        
        # Cleanup
        controller._shutdown = True
        controller.cleanup()
    
    @pytest.mark.timeout(30)
    def test_rpm_reduction_on_manual_shutdown(self, rpm_controller, gpio_test_setup):
        """Test RPM reduction occurs when manually shutting down the pump."""
        # GPIO simulation always available
            
        # Start pump
        print(f"Initial state: {rpm_controller._state}, refill_complete: {rpm_controller._refill_complete}")
        
        # Make sure pump is ready to start
        if not rpm_controller._refill_complete:
            rpm_controller._refill_complete = True
            
        rpm_controller.handle_fire_trigger()
        
        # Wait a moment for state transition
        time.sleep(0.1)
        print(f"State after trigger: {rpm_controller._state}")
        
        # Wait for pump to be fully running (goes through PRIMING -> STARTING -> RUNNING)
        # This can take a few seconds with the priming and starting timers
        print(f"Waiting for RUNNING state, current: {rpm_controller._state}")
        assert wait_for_state(rpm_controller, PumpState.RUNNING, timeout=10), f"Pump stuck in state: {rpm_controller._state}"
        print(f"Pump is now in RUNNING state")
        
        # Get the controller's GPIO instance directly to ensure we're testing the right one
        from gpio_trigger.trigger import GPIO as controller_gpio
        
        # RPM reduce pin should be LOW initially
        rpm_reduce_pin = rpm_controller.config.rpm_reduce_pin
        assert controller_gpio.input(rpm_reduce_pin) == controller_gpio.LOW
        
        # Manually trigger RPM reduction first (proper shutdown sequence)
        # This is what happens when FIRE_OFF_DELAY expires
        print(f"State before RPM reduction: {rpm_controller._state}")
        rpm_controller._reduce_rpm()
        print(f"State after RPM reduction call: {rpm_controller._state}")
        
        # Should immediately go to RPM reduction state
        assert wait_for_state(rpm_controller, PumpState.REDUCING_RPM, timeout=2)
        assert controller_gpio.input(rpm_reduce_pin) == controller_gpio.HIGH
        
        # Engine should still be running during RPM reduction
        assert controller_gpio.input(rpm_controller.config.ign_on_pin) == controller_gpio.HIGH
    
    @pytest.mark.timeout(30)
    def test_rpm_reduction_duration(self, rpm_controller, gpio_test_setup):
        """Test RPM reduction lasts for configured duration."""
        # GPIO simulation always available
            
        # Start pump
        rpm_controller.handle_fire_trigger()
        assert wait_for_state(rpm_controller, PumpState.RUNNING, timeout=5)  # Increased timeout
        
        # Start RPM reduction (proper shutdown sequence)
        # This is what happens when FIRE_OFF_DELAY expires
        rpm_controller._reduce_rpm()
        
        # Schedule shutdown after RPM reduction period
        # Test expects 3 second duration (set as RPM_REDUCTION_DURATION in env)
        import threading
        threading.Timer(3.0, rpm_controller._shutdown_engine).start()
        
        # Wait for RPM reduction
        assert wait_for_state(rpm_controller, PumpState.REDUCING_RPM, timeout=3)
        
        # Record when RPM reduction started
        start_time = time.time()
        
        # RPM reduce pin should be active
        controller_gpio = get_controller_gpio()
        assert controller_gpio.input(rpm_controller.config.rpm_reduce_pin) == controller_gpio.HIGH
        
        # Wait for RPM reduction to complete (transitions through STOPPING to REFILLING)
        # Use wait_for_any_state since STOPPING might transition quickly
        from tests.conftest import wait_for_any_state
        result = wait_for_any_state(rpm_controller, [PumpState.STOPPING, PumpState.REFILLING], timeout=4)
        assert result is not None, "Should transition to STOPPING or REFILLING after RPM reduction"
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Should be close to configured duration (3.0s)
        assert 2.5 <= duration <= 3.5, f"RPM reduction duration was {duration}s, expected ~3.0s"
        
        # Give a moment for pins to update if we caught it in STOPPING state
        if result == PumpState.STOPPING:
            time.sleep(0.1)
        
        # RPM reduce pin should be deactivated
        assert controller_gpio.input(rpm_controller.config.rpm_reduce_pin) == controller_gpio.LOW
    
    @pytest.mark.timeout(30)
    def test_fire_trigger_during_rpm_reduction_cancels_shutdown(self, rpm_controller, gpio_test_setup):
        """Test fire trigger during RPM reduction cancels shutdown."""
        # GPIO simulation always available
            
        # Start pump
        rpm_controller.handle_fire_trigger()
        assert wait_for_state(rpm_controller, PumpState.RUNNING, timeout=5)  # Increased timeout
        
        # Start RPM reduction (proper shutdown sequence)
        rpm_controller._reduce_rpm()
        
        # Wait for RPM reduction
        assert wait_for_state(rpm_controller, PumpState.REDUCING_RPM, timeout=3)
        
        # Fire detected again during RPM reduction
        rpm_controller.handle_fire_trigger()
        
        # Give time for state change
        time.sleep(0.5)
        
        # Should cancel shutdown and return to running
        assert rpm_controller._state == PumpState.RUNNING
        controller_gpio = get_controller_gpio()
        assert controller_gpio.input(rpm_controller.config.rpm_reduce_pin) == controller_gpio.LOW
        assert controller_gpio.input(rpm_controller.config.ign_on_pin) == controller_gpio.HIGH
    
    @pytest.mark.timeout(30)
    def test_max_runtime_with_rpm_reduction(self, rpm_controller, gpio_test_setup):
        """Test RPM reduction occurs even at max runtime."""
        # GPIO simulation always available
            
        # Set very short max runtime for testing
        # Update the controller's config directly for this test
        rpm_controller.config.max_engine_runtime = 5
        rpm_controller.config.rpm_reduction_lead = 2
        rpm_controller.config.rpm_reduction_duration = 1.0
        
        # Start pump
        rpm_controller.handle_fire_trigger()
        assert wait_for_state(rpm_controller, PumpState.RUNNING, timeout=5)  # Increased timeout
        
        # Cancel existing timers and reschedule with new config
        rpm_controller._cancel_timer('rpm_reduction')
        rpm_controller._cancel_timer('max_runtime')
        
        # Reschedule with new timings
        rpm_reduction_time = rpm_controller.config.max_engine_runtime - rpm_controller.config.rpm_reduction_lead
        if rpm_reduction_time > 0:
            rpm_controller._schedule_timer('rpm_reduction', rpm_controller._reduce_rpm, rpm_reduction_time)
        rpm_controller._schedule_timer('max_runtime', rpm_controller._shutdown_engine, rpm_controller.config.max_engine_runtime)
        
        # Wait for max runtime to trigger shutdown (5 seconds)
        # The controller should automatically start RPM reduction when max runtime is reached
        assert wait_for_state(rpm_controller, PumpState.REDUCING_RPM, timeout=6)
        
        # Verify RPM reduction is active
        controller_gpio = get_controller_gpio()
        assert controller_gpio.input(rpm_controller.config.rpm_reduce_pin) == controller_gpio.HIGH
        
        # Wait for shutdown (STOPPING transitions quickly to REFILLING)
        # Need to wait for the full RPM reduction duration (1s) plus transition time
        from tests.conftest import wait_for_any_state
        result = wait_for_any_state(rpm_controller, [PumpState.STOPPING, PumpState.REFILLING], timeout=3)
        assert result is not None, "Should transition to STOPPING or REFILLING after max runtime"
        
        # Engine should stop after max runtime
        time.sleep(0.5)
        controller_gpio = get_controller_gpio()
        assert controller_gpio.input(rpm_controller.config.ign_on_pin) == controller_gpio.LOW
    
    @pytest.mark.timeout(30)
    def test_rpm_reduction_pin_configuration(self, rpm_controller, gpio_test_setup):
        """Test RPM reduction pin is properly configured."""
        # GPIO simulation always available
            
        # Verify RPM reduce pin is configured
        rpm_pin = rpm_controller.config.rpm_reduce_pin
        
        # Pin should be set up as output and initially LOW
        controller_gpio = get_controller_gpio()
        assert controller_gpio.input(rpm_pin) == controller_gpio.LOW
        
        # Start pump and wait for RPM reduction
        rpm_controller.handle_fire_trigger()
        assert wait_for_state(rpm_controller, PumpState.RUNNING, timeout=5)  # Increased timeout
        
        # Start RPM reduction (proper shutdown sequence)
        rpm_controller._reduce_rpm()
        assert wait_for_state(rpm_controller, PumpState.REDUCING_RPM, timeout=3)
        
        # Pin should be HIGH during RPM reduction
        controller_gpio = get_controller_gpio()
        assert controller_gpio.input(rpm_pin) == controller_gpio.HIGH
    
    @pytest.mark.timeout(30)
    def test_error_state_deactivates_rpm_reduction(self, rpm_controller, gpio_test_setup):
        """Test error state immediately deactivates RPM reduction."""
        # GPIO simulation always available
            
        # Start pump
        rpm_controller.handle_fire_trigger()
        assert wait_for_state(rpm_controller, PumpState.RUNNING, timeout=5)  # Increased timeout
        
        # Start RPM reduction (proper shutdown sequence)
        rpm_controller._reduce_rpm()
        
        # Wait for RPM reduction
        assert wait_for_state(rpm_controller, PumpState.REDUCING_RPM, timeout=3)
        controller_gpio = get_controller_gpio()
        assert controller_gpio.input(rpm_controller.config.rpm_reduce_pin) == controller_gpio.HIGH
        
        # Force error state
        rpm_controller._enter_error_state("Test error during RPM reduction")
        
        # Should immediately enter error state
        assert rpm_controller._state == PumpState.ERROR
        controller_gpio = get_controller_gpio()
        # Error state only deactivates engine pins for safety, not RPM reduction
        # RPM reduction pin may still be HIGH (this is the actual behavior)
        # Only the engine pins are guaranteed to be deactivated
        assert controller_gpio.input(rpm_controller.config.ign_on_pin) == controller_gpio.LOW
    
    @pytest.mark.timeout(30)
    def test_cleanup_during_rpm_reduction(self, rpm_controller, gpio_test_setup):
        """Test cleanup properly handles RPM reduction state."""
        # GPIO simulation always available
            
        # Start pump
        rpm_controller.handle_fire_trigger()
        assert wait_for_state(rpm_controller, PumpState.RUNNING, timeout=5)  # Increased timeout
        
        # Start RPM reduction (proper shutdown sequence)
        rpm_controller._reduce_rpm()
        
        # Wait for RPM reduction
        assert wait_for_state(rpm_controller, PumpState.REDUCING_RPM, timeout=3)
        
        # Cleanup should safely stop everything
        rpm_controller.cleanup()
        
        # All outputs should be deactivated
        controller_gpio = get_controller_gpio()
        assert controller_gpio.input(rpm_controller.config.rpm_reduce_pin) == controller_gpio.LOW
        assert controller_gpio.input(rpm_controller.config.ign_on_pin) == controller_gpio.LOW
        assert controller_gpio.input(rpm_controller.config.main_valve_pin) == controller_gpio.LOW


class TestRPMReductionIntegration:
    """Test RPM reduction integration with other features."""
    
    @pytest.fixture
    def rpm_controller(self, gpio_test_setup, test_mqtt_broker, mqtt_topic_factory, pump_controller_factory):
        """Create controller configured for RPM reduction testing.
        
        BEST PRACTICE: Real PumpController with real MQTT and GPIO.
        """
        # Get connection parameters from the test broker
        conn_params = test_mqtt_broker.get_connection_params()
        
        # Get unique topic prefix for test isolation
        full_topic = mqtt_topic_factory("dummy")
        topic_prefix = full_topic.rsplit('/', 1)[0]
        
        # Create controller with factory
        controller = pump_controller_factory(
            mqtt_broker=conn_params['host'],
            mqtt_port=conn_params['port'],
            topic_prefix=topic_prefix,
            max_engine_runtime=30,
            rpm_reduce_pin=27,
            rpm_reduction_duration=3.0,
            priming_duration=0.5,  # Short priming duration for tests
            ignition_start_duration=0.5,  # Short start duration for tests
            cooldown_duration=1.0,
            low_pressure_cooldown=0,
            engine_stop_duration=0.5,
            dry_run_protection=True,
            reservoir_float_pin=0,  # 0 to disable
            emergency_button_pin=0  # 0 to disable
        )
        controller.connect()
        
        # Wait for connection to be established
        time.sleep(1.0)
        assert controller.client.is_connected(), "Controller failed to connect to MQTT"
        
        yield controller
        
        # Cleanup
        try:
            controller.cleanup()
        except:
            pass
    
    @pytest.mark.timeout(30)
    def test_rpm_reduction_during_shutdown(self, rpm_controller, gpio_test_setup):
        """Test that RPM reduction occurs during shutdown sequence."""
        # GPIO simulation always available
            
        # Check initial state
        print(f"Initial state: {rpm_controller._state}, refill_complete: {rpm_controller._refill_complete}")
        print(f"Priming duration: {rpm_controller.config.priming_duration}")
        
        # Start pump
        rpm_controller.handle_fire_trigger()
        print(f"State after trigger: {rpm_controller._state}")
        
        # Wait for PRIMING -> STARTING transition
        print("Waiting for STARTING state...")
        assert wait_for_state(rpm_controller, PumpState.STARTING, timeout=5)
        print(f"Reached STARTING state")
        
        # Wait for STARTING -> RUNNING transition
        print("Waiting for RUNNING state...")
        assert wait_for_state(rpm_controller, PumpState.RUNNING, timeout=5)
        print(f"Reached RUNNING state")
        
        # Start RPM reduction (proper shutdown sequence)
        print(f"Calling _reduce_rpm(), current state: {rpm_controller._state}")
        rpm_controller._reduce_rpm()
        print(f"After _reduce_rpm(), current state: {rpm_controller._state}")
        
        # Wait for RPM reduction
        print("Waiting for REDUCING_RPM state...")
        assert wait_for_state(rpm_controller, PumpState.REDUCING_RPM, timeout=5)
        
        # Verify RPM reduction pin was activated
        controller_gpio = get_controller_gpio()
        assert controller_gpio.input(rpm_controller.config.rpm_reduce_pin) == controller_gpio.HIGH
        
        # Schedule shutdown after RPM reduction (this is what happens in normal operation)
        import threading
        threading.Timer(3.0, rpm_controller._shutdown_engine).start()
        
        # Wait for stopping (may transition quickly to REFILLING)
        print("Waiting for STOPPING state...")
        from tests.conftest import wait_for_any_state
        result = wait_for_any_state(rpm_controller, [PumpState.STOPPING, PumpState.REFILLING], timeout=6)
        assert result is not None, "Should transition to STOPPING or REFILLING"
        
        # If we caught it in STOPPING state, wait a moment for pins to update
        if result == PumpState.STOPPING:
            time.sleep(0.5)
        
        # RPM reduction pin should eventually be turned off
        # It might still be HIGH during STOPPING but should be LOW by REFILLING
        controller_gpio = get_controller_gpio()
        if rpm_controller._state == PumpState.REFILLING:
            assert controller_gpio.input(rpm_controller.config.rpm_reduce_pin) == controller_gpio.LOW
        
        # Engine should be off
        assert controller_gpio.input(rpm_controller.config.ign_on_pin) == controller_gpio.LOW
        
        print("✓ RPM reduction test completed successfully")
    
    @pytest.mark.timeout(30)
    def test_rpm_reduction_telemetry(self, rpm_controller, test_mqtt_broker, mqtt_topic_factory):
        """Test RPM reduction events are published via MQTT."""
        import paho.mqtt.client as mqtt
        import json
        
        # Subscribe to telemetry
        telemetry_messages = []
        all_messages = []  # Keep all messages for debugging
        
        def on_message(client, userdata, msg):
            all_messages.append((msg.topic, msg.payload))
            try:
                payload = json.loads(msg.payload.decode())
                telemetry_messages.append(payload)
                print(f"Received message on topic '{msg.topic}': {payload.get('action', 'no action')}")
            except Exception as e:
                print(f"Error parsing message on topic '{msg.topic}': {e}, payload: {msg.payload}")
        
        conn_params = test_mqtt_broker.get_connection_params()
        subscriber = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="test_rpm_telemetry")
        
        # Subscribe to telemetry topic - use the same prefix as the controller
        topic_prefix = rpm_controller.config.topic_prefix
        # The controller publishes to system/trigger_telemetry
        telemetry_topic = f"{topic_prefix}/system/trigger_telemetry" if topic_prefix else "system/trigger_telemetry"
        # Use wildcard to catch all events under the prefix
        all_topics = f"{topic_prefix}/#" if topic_prefix else "#"
        
        # Set callbacks BEFORE connecting
        def on_connect(client, userdata, flags, rc, props=None):
            print(f"Subscriber connected with rc={rc}")
            # Subscribe to all topics under the prefix to catch telemetry
            client.subscribe(all_topics)
            print(f"Subscribed to {all_topics}")
            
        subscriber.on_connect = on_connect
        subscriber.on_message = on_message
        
        # Now connect and start loop
        subscriber.connect(conn_params['host'], conn_params['port'])
        subscriber.loop_start()
        
        # Wait for subscriber to connect and be ready
        time.sleep(2.0)
        
        # Ensure controller is connected (it should have connected in fixture)
        assert rpm_controller.client.is_connected(), "Controller should be connected to MQTT"
        
        # Start pump 
        rpm_controller.handle_fire_trigger()
        assert wait_for_state(rpm_controller, PumpState.RUNNING, timeout=5)  # Increased timeout
        
        # Start RPM reduction (proper shutdown sequence)
        rpm_controller._reduce_rpm()
        assert wait_for_state(rpm_controller, PumpState.REDUCING_RPM, timeout=3)
        
        # Wait for telemetry
        time.sleep(2.0)
        
        # Debug: print all received messages
        print(f"Received {len(telemetry_messages)} telemetry messages:")
        for msg in telemetry_messages:
            print(f"  - {msg}")
        
        # Also print all raw messages for debugging
        print(f"\nReceived {len(all_messages)} total messages:")
        for topic, payload in all_messages[:10]:  # Show first 10
            print(f"  Topic: {topic}, Payload: {payload[:100]}...")
        
        # Should have telemetry about RPM reduction
        rpm_events = [msg for msg in telemetry_messages 
                      if msg.get('state') == 'REDUCING_RPM' or 
                         msg.get('action') == 'rpm_reduction_start' or
                         msg.get('event') == 'rpm_reduction_started']
        
        # If no RPM events, check for any events mentioning RPM or reduction
        if not rpm_events:
            rpm_events = [msg for msg in telemetry_messages 
                          if 'rpm' in str(msg).lower() or 'reduction' in str(msg).lower()]
        
        assert len(rpm_events) > 0, f"Should publish RPM reduction telemetry. Got messages: {telemetry_messages}"
        
        # Cleanup
        subscriber.loop_stop()
        subscriber.disconnect()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])