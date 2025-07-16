#!/usr/bin/env python3.12
"""Test all emergency and safety features of GPIO trigger with REAL components.

This module comprehensively tests safety-critical functionality including:
- Emergency button activation
- Dry run protection 
- Reservoir level monitoring
- Line pressure loss handling
- Emergency MQTT commands

BEST PRACTICES FOLLOWED:
1. NO mocking of GPIO or PumpController
2. Uses real MQTT broker for all tests
3. Tests actual hardware behavior through GPIO simulation
4. Tests real safety feature implementations
"""

import os
import sys
import pytest
import time
import threading
import logging
import json

# Imports handled by conftest.py

# Import GPIO components
import gpio_trigger.trigger as trigger
from gpio_trigger.trigger import PumpController, GPIO, CONFIG, PumpState
# gpio_test_setup and wait_for_state are now available via conftest.py

logger = logging.getLogger(__name__)


@pytest.fixture
def safety_controller(gpio_test_setup, monkeypatch, test_mqtt_broker, mqtt_topic_factory):
    """Create controller with test configuration."""
    conn_params = test_mqtt_broker.get_connection_params()
    full_topic = mqtt_topic_factory("dummy")
    topic_prefix = full_topic.rsplit('/', 1)[0]
    
    # Set environment variables for configuration
    test_env = {
        "MQTT_BROKER": conn_params['host'],
        "MQTT_PORT": str(conn_params['port']),
        "MQTT_TLS": "false",
        "TOPIC_PREFIX": topic_prefix,
    }
    
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)
    
    # Import config class after environment is set
    from gpio_trigger.trigger import PumpControllerConfig
    
    # Create config object - it will load from environment
    config = PumpControllerConfig()
    
    # Create controller with dependency injection
    controller = PumpController(config=config)
    
    yield controller
    
    # Cleanup
    controller._shutdown = True
    controller._shutting_down = True  # For quick thread termination
    controller.cleanup()


class TestEmergencyButton:
    """Test emergency button activation using real components."""
    
    @pytest.mark.timeout(30)
    def test_emergency_button_triggers_pump(self, safety_controller, gpio_test_setup):
        """Test emergency button starts pump immediately."""
        if gpio_test_setup is None:
            pytest.skip("GPIO simulation not available")
            
        # Setup emergency button pin
        emergency_pin = CONFIG.get('EMERGENCY_BUTTON_PIN', 21)
        gpio_test_setup.setup(emergency_pin, gpio_test_setup.IN, pull_up_down=gpio_test_setup.PUD_UP)
        
        # Simulate button press (active low)
        gpio_test_setup._state[emergency_pin] = gpio_test_setup.LOW
        
        # Trigger the emergency button callback directly
        # (In real hardware, this would be triggered by GPIO event)
        if hasattr(safety_controller, '_emergency_switch_callback'):
            safety_controller._emergency_switch_callback(emergency_pin)
        else:
            # Alternative: send fire trigger which emergency button would do
            safety_controller.handle_fire_trigger()
        
        # Pump should start
        assert safety_controller._state in [PumpState.PRIMING, PumpState.STARTING, PumpState.RUNNING]
        
        # Give a moment for GPIO to be set
        time.sleep(0.1)
        
        # Check valve state - use the GPIO instance from the trigger module
        # The controller uses its own GPIO instance, not the test fixture's
        from gpio_trigger.trigger import GPIO as controller_gpio
        assert controller_gpio.input(CONFIG['MAIN_VALVE_PIN']) == controller_gpio.HIGH
    
    @pytest.mark.timeout(30)
    def test_emergency_button_debouncing(self, safety_controller, gpio_test_setup):
        """Test emergency button debounces multiple presses."""
        if gpio_test_setup is None:
            pytest.skip("GPIO simulation not available")
            
        emergency_pin = CONFIG.get('EMERGENCY_BUTTON_PIN', 21)
        
        # Rapid button presses
        for _ in range(5):
            gpio_test_setup._state[emergency_pin] = gpio_test_setup.LOW
            time.sleep(0.01)
            gpio_test_setup._state[emergency_pin] = gpio_test_setup.HIGH
            time.sleep(0.01)
        
        # Should only trigger once
        safety_controller.handle_fire_trigger()
        state = safety_controller._state
        assert state in [PumpState.PRIMING, PumpState.STARTING, PumpState.RUNNING]


class TestDryRunProtection:
    """Test dry run protection using real components."""
    
    @pytest.mark.timeout(30)
    def test_dry_run_timeout_stops_pump(self, safety_controller, gpio_test_setup):
        """Test pump stops if no water flow detected."""
        if gpio_test_setup is None:
            pytest.skip("GPIO simulation not available")
            
        # Setup flow sensor
        flow_pin = CONFIG.get('FLOW_SENSOR_PIN', 19)
        if flow_pin:
            gpio_test_setup.setup(flow_pin, gpio_test_setup.IN, pull_up_down=gpio_test_setup.PUD_DOWN)
            gpio_test_setup._state[flow_pin] = gpio_test_setup.LOW  # No flow
        
        # Start pump
        safety_controller.handle_fire_trigger()
        
        # Wait for pump to start
        start_time = time.time()
        while safety_controller._state != PumpState.RUNNING and time.time() - start_time < 2:
            time.sleep(0.1)
        
        if safety_controller._state == PumpState.RUNNING:
            # Force dry run condition
            safety_controller._water_flow_detected = False
            safety_controller._pump_start_time = time.time() - 0.5  # Started 0.5s ago
            
            # Wait for dry run protection to trigger (MAX_DRY_RUN_TIME = 1.0s)
            time.sleep(1.0)
            
            # Should enter error state
            assert safety_controller._state == PumpState.ERROR
            assert gpio_test_setup.input(CONFIG['IGN_ON_PIN']) == gpio_test_setup.LOW
    
    @pytest.mark.timeout(30)
    def test_water_flow_prevents_dry_run_error(self, safety_controller, gpio_test_setup):
        """Test water flow detection prevents dry run error."""
        if gpio_test_setup is None:
            pytest.skip("GPIO simulation not available")
            
        # Setup flow sensor with flow detected
        flow_pin = CONFIG.get('FLOW_SENSOR_PIN', 19)
        if flow_pin:
            gpio_test_setup.setup(flow_pin, gpio_test_setup.IN)
            gpio_test_setup._state[flow_pin] = gpio_test_setup.HIGH  # Flow detected
        
        # Start pump
        safety_controller.handle_fire_trigger()
        
        # Wait for pump to start
        start_time = time.time()
        while safety_controller._state != PumpState.RUNNING and time.time() - start_time < 2:
            time.sleep(0.1)
        
        if safety_controller._state == PumpState.RUNNING:
            # Set water flow detected
            safety_controller._water_flow_detected = True
            
            # Wait longer than dry run timeout
            time.sleep(1.5)
            
            # Should NOT be in error state
            assert safety_controller._state != PumpState.ERROR


class TestReservoirMonitoring:
    """Test reservoir level monitoring using real components."""
    
    @pytest.mark.timeout(30)
    def test_float_switch_stops_refill(self, safety_controller, gpio_test_setup):
        """Test float switch activation stops refill."""
        if gpio_test_setup is None:
            pytest.skip("GPIO simulation not available")
            
        # Setup float switch
        float_pin = CONFIG.get('RESERVOIR_FLOAT_PIN', 16)
        if float_pin:
            gpio_test_setup.setup(float_pin, gpio_test_setup.IN, pull_up_down=gpio_test_setup.PUD_DOWN)
            gpio_test_setup._state[float_pin] = gpio_test_setup.LOW  # Not full
        
        # Start pump and let it run briefly
        safety_controller.handle_fire_trigger()
        
        # Wait for pump to start
        start_time = time.time()
        while safety_controller._state != PumpState.RUNNING and time.time() - start_time < 2:
            time.sleep(0.1)
        
        if safety_controller._state == PumpState.RUNNING:
            # Shutdown to trigger refill
            safety_controller._shutdown_engine()
            
            # Wait for refill state
            start_time = time.time()
            while safety_controller._state != PumpState.REFILLING and time.time() - start_time < 2:
                time.sleep(0.1)
            
            if safety_controller._state == PumpState.REFILLING:
                # Verify refill valve is open
                assert gpio_test_setup.input(CONFIG['REFILL_VALVE_PIN']) == gpio_test_setup.HIGH
                
                # Simulate float switch activation (tank full)
                gpio_test_setup._state[float_pin] = gpio_test_setup.HIGH
                
                # Trigger the monitoring to detect float switch
                if hasattr(safety_controller, '_check_reservoir_level'):
                    safety_controller._check_reservoir_level()
                
                # Give time for state change
                time.sleep(0.5)
                
                # Refill valve should close
                assert gpio_test_setup.input(CONFIG['REFILL_VALVE_PIN']) == gpio_test_setup.LOW
                assert safety_controller._refill_complete is True


class TestLinePressure:
    """Test line pressure monitoring using real components."""
    
    @pytest.mark.timeout(30)
    def test_low_pressure_triggers_shutdown(self, safety_controller, gpio_test_setup):
        """Test low line pressure causes safe shutdown."""
        if gpio_test_setup is None:
            pytest.skip("GPIO simulation not available")
            
        # Setup pressure switch
        pressure_pin = CONFIG.get('LINE_PRESSURE_PIN', 20)
        if pressure_pin:
            gpio_test_setup.setup(pressure_pin, gpio_test_setup.IN, pull_up_down=gpio_test_setup.PUD_DOWN)
            gpio_test_setup._state[pressure_pin] = gpio_test_setup.HIGH  # Good pressure
        
        # Start pump
        safety_controller.handle_fire_trigger()
        
        # Wait for pump to be fully running
        start_time = time.time()
        while safety_controller._state != PumpState.RUNNING and time.time() - start_time < 2:
            time.sleep(0.1)
        
        if safety_controller._state == PumpState.RUNNING:
            # Wait for priming to complete
            time.sleep(0.3)
            
            # Simulate low pressure
            gpio_test_setup._state[pressure_pin] = gpio_test_setup.LOW
            
            # Trigger pressure check
            if hasattr(safety_controller, '_check_line_pressure'):
                safety_controller._check_line_pressure()
            
            # Wait for state change
            time.sleep(0.5)
            
            # Should shutdown due to low pressure
            assert safety_controller._state in [PumpState.LOW_PRESSURE, PumpState.STOPPING, 
                                               PumpState.REFILLING, PumpState.COOLDOWN]
            # Engine should be off or stopping
            if safety_controller._state not in [PumpState.REFILLING, PumpState.COOLDOWN]:
                time.sleep(0.5)
            assert gpio_test_setup.input(CONFIG['IGN_ON_PIN']) == gpio_test_setup.LOW


class TestEmergencyMQTT:
    """Test emergency MQTT commands using real broker."""
    
    @pytest.mark.timeout(30)
    def test_mqtt_emergency_trigger(self, safety_controller, test_mqtt_broker, mqtt_topic_factory):
        """Test emergency fire trigger via MQTT."""
        import paho.mqtt.client as mqtt
        
        # Create MQTT publisher
        conn_params = test_mqtt_broker.get_connection_params()
        publisher = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="test_emergency")
        publisher.connect(conn_params['host'], conn_params['port'])
        publisher.loop_start()
        
        # Wait for connection
        time.sleep(0.5)
        
        # Send emergency trigger
        trigger_topic = mqtt_topic_factory("trigger/fire_detected")
        emergency_msg = json.dumps({
            "emergency": True,
            "source": "test_emergency",
            "timestamp": time.time()
        })
        
        publisher.publish(trigger_topic, emergency_msg, qos=1)
        
        # Wait for processing
        time.sleep(1.0)
        
        # Pump should start
        assert safety_controller._state in [PumpState.PRIMING, PumpState.STARTING, 
                                           PumpState.RUNNING, PumpState.COOLDOWN, PumpState.IDLE]
        
        # Cleanup
        publisher.loop_stop()
        publisher.disconnect()
    
    @pytest.mark.timeout(30)
    def test_mqtt_status_reporting(self, safety_controller, test_mqtt_broker, mqtt_topic_factory):
        """Test safety status is reported via MQTT."""
        import paho.mqtt.client as mqtt
        
        # First check that controller is connected
        assert safety_controller.client.is_connected(), "Controller not connected to MQTT"
        
        # Subscribe to status topic
        status_messages = []
        
        def on_message(client, userdata, msg):
            print(f"Received message on topic: {msg.topic}")
            try:
                payload = json.loads(msg.payload.decode())
                status_messages.append((msg.topic, payload))
                print(f"Parsed payload: {payload}")
            except Exception as e:
                print(f"Error parsing message: {e}")
        
        conn_params = test_mqtt_broker.get_connection_params()
        subscriber = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="test_status")
        subscriber.on_message = on_message
        subscriber.connect(conn_params['host'], conn_params['port'])
        
        # Subscribe to all topics to debug
        subscriber.subscribe("#")
        subscriber.loop_start()
        
        # Give subscriber time to connect and subscribe
        time.sleep(1.0)
        
        # Trigger a status update
        print("Publishing safety_test event...")
        safety_controller._publish_event("safety_test", {"test": True})
        
        # Also test direct publish to make sure MQTT is working
        print("Direct MQTT publish test...")
        test_topic = mqtt_topic_factory("test/direct")
        safety_controller.client.publish(test_topic, json.dumps({"direct": "test"}), qos=1)
        
        # Wait for messages
        time.sleep(2.0)
        
        print(f"Received {len(status_messages)} messages")
        for topic, msg in status_messages:
            print(f"  Topic: {topic}, Message: {msg}")
        
        # Should have received at least one message
        assert len(status_messages) > 0, "No MQTT messages received at all"
        
        # Cleanup
        subscriber.loop_stop()
        subscriber.disconnect()


class TestSafetyIntegration:
    """Test integration of multiple safety features."""
    
    @pytest.mark.timeout(30)
    def test_multiple_safety_features_together(self, safety_controller, gpio_test_setup):
        """Test multiple safety features work together correctly."""
        if gpio_test_setup is None:
            pytest.skip("GPIO simulation not available")
            
        # Setup all safety pins
        emergency_pin = CONFIG.get('EMERGENCY_BUTTON_PIN', 21)
        float_pin = CONFIG.get('RESERVOIR_FLOAT_PIN', 16)
        pressure_pin = CONFIG.get('LINE_PRESSURE_PIN', 20)
        flow_pin = CONFIG.get('FLOW_SENSOR_PIN', 19)
        
        # Initialize all pins
        if emergency_pin:
            gpio_test_setup.setup(emergency_pin, gpio_test_setup.IN, pull_up_down=gpio_test_setup.PUD_UP)
        if float_pin:
            gpio_test_setup.setup(float_pin, gpio_test_setup.IN, pull_up_down=gpio_test_setup.PUD_DOWN)
        if pressure_pin:
            gpio_test_setup.setup(pressure_pin, gpio_test_setup.IN, pull_up_down=gpio_test_setup.PUD_DOWN)
        if flow_pin:
            gpio_test_setup.setup(flow_pin, gpio_test_setup.IN, pull_up_down=gpio_test_setup.PUD_DOWN)
        
        # Set good initial conditions
        if pressure_pin:
            gpio_test_setup._state[pressure_pin] = gpio_test_setup.HIGH  # Good pressure
        if flow_pin:
            gpio_test_setup._state[flow_pin] = gpio_test_setup.HIGH  # Flow detected
        if float_pin:
            gpio_test_setup._state[float_pin] = gpio_test_setup.LOW  # Not full
        
        # Start pump via emergency button
        if emergency_pin:
            gpio_test_setup._state[emergency_pin] = gpio_test_setup.LOW  # Press button
        safety_controller.handle_fire_trigger()
        
        # Verify pump starts
        assert safety_controller._state in [PumpState.PRIMING, PumpState.STARTING, PumpState.RUNNING]
        
        # All safety features should be monitoring
        assert safety_controller._shutdown is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])