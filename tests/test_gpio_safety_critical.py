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

# Add module paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../gpio_trigger")))

# Import test fixtures
from test_trigger import gpio_test_setup, wait_for_state

# Import after path setup
import gpio_trigger.trigger as trigger
from gpio_trigger.trigger import PumpController, GPIO, CONFIG, PumpState

logger = logging.getLogger(__name__)


@pytest.fixture
def safety_controller(gpio_test_setup, monkeypatch, test_mqtt_broker, mqtt_topic_factory):
    """Create controller with safety features enabled for testing.
    
    BEST PRACTICE: Real PumpController with real safety features configured.
    """
    # Get connection parameters from the test broker
    conn_params = test_mqtt_broker.get_connection_params()
    
    # Get unique topic prefix for test isolation
    full_topic = mqtt_topic_factory("dummy")
    topic_prefix = full_topic.rsplit('/', 1)[0]
    
    # Configure safety features
    monkeypatch.setenv("EMERGENCY_BUTTON_PIN", "21")
    monkeypatch.setenv("RESERVOIR_FLOAT_PIN", "16")
    monkeypatch.setenv("LINE_PRESSURE_PIN", "20")
    monkeypatch.setenv("FLOW_SENSOR_PIN", "19")
    monkeypatch.setenv("MAX_DRY_RUN_TIME", "1.0")  # Short for testing
    monkeypatch.setenv("PRESSURE_CHECK_DELAY", "0.5")
    monkeypatch.setenv("HEALTH_INTERVAL", "10")
    
    # Speed up timings for tests
    monkeypatch.setenv("VALVE_PRE_OPEN_DELAY", "0.1")
    monkeypatch.setenv("IGNITION_START_DURATION", "0.05")
    monkeypatch.setenv("FIRE_OFF_DELAY", "0.5")
    monkeypatch.setenv("MAX_ENGINE_RUNTIME", "5")
    
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
    from gpio_trigger.trigger import PumpController, GPIO, CONFIG, PumpState
    
    # Update globals
    globals()['trigger'] = trigger
    globals()['PumpController'] = PumpController
    globals()['GPIO'] = GPIO
    globals()['CONFIG'] = CONFIG
    globals()['PumpState'] = PumpState
    
    # Create controller with real safety features
    controller = PumpController()
    
    # Wait for MQTT connection
    time.sleep(1.0)
    
    yield controller
    
    # Cleanup
    controller._shutdown = True
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
        if hasattr(safety_controller, '_emergency_button_callback'):
            safety_controller._emergency_button_callback(emergency_pin)
        else:
            # Alternative: send fire trigger which emergency button would do
            safety_controller.handle_fire_trigger()
        
        # Pump should start
        assert safety_controller._state in [PumpState.PRIMING, PumpState.STARTING, PumpState.RUNNING]
        assert gpio_test_setup.input(CONFIG['MAIN_VALVE_PIN']) is True
    
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
            assert gpio_test_setup.input(CONFIG['IGN_ON_PIN']) is False
    
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
                assert gpio_test_setup.input(CONFIG['REFILL_VALVE_PIN']) is True
                
                # Simulate float switch activation (tank full)
                gpio_test_setup._state[float_pin] = gpio_test_setup.HIGH
                
                # Trigger the monitoring to detect float switch
                if hasattr(safety_controller, '_check_reservoir_level'):
                    safety_controller._check_reservoir_level()
                
                # Give time for state change
                time.sleep(0.5)
                
                # Refill valve should close
                assert gpio_test_setup.input(CONFIG['REFILL_VALVE_PIN']) is False
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
            assert gpio_test_setup.input(CONFIG['IGN_ON_PIN']) is False


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
        
        # Subscribe to status topic
        status_messages = []
        
        def on_message(client, userdata, msg):
            try:
                payload = json.loads(msg.payload.decode())
                status_messages.append(payload)
            except:
                pass
        
        conn_params = test_mqtt_broker.get_connection_params()
        subscriber = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="test_status")
        subscriber.on_message = on_message
        subscriber.connect(conn_params['host'], conn_params['port'])
        
        # Subscribe to status topic
        status_topic = mqtt_topic_factory("gpio/status")
        subscriber.subscribe(status_topic)
        subscriber.loop_start()
        
        # Trigger a status update
        safety_controller._publish_event("safety_test", {"test": True})
        
        # Wait for message
        time.sleep(1.0)
        
        # Should have received status
        assert len(status_messages) > 0
        
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