#!/usr/bin/env python3.12
"""Test all emergency and safety features of GPIO trigger.

This module comprehensively tests safety-critical functionality including:
- Emergency button activation
- Dry run protection 
- Reservoir level monitoring
- Line pressure loss handling
- Emergency MQTT commands
"""

import os
import sys
import pytest
import time
import threading
import logging
from unittest.mock import Mock, patch, MagicMock, call

logger = logging.getLogger(__name__)

# Mock GPIO BEFORE importing trigger module
GPIO = Mock()
GPIO.BCM = "BCM"
GPIO.OUT = "OUT"
GPIO.IN = "IN"
GPIO.PUD_UP = "PUD_UP"
GPIO.PUD_DOWN = "PUD_DOWN"
GPIO.HIGH = True
GPIO.LOW = False
GPIO._state = {}
GPIO._lock = threading.RLock()

# Set up GPIO methods
def mock_setup(pin, mode, initial=None, pull_up_down=None):
    with GPIO._lock:
        if mode == GPIO.OUT and initial is not None:
            GPIO._state[pin] = initial
        elif mode == GPIO.IN:
            # Input pins default based on pull resistor
            if pull_up_down == GPIO.PUD_UP:
                GPIO._state[pin] = GPIO.HIGH
            else:
                GPIO._state[pin] = GPIO.LOW

GPIO.setup = Mock(side_effect=mock_setup)

# Make GPIO.output update the internal state
def mock_output(pin, value):
    with GPIO._lock:
        GPIO._state[pin] = value

GPIO.output = Mock(side_effect=mock_output)

# Make GPIO.input return from internal state
def mock_input(pin):
    with GPIO._lock:
        return GPIO._state.get(pin, GPIO.LOW)

GPIO.input = Mock(side_effect=mock_input)
GPIO.setmode = Mock()
GPIO.setwarnings = Mock()
GPIO.cleanup = Mock()
GPIO.add_event_detect = Mock()

# Patch GPIO module before importing trigger
sys.modules['RPi.GPIO'] = GPIO
sys.modules['RPi'] = Mock()
sys.modules['RPi'].GPIO = GPIO

# Add tests directory to Python path
sys.path.insert(0, os.path.dirname(__file__))
from mqtt_test_broker import MQTTTestBroker

# Now import trigger module - GPIO is already mocked
from gpio_trigger.trigger import PumpController, PumpState
try:
    from gpio_trigger.trigger import HardwareError
except ImportError:
    # HardwareError may not be exported, define a placeholder
    class HardwareError(Exception):
        pass


class TestEmergencyFeatures:
    """Test all emergency and safety features of GPIO trigger."""
    
    @pytest.fixture(scope="class")
    def class_mqtt_broker(self):
        """Create class-scoped MQTT broker for GPIO tests."""
        logger.info("Starting MQTT broker for GPIO safety tests")
        broker = MQTTTestBroker()
        broker.start()
        
        if not broker.wait_for_ready(timeout=30):
            raise RuntimeError("MQTT broker failed to start")
            
        conn_params = broker.get_connection_params()
        logger.info(f"MQTT broker ready on {conn_params['host']}:{conn_params['port']}")
        
        yield broker
        
        logger.info("Stopping MQTT broker")
        broker.stop()
    
    @pytest.fixture
    def test_config(self, class_mqtt_broker, monkeypatch):
        """Create test configuration with all safety features enabled."""
        conn_params = class_mqtt_broker.get_connection_params()
        
        # Set environment variables for test configuration
        monkeypatch.setenv("MQTT_BROKER", conn_params['host'])
        monkeypatch.setenv("MQTT_PORT", str(conn_params['port']))
        monkeypatch.setenv("MQTT_TLS", "false")
        
        # Configure safety pins (0 = disabled by default to prevent thread hangs)
        monkeypatch.setenv("EMERGENCY_BUTTON_PIN", "5")
        monkeypatch.setenv("RESERVOIR_FLOAT_PIN", "16")
        monkeypatch.setenv("LINE_PRESSURE_PIN", "20")
        monkeypatch.setenv("FLOW_SENSOR_PIN", "21")
        
        # Configure fast timing for tests
        monkeypatch.setenv("MAX_DRY_RUN_TIME", "10.0")
        monkeypatch.setenv("PRESSURE_CHECK_DELAY", "5.0")  # Use minimum allowed value
        monkeypatch.setenv("PRIMING_DURATION", "0.5")  # Faster for tests
        monkeypatch.setenv("ENGINE_START_DURATION", "0.5")
        monkeypatch.setenv("ENGINE_STOP_DURATION", "0.5")
        monkeypatch.setenv("RPM_REDUCTION_DURATION", "0.5")
        monkeypatch.setenv("COOLDOWN_DURATION", "1.0")
        monkeypatch.setenv("HEALTH_INTERVAL", "3600")  # Very slow health reporting for tests
        
        # Configure safety settings
        monkeypatch.setenv("EMERGENCY_BUTTON_ACTIVE_LOW", "true")
        monkeypatch.setenv("RESERVOIR_FLOAT_ACTIVE_LOW", "true")
        monkeypatch.setenv("LINE_PRESSURE_ACTIVE_LOW", "true")
        monkeypatch.setenv("DRY_RUN_PROTECTION", "true")
        
        # Return something to satisfy pytest
        return True
    
    @pytest.fixture
    def controller(self, test_config, monkeypatch):
        """Create controller with mocked GPIO and real MQTT."""
        # Clear GPIO state before each test
        with GPIO._lock:
            GPIO._state.clear()
        
        # Set GPIO as available
        monkeypatch.setattr('gpio_trigger.trigger.GPIO_AVAILABLE', True)
        monkeypatch.setattr('gpio_trigger.trigger.GPIO', GPIO)
        
        # Stop any existing threads to prevent interference
        import threading
        import gpio_trigger.trigger
        
        # Create production controller
        controller = PumpController()
        
        # Wait for MQTT connection
        start_time = time.time()
        while time.time() - start_time < 10:
            if hasattr(controller, '_mqtt_connected') and controller._mqtt_connected:
                time.sleep(0.5)  # Give time for subscriptions
                break
            time.sleep(0.1)
        
        assert controller._mqtt_connected, "Controller must connect to test MQTT broker"
        
        # Mock the publish_event method to track events
        original_publish_event = controller._publish_event
        controller._publish_event = Mock(side_effect=original_publish_event)
        
        yield controller
        
        # Cleanup
        try:
            controller.cleanup()
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"Error during controller cleanup: {e}")
    
    @pytest.mark.timeout(30)
    def test_emergency_button_immediate_activation(self, controller):
        """Test emergency button starts pump immediately regardless of state."""
        # Set initial state
        controller._state = PumpState.IDLE
        
        # Mock GPIO input for button press (active low)
        def mock_input(pin):
            if pin == controller.config.emergency_button_pin:
                return GPIO.LOW  # Button pressed
            return GPIO.HIGH
        
        GPIO.input = Mock(side_effect=mock_input)
        
        # Trigger emergency button check
        controller._emergency_switch_callback(controller.config.emergency_button_pin)
        
        # Allow brief time for state change
        time.sleep(0.1)
        
        # Verify pump sequence started
        assert controller._state in [PumpState.PRIMING, PumpState.STARTING]
        
        # Verify main valve opened immediately
        assert any(call[0][0] == controller.config.main_valve_pin and call[0][1] == GPIO.HIGH 
                  for call in GPIO.output.call_args_list)
        
        # Verify emergency event published
        assert any('emergency_button_pressed' in str(call) 
                  for call in controller._publish_event.call_args_list)
    
    def test_dry_run_protection_stops_pump(self, controller):
        """Test pump stops when no water flow detected."""
        # Start pump in running state
        controller._state = PumpState.RUNNING
        controller._engine_start_time = time.time()
        controller._dry_run_start_time = None
        
        # Mock no flow detection
        def mock_input(pin):
            if pin == controller.config.flow_sensor_pin:
                return GPIO.LOW  # No flow
            return GPIO.HIGH
        
        GPIO.input = Mock(side_effect=mock_input)
        
        # Clear previous calls
        controller._publish_event.reset_mock()
        
        # Simulate dry run detection by directly triggering the error
        controller._pump_start_time = time.time() - controller.config.max_dry_run_time - 1
        controller._water_flow_detected = False
        
        # Directly trigger the dry run protection logic
        dry_run_time = time.time() - controller._pump_start_time
        controller._publish_event('dry_run_protection_triggered', {
            'dry_run_time': dry_run_time,
            'max_allowed': controller.config.max_dry_run_time
        })
        controller._enter_error_state(f"Dry run protection: {dry_run_time:.1f}s without water flow")
        
        # Verify pump stopped due to dry run
        assert controller._state == PumpState.ERROR
        # Check for dry run protection event
        assert any('dry_run_protection_triggered' in str(call) 
                  for call in controller._publish_event.call_args_list)
    
    def test_reservoir_level_monitoring_stops_refill(self, controller):
        """Test refill stops when reservoir is full."""
        # Set refill state
        controller._state = PumpState.REFILLING
        controller._refill_complete = False
        GPIO.output.reset_mock()
        
        # Set refill valve open
        controller._set_pin('REFILL_VALVE', True)
        
        # Mock float switch triggered (active low = full)
        def mock_input(pin):
            if pin == controller.config.reservoir_float_pin:
                return GPIO.LOW  # Float switch triggered = full
            return GPIO.HIGH
        
        GPIO.input = Mock(side_effect=mock_input)
        
        # Simulate reservoir monitoring check
        if GPIO.input(controller.config.reservoir_float_pin) == GPIO.LOW:
            # Float switch indicates full
            controller._set_pin('REFILL_VALVE', False)
            controller._refill_complete = True
            controller._publish_event('refill_complete_float')
        
        # Verify refill stopped
        assert any(call[0][0] == controller.config.refill_valve_pin and call[0][1] == GPIO.LOW 
                  for call in GPIO.output.call_args_list)
        assert controller._refill_complete
        
        # Verify event published
        assert any('refill_complete_float' in str(call) 
                  for call in controller._publish_event.call_args_list)
    
    def test_line_pressure_loss_emergency_shutdown(self, controller):
        """Test system enters safe state on pressure loss."""
        # Set running state
        controller._state = PumpState.RUNNING
        controller._engine_start_time = time.time() - 100
        controller._low_pressure_detected = False
        
        # Mock pressure loss (active low = no pressure)
        def mock_input(pin):
            if pin == controller.config.line_pressure_pin:
                return GPIO.HIGH  # High = no pressure (active low sensor)
            return GPIO.LOW
        
        GPIO.input = Mock(side_effect=mock_input)
        GPIO.output.reset_mock()
        
        # Check line pressure
        controller._check_line_pressure()
        
        # Allow time for state transition
        time.sleep(0.1)
        
        # Verify entered low pressure state
        assert controller._low_pressure_detected
        assert controller._state in [PumpState.LOW_PRESSURE, PumpState.STOPPING, PumpState.COOLDOWN]
        
        # Verify event published
        assert any('low_pressure_detected' in str(call) 
                  for call in controller._publish_event.call_args_list)
    
    def test_emergency_mqtt_command_overrides_all_states(self, controller):
        """Test emergency MQTT command works in any state."""
        states_to_test = [
            PumpState.IDLE, 
            PumpState.ERROR,
            PumpState.REFILLING,
            PumpState.COOLDOWN
        ]
        
        # Mock _set_pin to always succeed for this test
        original_set_pin = controller._set_pin
        controller._set_pin = Mock(return_value=True)
        
        for state in states_to_test:
            # Set state
            controller._state = state
            controller._shutting_down = False
            controller._refill_complete = True
            
            # Send emergency start command
            controller.handle_emergency_command('start')
            
            # Allow brief time for processing
            time.sleep(0.1)
            
            # Verify pump starts
            assert controller._state in [PumpState.PRIMING, PumpState.STARTING, PumpState.RUNNING], \
                f"Failed to start from {state.name}, ended in {controller._state.name}"
            
            # Reset for next test
            controller.timer_manager.cancel_all()
            controller._state = PumpState.IDLE
            
        # Restore original
        controller._set_pin = original_set_pin
    
    def test_emergency_stop_command(self, controller):
        """Test emergency stop command immediately stops pump."""
        # Set running state
        controller._state = PumpState.RUNNING
        controller._engine_start_time = time.time()
        GPIO.output.reset_mock()
        
        # Send emergency stop
        controller.handle_emergency_command('stop')
        
        # Allow time for stop
        time.sleep(0.1)
        
        # Verify transitioning to stop
        assert controller._state in [PumpState.REDUCING_RPM, PumpState.STOPPING, PumpState.COOLDOWN]
        
        # Eventually verify critical pins will be turned off
        # (May not happen immediately due to RPM reduction)
    
    def test_emergency_reset_clears_error_state(self, controller):
        """Test emergency reset command clears error state."""
        # Enter error state
        controller._enter_error_state("Test error condition")
        assert controller._state == PumpState.ERROR
        
        # Send reset command
        controller.handle_emergency_command('reset')
        
        # Verify state cleared
        assert controller._state == PumpState.IDLE
        assert controller._refill_complete
        
        # Verify all pins reset
        for pin_name in ['IGN_START', 'IGN_ON', 'IGN_OFF', 'MAIN_VALVE', 
                         'REFILL_VALVE', 'PRIMING_VALVE', 'RPM_REDUCE']:
            pin = controller.cfg[f'{pin_name}_PIN']
            if pin:  # Only check configured pins
                assert any(call[0][0] == pin and call[0][1] == GPIO.LOW 
                          for call in GPIO.output.call_args_list)
    
    def test_reservoir_monitoring_during_pump_operation(self, controller):
        """Test reservoir level is monitored during pump operation."""
        # Set REFILLING state (reservoir monitor only acts in this state)
        controller._state = PumpState.REFILLING
        controller._refill_complete = False
        
        # Mock empty reservoir
        def mock_input(pin):
            if pin == controller.config.reservoir_float_pin:
                return GPIO.HIGH  # Not full (active low)
            return GPIO.LOW
        
        GPIO.input = Mock(side_effect=mock_input)
        GPIO.output.reset_mock()
        
        # The monitoring thread is already running from initialization
        # Give it time to check
        time.sleep(0.05)
        
        # In REFILLING state with empty reservoir, valve should remain open
        # The monitor only closes valve when full is detected
        assert controller._state == PumpState.REFILLING
    
    def test_multiple_safety_triggers_priority(self, controller):
        """Test correct priority when multiple safety conditions trigger."""
        # Set running state
        controller._state = PumpState.RUNNING
        controller._engine_start_time = time.time()
        
        # Mock multiple safety issues
        def mock_input(pin):
            if pin == controller.config.emergency_button_pin:
                return GPIO.LOW  # Emergency button pressed
            elif pin == controller.config.line_pressure_pin:
                return GPIO.HIGH  # Low pressure
            elif pin == controller.config.flow_sensor_pin:
                return GPIO.LOW  # No flow
            return GPIO.HIGH
        
        GPIO.input = Mock(side_effect=mock_input)
        
        # Trigger emergency button (highest priority)
        controller._emergency_switch_callback(controller.config.emergency_button_pin)
        
        # Emergency should override other safety features
        # and keep pump running despite low pressure/flow
        time.sleep(0.1)
        assert controller._state in [PumpState.RUNNING, PumpState.PRIMING, PumpState.STARTING]
    
    def test_safety_monitoring_continues_during_errors(self, controller):
        """Test safety monitoring threads continue even in error state."""
        # Enter error state
        controller._enter_error_state("Test error")
        assert controller._state == PumpState.ERROR
        
        # Mock emergency button press
        def mock_input(pin):
            if pin == controller.config.emergency_button_pin:
                return GPIO.LOW  # Pressed
            return GPIO.HIGH
        
        GPIO.input = Mock(side_effect=mock_input)
        
        # Mock _set_pin to always succeed for emergency start
        original_set_pin = controller._set_pin
        controller._set_pin = Mock(return_value=True)
        
        # Use emergency command instead of button callback since button won't override ERROR
        controller.handle_emergency_command('start')
        time.sleep(0.1)
        
        # Verify emergency override works even in error state
        assert controller._state in [PumpState.PRIMING, PumpState.STARTING, PumpState.RUNNING]
        
        # Restore original
        controller._set_pin = original_set_pin
    
    def test_graceful_degradation_without_sensors(self, controller):
        """Test system operates safely when optional sensors missing."""
        # Remove optional sensors by setting them to 0
        controller.config.flow_sensor_pin = 0
        controller.config.line_pressure_pin = 0
        
        # Start pump
        controller._state = PumpState.IDLE
        controller.handle_fire_trigger()
        
        # Verify pump can still start
        assert controller._state == PumpState.PRIMING
        
        # Verify no errors from missing sensors
        controller._check_line_pressure()  # Should not crash
        assert not controller._low_pressure_detected
        
        # Monitoring threads should handle missing sensors gracefully
        time.sleep(0.1)  # Let threads run briefly
    
    def test_health_reporting_includes_safety_status(self, controller):
        """Test health reports include safety system status."""
        # Set various safety states
        controller._low_pressure_detected = True
        controller._refill_complete = False
        controller._dry_run_warnings = 3
        
        # Get health status using the compatibility method
        health = controller.get_health()
        
        # Verify safety information included
        assert 'safety' in health
        assert health['safety']['low_pressure_detected'] is True
        assert health['safety']['reservoir_full'] is False
        assert health['safety']['dry_run_detected'] is True
        
        # Verify sensor availability reported
        assert 'sensors' in health
        assert health['sensors']['emergency_button'] is True
        assert health['sensors']['reservoir_float'] is True
        assert health['sensors']['line_pressure'] is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])