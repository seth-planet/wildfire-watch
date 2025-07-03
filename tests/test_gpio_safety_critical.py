#!/usr/bin/env python3.12
"""Test all emergency and safety features of GPIO trigger.

This module comprehensively tests safety-critical functionality including:
- Emergency button activation
- Dry run protection 
- Reservoir level monitoring
- Line pressure loss handling
- Emergency MQTT commands
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock, call
from gpio_trigger.trigger import PumpController, PumpState, HardwareError, CONFIG

# Mock GPIO for tests
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


class TestEmergencyFeatures:
    """Test all emergency and safety features of GPIO trigger."""
    
    @pytest.fixture
    def test_config(self):
        """Create test configuration with all safety features enabled."""
        config = CONFIG.copy()
        config.update({
            # Safety pins
            'EMERGENCY_BUTTON_PIN': 5,
            'RESERVOIR_FLOAT_PIN': 16,
            'LINE_PRESSURE_PIN': 20,
            'FLOW_SENSOR_PIN': 21,
            
            # Timing
            'MAX_DRY_RUN_TIME': 10.0,  # Fast for testing
            'PRESSURE_CHECK_DELAY': 2.0,
            'PRIMING_DURATION': 3.0,
            
            # Safety settings
            'EMERGENCY_BUTTON_ACTIVE_LOW': True,
            'RESERVOIR_FLOAT_ACTIVE_LOW': True,
            'LINE_PRESSURE_ACTIVE_LOW': True,
            'DRY_RUN_PROTECTION': True,
            
            # MQTT
            'MQTT_BROKER': 'localhost',
            'MQTT_PORT': 1883,
        })
        return config
    
    @pytest.fixture
    def controller(self, test_config):
        """Create controller with mocked GPIO and MQTT."""
        with patch('gpio_trigger.trigger.GPIO', GPIO):
            with patch('gpio_trigger.trigger.mqtt.Client'):
                controller = PumpController(test_config)
                # Mock MQTT
                controller._mqtt_client = Mock()
                controller._mqtt_connected = True
                controller._publish_message = Mock(return_value=True)
                # Start monitoring threads
                controller._start_monitoring_tasks()
                yield controller
                # Cleanup
                controller.shutdown()
    
    def test_emergency_button_immediate_activation(self, controller):
        """Test emergency button starts pump immediately regardless of state."""
        # Set initial state
        controller._state = PumpState.IDLE
        
        # Mock GPIO input for button press (active low)
        def mock_input(pin):
            if pin == controller.cfg['EMERGENCY_BUTTON_PIN']:
                return GPIO.LOW  # Button pressed
            return GPIO.HIGH
        
        GPIO.input = Mock(side_effect=mock_input)
        
        # Trigger emergency button check
        controller._emergency_switch_callback(controller.cfg['EMERGENCY_BUTTON_PIN'])
        
        # Allow brief time for state change
        time.sleep(0.1)
        
        # Verify pump sequence started
        assert controller._state in [PumpState.PRIMING, PumpState.STARTING]
        
        # Verify main valve opened immediately
        assert any(call[0][0] == controller.cfg['MAIN_VALVE_PIN'] and call[0][1] == GPIO.HIGH 
                  for call in GPIO.output.call_args_list)
        
        # Verify emergency event published
        assert any('emergency_button_activated' in str(call) 
                  for call in controller._publish_message.call_args_list)
    
    def test_dry_run_protection_stops_pump(self, controller):
        """Test pump stops when no water flow detected."""
        # Start pump in running state
        controller._state = PumpState.RUNNING
        controller._engine_start_time = time.time()
        controller._dry_run_start_time = None
        
        # Mock no flow detection
        def mock_input(pin):
            if pin == controller.cfg['FLOW_SENSOR_PIN']:
                return GPIO.LOW  # No flow
            return GPIO.HIGH
        
        GPIO.input = Mock(side_effect=mock_input)
        
        # Clear previous calls
        controller._publish_message.reset_mock()
        
        # Simulate dry run detection over time
        for _ in range(int(controller.cfg['MAX_DRY_RUN_TIME']) + 2):
            controller._monitor_dry_run_protection()
            time.sleep(1)
            
            # Check if error state entered
            if controller._state == PumpState.ERROR:
                break
        
        # Verify pump stopped due to dry run
        assert controller._state == PumpState.ERROR
        assert any('dry_run_detected' in str(call) 
                  for call in controller._publish_message.call_args_list)
    
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
            if pin == controller.cfg['RESERVOIR_FLOAT_PIN']:
                return GPIO.LOW  # Float switch triggered = full
            return GPIO.HIGH
        
        GPIO.input = Mock(side_effect=mock_input)
        
        # Run reservoir monitoring
        controller._monitor_reservoir_level()
        time.sleep(0.1)
        
        # Verify refill stopped
        assert any(call[0][0] == controller.cfg['REFILL_VALVE_PIN'] and call[0][1] == GPIO.LOW 
                  for call in GPIO.output.call_args_list)
        assert controller._refill_complete
        
        # Verify event published
        assert any('refill_complete_float' in str(call) 
                  for call in controller._publish_message.call_args_list)
    
    def test_line_pressure_loss_emergency_shutdown(self, controller):
        """Test system enters safe state on pressure loss."""
        # Set running state
        controller._state = PumpState.RUNNING
        controller._engine_start_time = time.time() - 100
        controller._low_pressure_detected = False
        
        # Mock pressure loss (active low = no pressure)
        def mock_input(pin):
            if pin == controller.cfg['LINE_PRESSURE_PIN']:
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
        assert controller._state in [PumpState.LOW_PRESSURE, PumpState.STOPPING]
        
        # Verify event published
        assert any('low_pressure_detected' in str(call) 
                  for call in controller._publish_message.call_args_list)
    
    def test_emergency_mqtt_command_overrides_all_states(self, controller):
        """Test emergency MQTT command works in any state."""
        states_to_test = [
            PumpState.IDLE, 
            PumpState.ERROR,
            PumpState.REFILLING,
            PumpState.COOLDOWN
        ]
        
        for state in states_to_test:
            # Set state
            controller._state = state
            controller._shutting_down = False
            GPIO.output.reset_mock()
            
            # Send emergency start command
            controller.handle_emergency_command('start')
            
            # Allow brief time for processing
            time.sleep(0.1)
            
            # Verify pump starts
            assert controller._state in [PumpState.PRIMING, PumpState.STARTING, PumpState.RUNNING]
            
            # Reset for next test
            controller._cancel_all_timers()
            controller._state = PumpState.IDLE
    
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
        
        # Verify stopped
        assert controller._state == PumpState.COOLDOWN
        
        # Verify all critical pins turned off
        critical_pins = ['IGN_ON', 'IGN_START', 'MAIN_VALVE', 'PRIMING_VALVE']
        for pin_name in critical_pins:
            pin = controller.cfg[f'{pin_name}_PIN']
            assert any(call[0][0] == pin and call[0][1] == GPIO.LOW 
                      for call in GPIO.output.call_args_list)
    
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
            assert any(call[0][0] == pin and call[0][1] == GPIO.LOW 
                      for call in GPIO.output.call_args_list)
    
    def test_reservoir_monitoring_during_pump_operation(self, controller):
        """Test reservoir level is monitored during pump operation."""
        # Set running state
        controller._state = PumpState.RUNNING
        controller._refill_complete = False
        
        # Mock empty reservoir
        def mock_input(pin):
            if pin == controller.cfg['RESERVOIR_FLOAT_PIN']:
                return GPIO.HIGH  # Not full (active low)
            return GPIO.LOW
        
        GPIO.input = Mock(side_effect=mock_input)
        
        # Monitor reservoir
        controller._monitor_reservoir_level()
        
        # Verify refill valve opened
        assert any(call[0][0] == controller.cfg['REFILL_VALVE_PIN'] and call[0][1] == GPIO.HIGH 
                  for call in GPIO.output.call_args_list)
    
    def test_multiple_safety_triggers_priority(self, controller):
        """Test correct priority when multiple safety conditions trigger."""
        # Set running state
        controller._state = PumpState.RUNNING
        controller._engine_start_time = time.time()
        
        # Mock multiple safety issues
        def mock_input(pin):
            if pin == controller.cfg['EMERGENCY_BUTTON_PIN']:
                return GPIO.LOW  # Emergency button pressed
            elif pin == controller.cfg['LINE_PRESSURE_PIN']:
                return GPIO.HIGH  # Low pressure
            elif pin == controller.cfg['FLOW_SENSOR_PIN']:
                return GPIO.LOW  # No flow
            return GPIO.HIGH
        
        GPIO.input = Mock(side_effect=mock_input)
        
        # Trigger emergency button (highest priority)
        controller._emergency_switch_callback(controller.cfg['EMERGENCY_BUTTON_PIN'])
        
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
            if pin == controller.cfg['EMERGENCY_BUTTON_PIN']:
                return GPIO.LOW  # Pressed
            return GPIO.HIGH
        
        GPIO.input = Mock(side_effect=mock_input)
        
        # Trigger emergency button
        controller._emergency_switch_callback(controller.cfg['EMERGENCY_BUTTON_PIN'])
        time.sleep(0.1)
        
        # Verify emergency override works even in error state
        assert controller._state in [PumpState.PRIMING, PumpState.STARTING, PumpState.RUNNING]
    
    def test_graceful_degradation_without_sensors(self, controller):
        """Test system operates safely when optional sensors missing."""
        # Remove optional sensors
        controller.cfg['FLOW_SENSOR_PIN'] = None
        controller.cfg['LINE_PRESSURE_PIN'] = None
        
        # Start pump
        controller._state = PumpState.IDLE
        controller.handle_trigger(True)
        
        # Verify pump can still start
        assert controller._state == PumpState.PRIMING
        
        # Verify no errors from missing sensors
        controller._check_line_pressure()  # Should not crash
        assert not controller._low_pressure_detected
        
        # Dry run protection should be disabled without flow sensor
        controller._monitor_dry_run_protection()  # Should not crash
    
    def test_health_reporting_includes_safety_status(self, controller):
        """Test health reports include safety system status."""
        # Set various safety states
        controller._low_pressure_detected = True
        controller._refill_complete = False
        controller._dry_run_detected = True
        
        # Get health status
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