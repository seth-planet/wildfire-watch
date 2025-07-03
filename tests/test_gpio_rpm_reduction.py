#!/usr/bin/env python3.12
"""Test RPM reduction functionality for safe motor shutdown.

This module tests the REDUCING_RPM state and ensures the motor properly
slows down before stopping to prevent damage to the pump.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from gpio_trigger.trigger import PumpController, PumpState, CONFIG

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


class TestRPMReduction:
    """Test motor RPM reduction before shutdown."""
    
    @pytest.fixture
    def test_config(self):
        """Create test configuration."""
        config = CONFIG.copy()
        config.update({
            'FIRE_OFF_DELAY': 5.0,  # Fast for testing
            'RPM_REDUCTION_LEAD': 3.0,  # 3 seconds before shutdown
            'MAX_ENGINE_RUNTIME': 30.0,  # 30 seconds for testing
            'RPM_REDUCE_PIN': 27,
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
                # Mock MQTT methods
                controller._mqtt_client = Mock()
                controller._mqtt_connected = True
                controller._publish_message = Mock(return_value=True)
                yield controller
    
    def test_rpm_reduces_before_normal_shutdown(self, controller):
        """Test motor slows down when fire is detected as out."""
        # Start pump
        controller._state = PumpState.IDLE
        controller.handle_trigger(True)
        assert controller._state == PumpState.PRIMING
        
        # Simulate engine running
        controller._state = PumpState.RUNNING
        controller._engine_start_time = time.time()
        controller._last_trigger_time = time.time()
        
        # Simulate fire out
        controller.handle_trigger(False)
        
        # Fast forward time to fire off delay
        controller._last_trigger_time = time.time() - controller.cfg['FIRE_OFF_DELAY'] - 1
        
        # Run fire off check
        controller._check_fire_off()
        
        # Verify RPM reduction started
        assert controller._state == PumpState.REDUCING_RPM
        assert GPIO.output.called
        assert any(call[0][0] == controller.cfg['RPM_REDUCE_PIN'] and call[0][1] == GPIO.HIGH 
                  for call in GPIO.output.call_args_list)
        
        # Verify shutdown is scheduled after RPM reduction
        assert controller._has_timer('delayed_shutdown_after_rpm')
        
        # Verify event published
        assert any('rpm_reduced' in str(call) for call in controller._publish_message.call_args_list)
    
    def test_rpm_reduces_on_max_runtime(self, controller):
        """Test motor slows down before max runtime shutdown."""
        # Start pump
        controller._state = PumpState.IDLE
        controller.handle_trigger(True)
        
        # Simulate entering running state
        controller._state = PumpState.STARTING
        controller._enter_running()
        
        # Verify RPM reduction timer scheduled
        assert controller._has_timer('rpm_reduction')
        
        # Get the scheduled time
        rpm_reduction_time = controller.cfg['MAX_ENGINE_RUNTIME'] - controller.cfg['RPM_REDUCTION_LEAD']
        assert rpm_reduction_time > 0
        
        # Fast forward to RPM reduction time
        controller._engine_start_time = time.time() - rpm_reduction_time - 1
        
        # Manually trigger RPM reduction (simulating timer)
        controller._reduce_rpm()
        
        # Verify state change and pin activation
        assert controller._state == PumpState.REDUCING_RPM
        assert GPIO.output.called
        assert any(call[0][0] == controller.cfg['RPM_REDUCE_PIN'] and call[0][1] == GPIO.HIGH 
                  for call in GPIO.output.call_args_list)
    
    def test_emergency_stop_applies_brief_rpm_reduction(self, controller):
        """Test emergency stop applies brief RPM reduction for safety."""
        # Setup running state
        controller._state = PumpState.RUNNING
        controller._engine_start_time = time.time()
        
        # Clear previous GPIO calls
        GPIO.output.reset_mock()
        
        # Perform emergency stop
        with patch('time.sleep') as mock_sleep:
            controller._emergency_stop()
            
            # Verify brief RPM reduction applied
            assert GPIO.output.called
            assert any(call[0][0] == controller.cfg['RPM_REDUCE_PIN'] and call[0][1] == GPIO.HIGH 
                      for call in GPIO.output.call_args_list)
            
            # Verify brief sleep for emergency RPM reduction
            mock_sleep.assert_called_with(2.0)
        
        # Verify state changed to COOLDOWN
        assert controller._state == PumpState.COOLDOWN
    
    def test_shutdown_from_reducing_rpm_state(self, controller):
        """Test shutdown works correctly from REDUCING_RPM state."""
        # Setup in REDUCING_RPM state
        controller._state = PumpState.REDUCING_RPM
        controller._engine_start_time = time.time() - 100
        controller._set_pin('RPM_REDUCE', True)
        
        # Shutdown engine
        controller._shutdown_engine()
        
        # Verify shutdown proceeded without additional RPM reduction
        assert controller._state == PumpState.STOPPING
        assert controller._shutting_down
        
        # Verify RPM reduce pin turned off
        assert any(call[0][0] == controller.cfg['RPM_REDUCE_PIN'] and call[0][1] == GPIO.LOW 
                  for call in GPIO.output.call_args_list)
    
    def test_direct_shutdown_applies_safety_rpm_reduction(self, controller):
        """Test direct shutdown from RUNNING applies safety RPM reduction."""
        # Setup running state
        controller._state = PumpState.RUNNING
        controller._engine_start_time = time.time() - 60
        
        # Clear GPIO calls
        GPIO.output.reset_mock()
        
        # Direct shutdown (e.g., from max runtime)
        with patch('time.sleep') as mock_sleep:
            controller._shutdown_engine()
            
            # Verify warning logged
            assert "Direct shutdown from RUNNING" in str(controller._publish_message.call_args_list)
            
            # Verify RPM reduction applied
            assert any(call[0][0] == controller.cfg['RPM_REDUCE_PIN'] and call[0][1] == GPIO.HIGH 
                      for call in GPIO.output.call_args_list)
            
            # Verify brief safety delay
            mock_sleep.assert_called_with(2.0)
    
    def test_cancel_shutdown_restores_rpm(self, controller):
        """Test cancelling shutdown turns off RPM reduction."""
        # Setup in REDUCING_RPM state
        controller._state = PumpState.REDUCING_RPM
        controller._set_pin('RPM_REDUCE', True)
        controller._engine_start_time = time.time() - 100
        
        # Cancel shutdown (fire detected again)
        controller._cancel_shutdown()
        
        # Verify state restored
        assert controller._state == PumpState.RUNNING
        
        # Verify RPM reduction turned off
        assert any(call[0][0] == controller.cfg['RPM_REDUCE_PIN'] and call[0][1] == GPIO.LOW 
                  for call in GPIO.output.call_args_list)
    
    def test_rpm_reduction_with_concurrent_triggers(self, controller):
        """Test RPM reduction handles concurrent fire on/off triggers correctly."""
        # Start pump
        controller._state = PumpState.RUNNING
        controller._engine_start_time = time.time()
        controller._last_trigger_time = time.time()
        
        # Fire off
        controller.handle_trigger(False)
        controller._last_trigger_time = time.time() - controller.cfg['FIRE_OFF_DELAY'] - 1
        controller._check_fire_off()
        
        # Verify RPM reduction started
        assert controller._state == PumpState.REDUCING_RPM
        
        # Fire detected again before shutdown
        controller.handle_trigger(True)
        
        # Verify pump continues running (cancel shutdown)
        if hasattr(controller, '_cancel_shutdown'):
            controller._cancel_shutdown()
            assert controller._state == PumpState.RUNNING
            assert not controller._has_timer('delayed_shutdown_after_rpm')
    
    def test_rpm_state_in_status_report(self, controller):
        """Test RPM reduction state appears in status reports."""
        # Set to REDUCING_RPM state
        controller._state = PumpState.REDUCING_RPM
        controller._set_pin('RPM_REDUCE', True)
        
        # Get status
        status = controller._get_state_snapshot()
        
        # Verify state reported correctly
        assert status['state'] == 'REDUCING_RPM'
        assert 'rpm_reduce' in status['gpio_state']
        assert status['gpio_state']['rpm_reduce'] is True
    
    def test_rpm_reduction_survives_mqtt_disconnect(self, controller):
        """Test RPM reduction continues even if MQTT disconnects."""
        # Setup running state
        controller._state = PumpState.RUNNING
        controller._engine_start_time = time.time()
        
        # Simulate MQTT disconnect
        controller._mqtt_connected = False
        
        # Trigger RPM reduction
        controller._reduce_rpm()
        
        # Verify state changed despite MQTT issue
        assert controller._state == PumpState.REDUCING_RPM
        
        # Verify GPIO still activated
        assert any(call[0][0] == controller.cfg['RPM_REDUCE_PIN'] and call[0][1] == GPIO.HIGH 
                  for call in GPIO.output.call_args_list)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])