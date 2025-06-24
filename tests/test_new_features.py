#!/usr/bin/env python3.12
"""
Test new features: zone-based activation, emergency bypass, single camera mode
"""

import os
import sys
import time
import json
import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'fire_consensus'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'gpio_trigger'))


class TestZoneBasedActivation:
    """Test zone-based activation feature"""
    
    def test_zone_mapping_in_trigger_payload(self):
        """Test that zone information is included in trigger payload"""
        with patch('consensus.mqtt.Client'):
            from consensus import FireConsensus
            
            # Create instance without calling __init__ to avoid MQTT
            consensus = FireConsensus.__new__(FireConsensus)
            
            # Mock the config with zone mapping
            mock_config = Mock()
            mock_config.ZONE_ACTIVATION = True
            mock_config.ZONE_MAPPING = {'cam1': 'zone_a', 'cam2': 'zone_b'}
            mock_config.NODE_ID = 'test-node'
            consensus.config = mock_config
            consensus.trigger_count = 1
            consensus.last_trigger_time = 0
            consensus.consensus_events = []
            consensus.lock = Mock()
            consensus.lock.__enter__ = Mock(return_value=None)
            consensus.lock.__exit__ = Mock(return_value=None)
            
            # Mock MQTT client
            mock_mqtt = MagicMock()
            consensus.mqtt_client = mock_mqtt
            
            # Test trigger with cameras in different zones
            cameras = ['cam1', 'cam2']
            details = {
                'cam1': {'confidence': 0.8},
                'cam2': {'confidence': 0.85}
            }
            
            consensus._trigger_fire_response(cameras, details)
            
            # Verify payload includes zone information
            assert mock_mqtt.publish.called
            call_args = mock_mqtt.publish.call_args
            payload = json.loads(call_args[0][1])
            
            assert 'zones' in payload
            assert 'zone_activation' in payload
            assert payload['zone_activation'] is True
            assert set(payload['zones']) == {'zone_a', 'zone_b'}
    
    def test_zone_activation_disabled(self):
        """Test trigger payload when zone activation is disabled"""
        with patch('consensus.mqtt.Client'):
            from consensus import FireConsensus
            
            consensus = FireConsensus.__new__(FireConsensus)
            
            mock_config = Mock()
            mock_config.ZONE_ACTIVATION = False
            mock_config.ZONE_MAPPING = {}
            mock_config.NODE_ID = 'test-node'
            consensus.config = mock_config
            consensus.trigger_count = 1
            consensus.last_trigger_time = 0
            consensus.consensus_events = []
            consensus.lock = Mock()
            consensus.lock.__enter__ = Mock(return_value=None)
            consensus.lock.__exit__ = Mock(return_value=None)
            
            mock_mqtt = MagicMock()
            consensus.mqtt_client = mock_mqtt
            
            cameras = ['cam1']
            details = {'cam1': {'confidence': 0.8}}
            
            consensus._trigger_fire_response(cameras, details)
            
            call_args = mock_mqtt.publish.call_args
            payload = json.loads(call_args[0][1])
            
            assert payload['zones'] is None
            assert payload['zone_activation'] is False


class TestSingleCameraMode:
    """Test single camera trigger mode"""
    
    def test_single_camera_consensus_logic(self):
        """Test the consensus logic supports single camera mode"""
        with patch('consensus.mqtt.Client'):
            from consensus import FireConsensus
            
            consensus = FireConsensus.__new__(FireConsensus)
            
            # Mock config for single camera mode
            mock_config = Mock()
            mock_config.SINGLE_CAMERA_TRIGGER = True
            mock_config.CONSENSUS_THRESHOLD = 2  # Would normally require 2 cameras
            mock_config.COOLDOWN_PERIOD = 0
            mock_config.DETECTION_WINDOW = 10
            consensus.config = mock_config
            consensus.last_trigger_time = 0
            consensus.cameras = {}
            consensus.lock = Mock()
            consensus.lock.__enter__ = Mock(return_value=None)
            consensus.lock.__exit__ = Mock(return_value=None)
            
            # Mock a camera with growing fire
            mock_camera = Mock()
            mock_camera.is_online.return_value = True
            mock_camera.get_growing_fires.return_value = ['fire1']
            current_time = time.time()
            mock_detection = Mock()
            mock_detection.confidence = 0.8
            mock_detection.timestamp = current_time
            mock_camera.detections = [mock_detection]
            consensus.cameras['cam1'] = mock_camera
            
            # Mock the trigger response method
            consensus._trigger_fire_response = Mock()
            
            # Call consensus check
            consensus._check_consensus()
            
            # Should trigger with single camera
            consensus._trigger_fire_response.assert_called_once()
            call_args = consensus._trigger_fire_response.call_args[0]
            assert call_args[0] == ['cam1']  # cameras_with_fire
    
    def test_multi_camera_mode_still_works(self):
        """Test that normal multi-camera mode still works when single camera is disabled"""
        with patch('consensus.mqtt.Client'):
            from consensus import FireConsensus
            
            consensus = FireConsensus.__new__(FireConsensus)
            
            mock_config = Mock()
            mock_config.SINGLE_CAMERA_TRIGGER = False
            mock_config.CONSENSUS_THRESHOLD = 2
            mock_config.COOLDOWN_PERIOD = 0
            mock_config.DETECTION_WINDOW = 10
            consensus.config = mock_config
            consensus.last_trigger_time = 0
            consensus.cameras = {}
            consensus.lock = Mock()
            consensus.lock.__enter__ = Mock(return_value=None)
            consensus.lock.__exit__ = Mock(return_value=None)
            
            # Mock two cameras with growing fires
            for cam_id in ['cam1', 'cam2']:
                mock_camera = Mock()
                mock_camera.is_online.return_value = True
                mock_camera.get_growing_fires.return_value = ['fire1']
                current_time = time.time()
                mock_detection = Mock()
                mock_detection.confidence = 0.8
                mock_detection.timestamp = current_time
                mock_camera.detections = [mock_detection]
                consensus.cameras[cam_id] = mock_camera
            
            consensus._trigger_fire_response = Mock()
            consensus._check_consensus()
            
            # Should trigger with two cameras
            consensus._trigger_fire_response.assert_called_once()
            call_args = consensus._trigger_fire_response.call_args[0]
            assert len(call_args[0]) == 2  # Two cameras
    
    def test_single_camera_insufficient_when_disabled(self):
        """Test single camera doesn't trigger when single camera mode is disabled"""
        with patch('consensus.mqtt.Client'):
            from consensus import FireConsensus
            
            consensus = FireConsensus.__new__(FireConsensus)
            
            mock_config = Mock()
            mock_config.SINGLE_CAMERA_TRIGGER = False
            mock_config.CONSENSUS_THRESHOLD = 2
            mock_config.COOLDOWN_PERIOD = 0
            mock_config.DETECTION_WINDOW = 10
            consensus.config = mock_config
            consensus.last_trigger_time = 0
            consensus.cameras = {}
            consensus.lock = Mock()
            consensus.lock.__enter__ = Mock(return_value=None)
            consensus.lock.__exit__ = Mock(return_value=None)
            
            # Mock only one camera with growing fire
            mock_camera = Mock()
            mock_camera.is_online.return_value = True
            mock_camera.get_growing_fires.return_value = ['fire1']
            current_time = time.time()
            mock_detection = Mock()
            mock_detection.confidence = 0.8
            mock_detection.timestamp = current_time
            mock_camera.detections = [mock_detection]
            consensus.cameras['cam1'] = mock_camera
            
            consensus._trigger_fire_response = Mock()
            consensus._check_consensus()
            
            # Should NOT trigger with single camera when disabled
            consensus._trigger_fire_response.assert_not_called()


class TestEmergencyBypass:
    """Test emergency bypass functionality"""
    
    @patch('trigger.GPIO', create=True)
    def test_emergency_start_command(self, mock_gpio):
        """Test emergency start bypass command"""
        from trigger import PumpController, PumpState
        
        # Create controller without calling __init__ to avoid MQTT
        controller = PumpController.__new__(PumpController)
        controller.cfg = {
            'EMERGENCY_TOPIC': 'fire/emergency',
            'IGN_START_PIN': 23,
            'MAIN_VALVE_PIN': 18
        }
        controller._lock = Mock()
        controller._lock.__enter__ = Mock(return_value=None)
        controller._lock.__exit__ = Mock(return_value=None)
        controller._state = PumpState.ERROR  # Start in error state
        controller._refill_complete = False
        controller._start_pump_sequence = Mock()
        controller._publish_event = Mock()
        
        # Test emergency start command
        controller.handle_emergency_command('start')
        
        # Should force start regardless of state
        assert controller._refill_complete is True
        assert controller._state == PumpState.IDLE
        controller._start_pump_sequence.assert_called_once()
        controller._publish_event.assert_called_with('emergency_bypass_start')
    
    @patch('trigger.GPIO', create=True)
    def test_emergency_stop_command(self, mock_gpio):
        """Test emergency stop command"""
        from trigger import PumpController, PumpState
        
        controller = PumpController.__new__(PumpController)
        controller.cfg = {
            'IGN_START_PIN': 23,
            'IGN_ON_PIN': 24,
            'IGN_OFF_PIN': 25,
            'MAIN_VALVE_PIN': 18,
            'REFILL_VALVE_PIN': 22,
            'PRIMING_VALVE_PIN': 26,
            'RPM_REDUCE_PIN': 27,
            'COOLDOWN_DELAY': 300
        }
        controller._lock = Mock()
        controller._lock.__enter__ = Mock(return_value=None)
        controller._lock.__exit__ = Mock(return_value=None)
        controller._state = PumpState.RUNNING
        # Initialize required attributes
        controller.gpio = mock_gpio
        controller.gpio.emergency_all_off = Mock(return_value={})  # Mock successful emergency stop
        controller.timer_manager = None
        controller._internal_timers = {'test_timer': Mock()}
        controller._cancel_all_timers = Mock()  # Mock the method that's actually called
        controller._cancel_timer = Mock()
        controller._set_pin = Mock()
        controller._schedule_timer = Mock()
        controller._enter_idle = Mock()
        controller._publish_event = Mock()
        controller._enter_cooldown = Mock()  # Mock the cooldown method
        controller._emergency_stop = Mock()  # Mock the actual emergency stop method
        
        # Test emergency stop command
        controller.handle_emergency_command('emergency_stop')
        
        # Should call the emergency stop method
        controller._emergency_stop.assert_called_once()
    
    def test_emergency_command_parsing(self):
        """Test emergency command parsing (JSON and plain text)"""
        from trigger import PumpController
        
        controller = PumpController.__new__(PumpController)
        controller._lock = Mock()
        controller._lock.__enter__ = Mock(return_value=None)
        controller._lock.__exit__ = Mock(return_value=None)
        controller._emergency_start = Mock()
        controller._emergency_stop = Mock()
        controller._set_pin = Mock()
        controller._publish_event = Mock()
        
        # Test JSON command
        controller.handle_emergency_command('{"action": "start"}')
        controller._emergency_start.assert_called_once()
        
        # Test plain text command
        controller._emergency_start.reset_mock()
        controller.handle_emergency_command('bypass_start')
        controller._emergency_start.assert_called_once()
        
        # Test valve commands
        controller.handle_emergency_command('valve_open')
        controller._set_pin.assert_called_with('MAIN_VALVE', True)
        controller._publish_event.assert_called_with('emergency_valve_open')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])