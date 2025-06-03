#!/usr/bin/env python3
"""
Comprehensive tests for PumpController
Tests basic operation, edge cases, timing, concurrency, and fail-safe behavior
"""
import os
import sys
import time
import json
import threading
import pytest
from unittest.mock import Mock, MagicMock, patch, call

# Add trigger module to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../gpio_trigger")))

# Import after path setup
import trigger
from trigger import PumpController, GPIO, CONFIG, PumpState

# ─────────────────────────────────────────────────────────────
# Test Fixtures and Mocks
# ─────────────────────────────────────────────────────────────
class MockMQTTClient:
    """Mock MQTT client for testing"""
    def __init__(self):
        self.connected = False
        self.subscriptions = []
        self.publications = []
        self.will_topic = None
        self.will_payload = None
        self.on_connect = None
        self.on_message = None
        self.on_disconnect = None
    
    def will_set(self, topic, payload, qos, retain):
        self.will_topic = topic
        self.will_payload = payload
    
    def connect(self, broker, port, keepalive):
        self.connected = True
        if self.on_connect:
            self.on_connect(self, None, None, 0)
    
    def loop_start(self):
        pass
    
    def loop_stop(self):
        pass
    
    def disconnect(self):
        self.connected = False
        if self.on_disconnect:
            self.on_disconnect(self, None, 0)
    
    def subscribe(self, topics):
        self.subscriptions.extend(topics)
    
    def publish(self, topic, payload, qos=0):
        try:
            parsed = json.loads(payload)
            self.publications.append((topic, parsed, qos))
        except:
            self.publications.append((topic, payload, qos))
    
    def simulate_message(self, topic, payload):
        """Simulate receiving a message"""
        if self.on_message:
            msg = Mock()
            msg.topic = topic
            msg.payload = payload
            self.on_message(self, None, msg)

@pytest.fixture
def mock_gpio():
    """Reset GPIO state before each test"""
    GPIO._state.clear()
    # Set all pins to LOW initially
    for pin_name in ['MAIN_VALVE_PIN', 'IGN_START_PIN', 'IGN_ON_PIN',
                     'IGN_OFF_PIN', 'REFILL_VALVE_PIN', 'PRIMING_VALVE_PIN',
                     'RPM_REDUCE_PIN']:
        GPIO.setup(CONFIG[pin_name], GPIO.OUT, initial=GPIO.LOW)
    yield GPIO
    GPIO._state.clear()

@pytest.fixture
def mock_mqtt():
    """Mock MQTT client"""
    client = MockMQTTClient()
    with patch('trigger.mqtt.Client', return_value=client):
        yield client

@pytest.fixture
def controller(mock_gpio, mock_mqtt, monkeypatch):
    """Create controller with mocked dependencies"""
    # Speed up timings for tests
    monkeypatch.setenv("VALVE_PRE_OPEN_DELAY", "0.1")
    monkeypatch.setenv("IGNITION_START_DURATION", "0.05")
    monkeypatch.setenv("FIRE_OFF_DELAY", "0.5")
    monkeypatch.setenv("VALVE_CLOSE_DELAY", "0.3")
    monkeypatch.setenv("IGNITION_OFF_DURATION", "0.1")
    monkeypatch.setenv("MAX_ENGINE_RUNTIME", "2.0")
    monkeypatch.setenv("REFILL_MULTIPLIER", "2")
    monkeypatch.setenv("PRIMING_DURATION", "0.2")
    monkeypatch.setenv("RPM_REDUCTION_LEAD", "0.5")
    monkeypatch.setenv("HEALTH_INTERVAL", "10")
    
    # Reload config
    trigger.CONFIG.update({
        'PRE_OPEN_DELAY': 0.1,
        'IGNITION_START_DURATION': 0.05,
        'FIRE_OFF_DELAY': 0.5,
        'VALVE_CLOSE_DELAY': 0.3,
        'IGNITION_OFF_DURATION': 0.1,
        'MAX_ENGINE_RUNTIME': 2.0,
        'REFILL_MULTIPLIER': 2,
        'PRIMING_DURATION': 0.2,
        'RPM_REDUCTION_LEAD': 0.5,
        'HEALTH_INTERVAL': 10,
    })
    
    controller = PumpController()
    yield controller
    controller.cleanup()

def wait_for_state(controller, state, timeout=5):
    """Wait for controller to reach specific state"""
    start = time.time()
    while time.time() - start < timeout:
        if controller._state == state:
            return True
        time.sleep(0.01)
    return False

def get_published_actions(mqtt_client):
    """Extract action names from published events"""
    return [pub[1].get('action') for pub in mqtt_client.publications
            if pub[0] == CONFIG['TELEMETRY_TOPIC']]

# ─────────────────────────────────────────────────────────────
# Basic Operation Tests
# ─────────────────────────────────────────────────────────────
class TestBasicOperation:
    def test_initialization(self, controller, mock_gpio):
        """Test controller initializes to safe state"""
        assert controller._state == PumpState.IDLE
        assert all(not mock_gpio.input(CONFIG[pin])
                  for pin in ['MAIN_VALVE_PIN', 'IGN_ON_PIN', 'IGN_START_PIN'])
        assert controller._shutting_down is False
        assert controller._engine_start_time is None
    
    def test_fire_trigger_starts_sequence(self, controller, mock_mqtt):
        """Test fire trigger starts pump sequence"""
        controller.handle_fire_trigger()
        
        # Should open valves immediately
        assert GPIO.input(CONFIG['MAIN_VALVE_PIN']) is True
        assert GPIO.input(CONFIG['PRIMING_VALVE_PIN']) is True
        assert GPIO.input(CONFIG['REFILL_VALVE_PIN']) is True
        
        # Should transition to priming
        assert controller._state == PumpState.PRIMING
        
        # Should have scheduled engine start
        assert 'start_engine' in controller._timers
        
        # Wait for engine start
        assert wait_for_state(controller, PumpState.RUNNING, timeout=1)
        
        # Check engine is on and refill valve still open
        assert GPIO.input(CONFIG['IGN_ON_PIN']) is True
        assert GPIO.input(CONFIG['REFILL_VALVE_PIN']) is True
    
    def test_normal_shutdown_sequence(self, controller, mock_mqtt):
        """Test normal shutdown after fire off delay"""
        # Start pump
        controller.handle_fire_trigger()
        wait_for_state(controller, PumpState.RUNNING)
        
        # Wait for fire off delay
        time.sleep(0.6)
        
        # Should start shutdown
        assert wait_for_state(controller, PumpState.STOPPING)
        
        # Engine should be off
        assert GPIO.input(CONFIG['IGN_ON_PIN']) is False
        
        # Wait for cooldown
        assert wait_for_state(controller, PumpState.COOLDOWN, timeout=2)
    
    def test_multiple_triggers_extend_runtime(self, controller):
        """Test multiple fire triggers extend runtime"""
        # First trigger
        controller.handle_fire_trigger()
        wait_for_state(controller, PumpState.RUNNING)
        
        # Wait almost to fire off delay
        time.sleep(0.4)
        
        # Second trigger should reset timer
        controller.handle_fire_trigger()
        
        # Wait original fire off time
        time.sleep(0.2)
        
        # Should still be running
        assert controller._state == PumpState.RUNNING
        assert GPIO.input(CONFIG['IGN_ON_PIN']) is True

# ─────────────────────────────────────────────────────────────
# Safety and Fail-Safe Tests
# ─────────────────────────────────────────────────────────────
class TestSafety:
    def test_valve_must_be_open_for_ignition(self, controller):
        """Test engine won't start without valve open"""
        # Manually close valve
        GPIO.output(CONFIG['MAIN_VALVE_PIN'], GPIO.LOW)
        
        # Try to start engine directly
        controller._state = PumpState.PRIMING
        controller._start_engine()
        
        # Should enter error state
        assert controller._state == PumpState.ERROR
        assert GPIO.input(CONFIG['IGN_ON_PIN']) is False
    
    def test_max_runtime_enforcement(self, controller):
        """Test pump stops after max runtime"""
        controller.handle_fire_trigger()
        wait_for_state(controller, PumpState.RUNNING)
        
        # Continuously send triggers to prevent fire-off shutdown
        for _ in range(5):
            time.sleep(0.3)
            controller.handle_fire_trigger()
        
        # Should hit max runtime and shutdown
        assert wait_for_state(controller, PumpState.STOPPING, timeout=3)
        assert GPIO.input(CONFIG['IGN_ON_PIN']) is False
    
    def test_rpm_reduction_before_shutdown(self, controller):
        """Test RPM is reduced before shutdown"""
        controller.handle_fire_trigger()
        wait_for_state(controller, PumpState.RUNNING)
        
        # Wait for RPM reduction time (max_runtime - rpm_lead = 2.0 - 0.5 = 1.5s)
        time.sleep(1.6)
        
        # Should be in RPM reduction state
        assert controller._state == PumpState.REDUCING_RPM
        assert GPIO.input(CONFIG['RPM_REDUCE_PIN']) is True
    
    def test_emergency_valve_open_on_trigger(self, controller):
        """Test valve opens immediately on fire detection even if closed"""
        # Start pump and let it shutdown
        controller.handle_fire_trigger()
        wait_for_state(controller, PumpState.RUNNING)
        controller._shutdown_engine()
        wait_for_state(controller, PumpState.COOLDOWN)
        
        # Wait for valve to close
        time.sleep(0.4)
        assert GPIO.input(CONFIG['MAIN_VALVE_PIN']) is False
        
        # New fire trigger should immediately open valve
        controller.handle_fire_trigger()
        assert GPIO.input(CONFIG['MAIN_VALVE_PIN']) is True
    
    def test_refill_valve_runtime_multiplier(self, controller):
        """Test refill valve stays open for runtime * multiplier"""
        controller.handle_fire_trigger()
        wait_for_state(controller, PumpState.RUNNING)
        
        # Run for specific time
        run_time = 0.3
        time.sleep(run_time)
        
        # Shutdown
        controller._shutdown_engine()
        
        # Refill valve should still be open
        assert GPIO.input(CONFIG['REFILL_VALVE_PIN']) is True
        
        # Wait for refill time (runtime * multiplier)
        time.sleep(run_time * CONFIG['REFILL_MULTIPLIER'] + 0.1)
        
        # Refill valve should be closed
        assert GPIO.input(CONFIG['REFILL_VALVE_PIN']) is False

# ─────────────────────────────────────────────────────────────
# Concurrency and Edge Case Tests
# ─────────────────────────────────────────────────────────────
class TestConcurrency:
    def test_concurrent_triggers(self, controller):
        """Test handling multiple concurrent fire triggers"""
        # Start multiple threads triggering fires
        threads = []
        for _ in range(10):
            t = threading.Thread(target=controller.handle_fire_trigger)
            t.start()
            threads.append(t)
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Should be in valid state
        assert controller._state in [PumpState.PRIMING, PumpState.STARTING, PumpState.RUNNING]
        
        # Should have exactly one start timer
        start_timers = [name for name in controller._timers if name == 'start_engine']
        assert len(start_timers) <= 1
    
    def test_trigger_during_shutdown(self, controller):
        """Test fire trigger during shutdown cancels it"""
        controller.handle_fire_trigger()
        wait_for_state(controller, PumpState.RUNNING)
        
        # Start shutdown
        controller._shutdown_engine()
        assert controller._state == PumpState.STOPPING
        
        # New trigger should cancel shutdown
        controller.handle_fire_trigger()
        time.sleep(0.1)
        
        # Should be back to running
        assert controller._state == PumpState.RUNNING
        assert GPIO.input(CONFIG['IGN_ON_PIN']) is True
        assert controller._shutting_down is False
    
    def test_trigger_during_cooldown(self, controller):
        """Test fire trigger during cooldown restarts pump"""
        controller.handle_fire_trigger()
        wait_for_state(controller, PumpState.RUNNING)
        controller._shutdown_engine()
        wait_for_state(controller, PumpState.COOLDOWN)
        
        # Trigger during cooldown
        controller.handle_fire_trigger()
        
        # Should restart sequence
        assert wait_for_state(controller, PumpState.RUNNING, timeout=1)
        assert GPIO.input(CONFIG['IGN_ON_PIN']) is True
    
    def test_state_transitions_are_atomic(self, controller):
        """Test state transitions are thread-safe"""
        results = []
        
        def observe_states():
            states = []
            for _ in range(100):
                states.append(controller._state)
                time.sleep(0.001)
            results.append(states)
        
        # Start observer threads
        observers = []
        for _ in range(5):
            t = threading.Thread(target=observe_states)
            t.start()
            observers.append(t)
        
        # Trigger state changes
        controller.handle_fire_trigger()
        time.sleep(0.5)
        controller._shutdown_engine()
        
        # Wait for observers
        for t in observers:
            t.join()
        
        # All observers should see valid states
        valid_states = set(PumpState)
        for states in results:
            assert all(s in valid_states for s in states)

# ─────────────────────────────────────────────────────────────
# Error Handling Tests
# ─────────────────────────────────────────────────────────────
class TestErrorHandling:
    def test_gpio_failure_handling(self, controller, monkeypatch):
        """Test handling of GPIO failures"""
        # Make GPIO.output raise exception
        def failing_output(pin, value):
            raise Exception("GPIO Error")
        
        monkeypatch.setattr(GPIO, 'output', failing_output)
        
        # Try to start pump
        controller.handle_fire_trigger()
        
        # Should enter error state
        assert controller._state == PumpState.ERROR
    
    def test_error_state_ignores_triggers(self, controller):
        """Test error state ignores fire triggers"""
        # Force error state
        controller._enter_error_state("Test error")
        assert controller._state == PumpState.ERROR
        
        # Try to trigger
        controller.handle_fire_trigger()
        
        # Should still be in error state
        assert controller._state == PumpState.ERROR
        assert GPIO.input(CONFIG['IGN_ON_PIN']) is False
    
    def test_mqtt_disconnection_handling(self, controller, mock_mqtt):
        """Test MQTT disconnection doesn't crash controller"""
        # Simulate disconnection
        mock_mqtt.on_disconnect(mock_mqtt, None, 1)
        
        # Controller should still function
        controller.handle_fire_trigger()
        assert wait_for_state(controller, PumpState.RUNNING)
    
    def test_timer_exception_handling(self, controller, monkeypatch):
        """Test timer exceptions are caught"""
        # Make a timer function raise exception
        def failing_function():
            raise Exception("Timer failed")
        
        controller._schedule_timer('test_timer', failing_function, 0.1)
        
        # Wait for timer to fire
        time.sleep(0.2)
        
        # Controller should still be functional
        assert controller._state != PumpState.ERROR

# ─────────────────────────────────────────────────────────────
# MQTT and Telemetry Tests
# ─────────────────────────────────────────────────────────────
class TestMQTT:
    def test_mqtt_connection_and_subscription(self, controller, mock_mqtt):
        """Test MQTT connects and subscribes correctly"""
        assert mock_mqtt.connected
        assert (CONFIG['TRIGGER_TOPIC'], 0) in mock_mqtt.subscriptions
    
    def test_fire_trigger_via_mqtt(self, controller, mock_mqtt):
        """Test fire trigger via MQTT message"""
        mock_mqtt.simulate_message(CONFIG['TRIGGER_TOPIC'], '{}')
        
        # Should start pump sequence
        assert wait_for_state(controller, PumpState.RUNNING)
        assert GPIO.input(CONFIG['IGN_ON_PIN']) is True
    
    def test_telemetry_events_published(self, controller, mock_mqtt):
        """Test telemetry events are published"""
        controller.handle_fire_trigger()
        wait_for_state(controller, PumpState.RUNNING)
        
        # Check expected events were published
        actions = get_published_actions(mock_mqtt)
        assert 'pump_sequence_start' in actions
        assert 'valve_opened' in actions or 'emergency_valve_open' in actions
        assert 'refill_valve_opened_immediately' in actions
        assert 'engine_running' in actions
    
    def test_health_reports_published(self, controller, mock_mqtt):
        """Test periodic health reports"""
        # Clear initial publications
        mock_mqtt.publications.clear()
        
        # Trigger health report
        controller._publish_health()
        
        # Check health report published
        actions = get_published_actions(mock_mqtt)
        assert 'health_report' in actions
        
        # Verify health data
        health_pub = next(p for p in mock_mqtt.publications
                         if p[1].get('action') == 'health_report')
        assert 'total_runtime' in health_pub[1]
        assert 'state' in health_pub[1]
    
    def test_lwt_configuration(self, mock_mqtt):
        """Test Last Will and Testament is configured"""
        assert mock_mqtt.will_topic is not None
        assert 'offline' in mock_mqtt.will_payload

# ─────────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────────
class TestIntegration:
    def test_complete_fire_cycle(self, controller, mock_mqtt):
        """Test complete fire detection and response cycle"""
        # Verify initial state
        assert controller._state == PumpState.IDLE
        assert all(not GPIO.input(CONFIG[pin])
                  for pin in ['MAIN_VALVE_PIN', 'IGN_ON_PIN'])
        
        # Fire detected
        controller.handle_fire_trigger()
        
        # Valve opens immediately
        assert GPIO.input(CONFIG['MAIN_VALVE_PIN']) is True
        
        # Priming starts
        assert controller._state == PumpState.PRIMING
        assert GPIO.input(CONFIG['PRIMING_VALVE_PIN']) is True
        
        # Engine starts
        assert wait_for_state(controller, PumpState.RUNNING)
        assert GPIO.input(CONFIG['IGN_ON_PIN']) is True
        assert GPIO.input(CONFIG['REFILL_VALVE_PIN']) is True
        
        # Priming valve closes after duration
        time.sleep(0.3)
        assert GPIO.input(CONFIG['PRIMING_VALVE_PIN']) is False
        
        # Fire continues - send more triggers
        for _ in range(3):
            time.sleep(0.2)
            controller.handle_fire_trigger()
        
        # RPM reduction before max runtime
        assert wait_for_state(controller, PumpState.REDUCING_RPM, timeout=2)
        assert GPIO.input(CONFIG['RPM_REDUCE_PIN']) is True
        
        # Max runtime shutdown
        assert wait_for_state(controller, PumpState.STOPPING, timeout=1)
        assert GPIO.input(CONFIG['IGN_ON_PIN']) is False
        
        # Cooldown
        assert wait_for_state(controller, PumpState.COOLDOWN, timeout=2)
        
        # Return to idle
        assert wait_for_state(controller, PumpState.IDLE, timeout=65)
    
    def test_rapid_on_off_cycles(self, controller):
        """Test rapid on/off fire detection"""
        for cycle in range(5):
            # Fire on
            controller.handle_fire_trigger()
            wait_for_state(controller, PumpState.RUNNING)
            
            # Let it run briefly
            time.sleep(0.2)
            
            # Force shutdown
            controller._shutdown_engine()
            wait_for_state(controller, PumpState.COOLDOWN)
            
            # Immediate restart
            controller.handle_fire_trigger()
        
        # System should still be functional
        assert controller._state in [PumpState.PRIMING, PumpState.STARTING,
                                     PumpState.RUNNING]
    
    def test_cleanup_from_various_states(self, controller):
        """Test cleanup works from any state"""
        states_to_test = [
            PumpState.IDLE,
            PumpState.PRIMING,
            PumpState.RUNNING,
            PumpState.REDUCING_RPM,
            PumpState.STOPPING,
            PumpState.COOLDOWN,
        ]
        
        for state in states_to_test:
            # Reset
            controller._state = PumpState.IDLE
            GPIO._state.clear()
            controller._init_gpio()
            
            # Set specific state
            if state in [PumpState.PRIMING, PumpState.STARTING, PumpState.RUNNING]:
                controller.handle_fire_trigger()
                if state == PumpState.RUNNING:
                    wait_for_state(controller, PumpState.RUNNING)
            elif state == PumpState.REDUCING_RPM:
                controller.handle_fire_trigger()
                wait_for_state(controller, PumpState.RUNNING)
                controller._reduce_rpm()
            elif state == PumpState.STOPPING:
                controller.handle_fire_trigger()
                wait_for_state(controller, PumpState.RUNNING)
                controller._shutdown_engine()
            elif state == PumpState.COOLDOWN:
                controller.handle_fire_trigger()
                wait_for_state(controller, PumpState.RUNNING)
                controller._shutdown_engine()
                wait_for_state(controller, PumpState.COOLDOWN)
            
            # Cleanup
            controller.cleanup()
            
            # Verify safe state
            assert GPIO.input(CONFIG['IGN_ON_PIN']) is False
            assert GPIO.input(CONFIG['IGN_START_PIN']) is False

# ─────────────────────────────────────────────────────────────
# Performance Tests
# ─────────────────────────────────────────────────────────────
class TestPerformance:
    def test_timer_scheduling_performance(self, controller):
        """Test timer scheduling doesn't degrade with many operations"""
        start_time = time.time()
        
        # Schedule many timers
        for i in range(100):
            controller._schedule_timer(f'test_{i}', lambda: None, 10)
        
        # Should complete quickly
        elapsed = time.time() - start_time
        assert elapsed < 0.1
        
        # Cancel all
        for i in range(100):
            controller._cancel_timer(f'test_{i}')
    
    def test_concurrent_event_handling(self, controller):
        """Test handling many concurrent events"""
        event_count = 50
        handled = []
        
        def fire_event():
            controller.handle_fire_trigger()
            handled.append(True)
        
        # Start many threads
        threads = []
        for _ in range(event_count):
            t = threading.Thread(target=fire_event)
            t.start()
            threads.append(t)
        
        # Wait for completion
        for t in threads:
            t.join(timeout=1)
        
        # All events should be handled
        assert len(handled) == event_count
        
        # System should be in valid state
        assert controller._state in [PumpState.PRIMING, PumpState.STARTING,
                                     PumpState.RUNNING]

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
