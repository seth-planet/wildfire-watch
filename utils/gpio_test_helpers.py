#!/usr/bin/env python3.12
"""GPIO Test Helper Utilities

This module consolidates common GPIO testing patterns and utilities
used across all GPIO test files. It provides reusable functions for:
- Pin state verification
- GPIO mock setup helpers
- Hardware state tracking utilities
- State waiting functions

BEST PRACTICES FOLLOWED:
1. NO mocking of PumpController or internal components
2. Uses real GPIO module or built-in simulation
3. Tests actual hardware behavior
4. Provides utilities that work with real components
"""

import time
import threading
from typing import Dict, Optional, Any, List, Callable, Union
from enum import Enum

# Import types from gpio_trigger
from gpio_trigger.trigger import PumpState, GPIO


def _get_config_value(config: Union[dict, Any], key: str) -> Any:
    """Get a value from config, supporting both dict and object access.
    
    Args:
        config: Either a dict or an object with attributes
        key: The configuration key to retrieve (e.g., 'MAIN_VALVE_PIN')
        
    Returns:
        The configuration value
    """
    if isinstance(config, dict):
        return config.get(key)
    else:
        # Convert uppercase key to lowercase for attribute access
        attr_name = key.lower()
        return getattr(config, attr_name, None)


def wait_for_state(controller, state: PumpState, timeout: float = 5) -> bool:
    """Wait for controller to reach specific state.
    
    Args:
        controller: The PumpController instance
        state: The desired PumpState to wait for
        timeout: Maximum time to wait in seconds (default: 5)
        
    Returns:
        bool: True if state was reached, False if timeout
        
    Note:
        Logs ERROR state entries for debugging if encountered while waiting.
    """
    import logging
    import inspect
    
    logger = logging.getLogger(__name__)
    start = time.time()
    
    while time.time() - start < timeout:
        if controller._state == state:
            return True
            
        # Log ERROR state entries for analysis
        if controller._state == PumpState.ERROR:
            frame = inspect.currentframe()
            caller = frame.f_back.f_code.co_name if frame.f_back else "unknown"
            logger.warning(f"ERROR state reached in test: {caller}, waiting for: {state.name}")
            
        time.sleep(0.01)
        
    return False


def wait_for_any_state(controller, states: List[PumpState], timeout: float = 5) -> Optional[PumpState]:
    """Wait for controller to reach any of the specified states.
    
    Args:
        controller: The PumpController instance
        states: List of acceptable PumpStates to wait for
        timeout: Maximum time to wait in seconds (default: 5)
        
    Returns:
        Optional[PumpState]: The state that was reached, or None if timeout
    """
    start = time.time()
    
    while time.time() - start < timeout:
        if controller._state in states:
            return controller._state
        time.sleep(0.01)
        
    return None


def verify_all_pins_low(gpio_test_setup, config: Union[dict, Any], 
                       pin_names: Optional[List[str]] = None) -> bool:
    """Verify all specified pins are in LOW state.
    
    Args:
        gpio_test_setup: The GPIO test fixture
        config: Configuration dict or object
        pin_names: List of pin names from config to check. 
                  If None, checks all output pins.
                  
    Returns:
        bool: True if all pins are LOW
    """
        
    if pin_names is None:
        pin_names = [
            'MAIN_VALVE_PIN', 'IGN_START_PIN', 'IGN_ON_PIN',
            'IGN_OFF_PIN', 'REFILL_VALVE_PIN', 'PRIMING_VALVE_PIN',
            'RPM_REDUCE_PIN'
        ]
    
    for pin_name in pin_names:
        pin = _get_config_value(config, pin_name)
        if pin and gpio_test_setup.input(pin) != gpio_test_setup.LOW:
            return False
            
    return True


def verify_pin_states(gpio_test_setup, config: Union[dict, Any],
                     expected_states: Dict[str, bool]) -> Dict[str, bool]:
    """Verify multiple pin states match expected values.
    
    Args:
        gpio_test_setup: The GPIO test fixture
        config: Configuration dict or object
        expected_states: Dict mapping pin names to expected HIGH (True) or LOW (False)
        
    Returns:
        Dict[str, bool]: Mapping of pin names to verification results
        
    Example:
        results = verify_pin_states(gpio, my_config, {
            'MAIN_VALVE_PIN': True,   # Expect HIGH
            'IGN_ON_PIN': False        # Expect LOW
        })
    """
        
    results = {}
    
    for pin_name, expected_high in expected_states.items():
        pin = _get_config_value(config, pin_name)
        if pin:
            actual = gpio_test_setup.input(pin)
            expected = gpio_test_setup.HIGH if expected_high else gpio_test_setup.LOW
            results[pin_name] = (actual == expected)
        else:
            results[pin_name] = False  # Pin not configured
            
    return results


def assert_pin_state(gpio_test_setup, config: Union[dict, Any], 
                    pin_name: str, expected_high: bool, 
                    message: Optional[str] = None) -> None:
    """Assert a single pin is in the expected state.
    
    Args:
        gpio_test_setup: The GPIO test fixture
        config: Configuration dict or object
        pin_name: Name of the pin from config
        expected_high: True if expecting HIGH, False if expecting LOW
        message: Optional custom assertion message
        
    Raises:
        AssertionError: If pin state doesn't match expected
    """
        
    pin = _get_config_value(config, pin_name)
    if not pin:
        raise AssertionError(f"Pin {pin_name} not configured")
        
    actual = gpio_test_setup.input(pin)
    expected = gpio_test_setup.HIGH if expected_high else gpio_test_setup.LOW
    
    if actual != expected:
        state_name = "HIGH" if expected_high else "LOW"
        actual_name = "HIGH" if actual == gpio_test_setup.HIGH else "LOW"
        msg = message or f"Pin {pin_name} should be {state_name}, but is {actual_name}"
        raise AssertionError(msg)


def setup_sensor_pin(gpio_test_setup, pin_name: str, pin_number: int, 
                    pull_up_down: Optional[int] = None) -> None:
    """Setup a sensor input pin with proper configuration.
    
    Args:
        gpio_test_setup: The GPIO test fixture
        pin_name: Name for reference (e.g., 'flow_sensor')
        pin_number: The GPIO pin number
        pull_up_down: Pull resistor config (GPIO.PUD_UP, GPIO.PUD_DOWN, or None)
    """
    if pull_up_down is not None:
        gpio_test_setup.setup(pin_number, gpio_test_setup.IN, pull_up_down=pull_up_down)
    else:
        gpio_test_setup.setup(pin_number, gpio_test_setup.IN)


def simulate_button_press(gpio_test_setup, pin_number: int, 
                         press_duration: float = 0.1,
                         active_low: bool = True) -> None:
    """Simulate a button press on a GPIO pin.
    
    Args:
        gpio_test_setup: The GPIO test fixture
        pin_number: The GPIO pin number
        press_duration: How long to hold the button (seconds)
        active_low: True if button is active low (default)
    """
    if active_low:
        # Active low: HIGH = not pressed, LOW = pressed
        gpio_test_setup._state[pin_number] = gpio_test_setup.LOW
        time.sleep(press_duration)
        gpio_test_setup._state[pin_number] = gpio_test_setup.HIGH
    else:
        # Active high: LOW = not pressed, HIGH = pressed
        gpio_test_setup._state[pin_number] = gpio_test_setup.HIGH
        time.sleep(press_duration)
        gpio_test_setup._state[pin_number] = gpio_test_setup.LOW


def simulate_sensor_state(gpio_test_setup, pin_number: int, 
                         active: bool, active_low: bool = False) -> None:
    """Set a sensor pin to active or inactive state.
    
    Args:
        gpio_test_setup: The GPIO test fixture
        pin_number: The GPIO pin number
        active: True if sensor should be active/triggered
        active_low: True if sensor is active low (default: False)
    """
    if active_low:
        # Active low: LOW = active, HIGH = inactive
        gpio_test_setup._state[pin_number] = gpio_test_setup.LOW if active else gpio_test_setup.HIGH
    else:
        # Active high: HIGH = active, LOW = inactive
        gpio_test_setup._state[pin_number] = gpio_test_setup.HIGH if active else gpio_test_setup.LOW


class PinMonitor:
    """Monitor GPIO pin state changes during tests.
    
    Usage:
        monitor = PinMonitor(gpio_test_setup, config=my_config)
        monitor.start_monitoring(['MAIN_VALVE_PIN', 'IGN_ON_PIN'])
        
        # Run test actions...
        
        changes = monitor.stop_monitoring()
        assert len(changes['MAIN_VALVE_PIN']) > 0  # Pin changed state
    """
    
    def __init__(self, gpio_test_setup, config: Union[dict, Any]):
        """Initialize pin monitor.
        
        Args:
            gpio_test_setup: The GPIO test fixture
            config: Configuration dict or object
        """
        self.gpio = gpio_test_setup
        self.config = config
        self.monitoring = False
        self.monitor_thread = None
        self.pin_changes = {}
        self.initial_states = {}
        self._lock = threading.Lock()
        
    def start_monitoring(self, pin_names: List[str], sample_rate: float = 0.001) -> None:
        """Start monitoring specified pins for state changes.
        
        Args:
            pin_names: List of pin names from config to monitor
            sample_rate: How often to check pin states (seconds)
        """
        with self._lock:
            if self.monitoring:
                raise RuntimeError("Already monitoring")
                
            self.pin_changes = {name: [] for name in pin_names}
            self.initial_states = {}
            
            # Record initial states
            for pin_name in pin_names:
                pin = _get_config_value(self.config, pin_name)
                if pin:
                    self.initial_states[pin_name] = self.gpio.input(pin)
                    
            self.monitoring = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop,
                args=(pin_names, sample_rate),
                daemon=True
            )
            self.monitor_thread.start()
            
    def stop_monitoring(self) -> Dict[str, List[tuple]]:
        """Stop monitoring and return pin state changes.
        
        Returns:
            Dict mapping pin names to list of (timestamp, state) tuples
        """
        with self._lock:
            if not self.monitoring:
                return {}
                
            self.monitoring = False
            
        # Wait for monitor thread to finish
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            
        return self.pin_changes
        
    def _monitor_loop(self, pin_names: List[str], sample_rate: float) -> None:
        """Monitor loop that runs in separate thread."""
        last_states = self.initial_states.copy()
        
        while self.monitoring:
            with self._lock:
                for pin_name in pin_names:
                    pin = _get_config_value(self.config, pin_name)
                    if pin:
                        current_state = self.gpio.input(pin)
                        
                        # Record state change
                        if pin_name not in last_states or current_state != last_states[pin_name]:
                            self.pin_changes[pin_name].append((time.time(), current_state))
                            last_states[pin_name] = current_state
                            
            time.sleep(sample_rate)


def wait_for_pin_state(gpio_test_setup, config: Union[dict, Any], 
                      pin_name: str, expected_high: bool,
                      timeout: float = 5, poll_interval: float = 0.01) -> bool:
    """Wait for a specific pin to reach expected state.
    
    Args:
        gpio_test_setup: The GPIO test fixture
        config: Configuration dict or object
        pin_name: Name of the pin from config
        expected_high: True to wait for HIGH, False to wait for LOW
        timeout: Maximum time to wait (seconds)
        poll_interval: How often to check the pin (seconds)
        
    Returns:
        bool: True if pin reached expected state, False if timeout
    """
        
    pin = _get_config_value(config, pin_name)
    if not pin:
        return False
        
    expected = gpio_test_setup.HIGH if expected_high else gpio_test_setup.LOW
    start = time.time()
    
    while time.time() - start < timeout:
        if gpio_test_setup.input(pin) == expected:
            return True
        time.sleep(poll_interval)
        
    return False


def get_all_pin_states(gpio_test_setup, config: Union[dict, Any],
                      pin_names: Optional[List[str]] = None) -> Dict[str, bool]:
    """Get current state of all specified pins.
    
    Args:
        gpio_test_setup: The GPIO test fixture
        config: Configuration dict or object
        pin_names: List of pin names from config. If None, checks all output pins.
        
    Returns:
        Dict mapping pin names to boolean (True = HIGH, False = LOW)
    """
        
    if pin_names is None:
        pin_names = [
            'MAIN_VALVE_PIN', 'IGN_START_PIN', 'IGN_ON_PIN',
            'IGN_OFF_PIN', 'REFILL_VALVE_PIN', 'PRIMING_VALVE_PIN',
            'RPM_REDUCE_PIN'
        ]
        
    states = {}
    for pin_name in pin_names:
        pin = _get_config_value(config, pin_name)
        if pin:
            state = gpio_test_setup.input(pin)
            states[pin_name] = (state == gpio_test_setup.HIGH)
        else:
            states[pin_name] = None  # Pin not configured
            
    return states


def verify_safe_shutdown_state(gpio_test_setup, config: Union[dict, Any]) -> bool:
    """Verify all critical pins are in safe shutdown state.
    
    Args:
        gpio_test_setup: The GPIO test fixture
        config: Configuration dict or object
        
    Returns:
        bool: True if all pins are in safe state
    """
    critical_pins = {
        'IGN_ON_PIN': False,      # Engine must be off
        'IGN_START_PIN': False,   # No starting signal
        'MAIN_VALVE_PIN': False,  # Valve closed (unless refilling)
        'RPM_REDUCE_PIN': False,  # No RPM reduction active
    }
    
    results = verify_pin_states(gpio_test_setup, config, critical_pins)
    return all(results.values())


def trigger_hardware_validation_failure(controller) -> None:
    """Simulate a hardware validation failure for testing error handling.
    
    Args:
        controller: The PumpController instance
        
    Note:
        This directly triggers error state as would happen with real
        hardware validation failure, following CLAUDE.md best practices
        of testing error handling without mocking internals.
    """
    controller._enter_error_state("Simulated hardware validation failure for testing")


def wait_for_stable_state(controller, timeout: float = 10) -> Optional[PumpState]:
    """Wait for controller to reach a stable (non-transitional) state.
    
    Args:
        controller: The PumpController instance
        timeout: Maximum time to wait (seconds)
        
    Returns:
        Optional[PumpState]: The stable state reached, or None if timeout
    """
    stable_states = [
        PumpState.IDLE,
        PumpState.RUNNING,
        PumpState.ERROR,
        PumpState.COOLDOWN,
        PumpState.REFILLING
    ]
    
    return wait_for_any_state(controller, stable_states, timeout)