#!/usr/bin/env python3.12
"""Fix for GPIO parallel test execution issues.

The problem: When multiple test workers import gpio_trigger.trigger, they share
the same SimulatedGPIO._state dictionary, causing conflicts.

The solution: Create a test-specific GPIO mock that isolates state per test instance.
"""

import threading
from typing import Dict, Any


class TestGPIO:
    """Test-specific GPIO mock with isolated state per instance.
    
    Unlike the SimulatedGPIO in trigger.py which has class-level shared state,
    this version creates instance-specific state for each test.
    """
    BCM = "BCM"
    OUT = "OUT"
    IN = "IN"
    PUD_UP = "PUD_UP"
    PUD_DOWN = "PUD_DOWN"
    HIGH = True
    LOW = False
    
    def __init__(self):
        """Initialize with instance-specific state."""
        self._state: Dict[int, bool] = {}
        self._lock = threading.RLock()
        self._mode = None
        self._warnings = False
    
    def setmode(self, mode):
        """Set pin numbering mode."""
        with self._lock:
            self._mode = mode
    
    def setwarnings(self, warnings):
        """Enable/disable warnings."""
        with self._lock:
            self._warnings = warnings
    
    def setup(self, pin, mode, initial=None, pull_up_down=None):
        """Setup a pin for input or output."""
        with self._lock:
            if mode == self.OUT:
                self._state[pin] = initial if initial is not None else self.LOW
            else:
                # Input pins default based on pull resistor
                if pull_up_down == self.PUD_UP:
                    self._state[pin] = self.HIGH
                else:
                    self._state[pin] = self.LOW
    
    def output(self, pin, value):
        """Set output pin state."""
        with self._lock:
            # Convert GPIO constants to boolean consistently
            if value == self.HIGH or value is True:
                self._state[pin] = True
            elif value == self.LOW or value is False:
                self._state[pin] = False
            else:
                self._state[pin] = bool(value)
    
    def input(self, pin):
        """Read input pin state."""
        with self._lock:
            return self._state.get(pin, self.LOW)
    
    def cleanup(self, pins=None):
        """Cleanup GPIO state."""
        with self._lock:
            if pins is None:
                self._state.clear()
            else:
                # Handle single pin or list of pins
                if isinstance(pins, int):
                    pins = [pins]
                for pin in pins:
                    self._state.pop(pin, None)
    
    def reset(self):
        """Reset all GPIO state (test-specific method)."""
        with self._lock:
            self._state.clear()
            self._mode = None
            self._warnings = False


def create_test_gpio():
    """Factory function to create isolated GPIO instance for tests."""
    return TestGPIO()


# Example usage in tests:
# 
# @pytest.fixture
# def isolated_gpio(monkeypatch):
#     """Create isolated GPIO instance for test."""
#     test_gpio = create_test_gpio()
#     
#     # Monkey-patch the GPIO module in trigger
#     import gpio_trigger.trigger
#     monkeypatch.setattr(gpio_trigger.trigger, 'GPIO', test_gpio)
#     
#     yield test_gpio
#     
#     # Cleanup
#     test_gpio.cleanup()