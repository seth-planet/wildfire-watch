#!/usr/bin/env python3.12
"""GPIO simulation module for testing on non-Raspberry Pi hardware.

This module provides a simulated GPIO interface that mimics the RPi.GPIO API,
allowing tests and development to run on systems without actual GPIO hardware.
The simulation maintains internal state for all pins and provides thread-safe
operations.

Usage:
    from utils.gpio_simulation import SimulatedGPIO
    GPIO = SimulatedGPIO()
    
    # Use like RPi.GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(18, GPIO.OUT)
    GPIO.output(18, GPIO.HIGH)
"""

import threading


class SimulatedGPIO:
    """Simulated GPIO module for testing on non-Pi hardware.
    
    This class mimics the RPi.GPIO interface to allow tests to run
    without actual hardware. It maintains internal state for all pins
    and provides thread-safe operations.
    """
    
    # Constants matching RPi.GPIO
    OUT = 0
    IN = 1
    HIGH = 1
    LOW = 0
    PUD_UP = 22
    PUD_DOWN = 21
    PUD_OFF = 20
    BCM = 11
    BOARD = 10
    RISING = 31
    FALLING = 32
    BOTH = 33
    
    def __init__(self):
        """Initialize the simulated GPIO module."""
        self._lock = threading.RLock()
        self._state = {}  # Pin states (HIGH/LOW)
        self._mode = {}   # Pin modes (IN/OUT)
        self._pull = {}   # Pull-up/down resistors
        self._warnings = True
        self._cleanup_done = False
        self._edge_callbacks = {}  # Edge detection callbacks
        self._numbering_mode = None
        
    def setmode(self, mode):
        """Set the pin numbering mode."""
        with self._lock:
            self._numbering_mode = mode
            
    def setwarnings(self, warnings):
        """Enable/disable warnings."""
        self._warnings = warnings
        
    def setup(self, channel, direction, initial=None, pull_up_down=None):
        """Setup a GPIO channel.
        
        Args:
            channel: Pin number or list of pin numbers
            direction: IN or OUT
            initial: Initial value for output pins (HIGH/LOW)
            pull_up_down: Pull resistor setting (PUD_UP/PUD_DOWN/PUD_OFF)
        """
        with self._lock:
            # Handle single pin or list of pins
            channels = channel if isinstance(channel, (list, tuple)) else [channel]
            
            for ch in channels:
                self._mode[ch] = direction
                
                if direction == self.OUT:
                    # Set initial value for output pins
                    self._state[ch] = initial if initial is not None else self.LOW
                else:
                    # Input pins default based on pull resistor
                    if pull_up_down == self.PUD_UP:
                        self._state[ch] = self.HIGH
                        self._pull[ch] = self.PUD_UP
                    elif pull_up_down == self.PUD_DOWN:
                        self._state[ch] = self.LOW
                        self._pull[ch] = self.PUD_DOWN
                    else:
                        self._state[ch] = self.LOW  # Default to LOW
                        self._pull[ch] = self.PUD_OFF
                        
    def output(self, channel, value):
        """Set output value for a channel.
        
        Args:
            channel: Pin number or list of pin numbers
            value: HIGH/LOW or list of values
        """
        with self._lock:
            # Handle single pin or list of pins
            channels = channel if isinstance(channel, (list, tuple)) else [channel]
            values = value if isinstance(value, (list, tuple)) else [value] * len(channels)
            
            for ch, val in zip(channels, values):
                if ch in self._mode and self._mode[ch] == self.OUT:
                    old_value = self._state.get(ch, self.LOW)
                    self._state[ch] = val
                    
                    # Trigger edge detection callbacks if value changed
                    if old_value != val and ch in self._edge_callbacks:
                        edge_type = self.RISING if val > old_value else self.FALLING
                        for callback_info in self._edge_callbacks[ch]:
                            if callback_info['edge'] in (edge_type, self.BOTH):
                                # Run callback in separate thread like real GPIO
                                threading.Thread(
                                    target=callback_info['callback'],
                                    args=(ch,),
                                    daemon=True
                                ).start()
                                
    def input(self, channel):
        """Read input value from a channel.
        
        Args:
            channel: Pin number
            
        Returns:
            HIGH or LOW
        """
        with self._lock:
            return self._state.get(channel, self.LOW)
            
    def cleanup(self, channel=None):
        """Cleanup GPIO resources.
        
        Args:
            channel: Specific channel(s) to cleanup, or None for all
        """
        with self._lock:
            if channel is None:
                # Cleanup all channels
                self._state.clear()
                self._mode.clear()
                self._pull.clear()
                self._edge_callbacks.clear()
                self._cleanup_done = True
            else:
                # Cleanup specific channels
                channels = channel if isinstance(channel, (list, tuple)) else [channel]
                for ch in channels:
                    self._state.pop(ch, None)
                    self._mode.pop(ch, None)
                    self._pull.pop(ch, None)
                    self._edge_callbacks.pop(ch, None)
                    
    def add_event_detect(self, channel, edge, callback=None, bouncetime=None):
        """Add edge detection to a channel.
        
        Args:
            channel: Pin number
            edge: RISING, FALLING, or BOTH
            callback: Function to call on edge detection
            bouncetime: Debounce time in milliseconds
        """
        with self._lock:
            if channel not in self._edge_callbacks:
                self._edge_callbacks[channel] = []
                
            if callback:
                self._edge_callbacks[channel].append({
                    'edge': edge,
                    'callback': callback,
                    'bouncetime': bouncetime
                })
                
    def remove_event_detect(self, channel):
        """Remove edge detection from a channel.
        
        Args:
            channel: Pin number
        """
        with self._lock:
            self._edge_callbacks.pop(channel, None)
            
