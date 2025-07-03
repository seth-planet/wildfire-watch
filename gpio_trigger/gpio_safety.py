#!/usr/bin/env python3.12
"""
GPIO Safety Wrapper Module
Provides thread-safe, verified GPIO operations with comprehensive error handling

This module implements the safety recommendations from the error analysis:
1. All GPIO operations wrapped with try-except and retry logic
2. Read-after-write verification for critical pins
3. Thread-safe operations with proper locking
4. Safe failure modes with best-effort shutdown
"""
import time
import logging
import threading
from typing import Optional, Dict, Any, Tuple
from enum import Enum, auto

logger = logging.getLogger(__name__)

# Custom exceptions for clear error handling
class HardwareError(Exception):
    """Base exception for hardware interaction failures."""
    pass

class GPIOVerificationError(HardwareError):
    """Raised when a GPIO pin state cannot be verified after setting."""
    pass

class GPIOTimeoutError(HardwareError):
    """Raised when GPIO operations timeout."""
    pass

class PinType(Enum):
    """Pin criticality levels for error handling decisions"""
    CRITICAL = auto()      # System won't work without (main valve, ignition)
    IMPORTANT = auto()     # Degraded operation (priming, refill)
    AUXILIARY = auto()     # Nice to have (RPM reduction)

class SafeGPIO:
    """Thread-safe GPIO wrapper with verification and error handling"""
    
    # Pin criticality mapping
    PIN_CRITICALITY = {
        'MAIN_VALVE': PinType.CRITICAL,
        'IGN_ON': PinType.CRITICAL,
        'IGN_START': PinType.CRITICAL,
        'PRIMING_VALVE': PinType.IMPORTANT,
        'REFILL_VALVE': PinType.IMPORTANT,
        'IGN_OFF': PinType.IMPORTANT,
        'RPM_REDUCE': PinType.AUXILIARY,
    }
    
    def __init__(self, gpio_module, simulation_mode: bool = False):
        """
        Initialize the safe GPIO wrapper
        
        Args:
            gpio_module: The GPIO module (RPi.GPIO or simulation)
            simulation_mode: Whether running in simulation mode
        """
        self.GPIO = gpio_module
        self.simulation_mode = simulation_mode
        self._lock = threading.RLock()
        self._pin_states: Dict[int, bool] = {}
        self._verification_enabled = not simulation_mode
        self._failure_counts: Dict[str, int] = {}
        
    def safe_write(self, pin: int, value: bool, pin_name: str = "", 
                   retries: int = 3, retry_delay_ms: int = 50) -> bool:
        """
        Safely write a value to a GPIO pin with verification and retries.
        
        Args:
            pin: GPIO pin number
            value: Desired state (True=HIGH, False=LOW)
            pin_name: Configuration name for logging and criticality
            retries: Number of retry attempts
            retry_delay_ms: Delay between retries in milliseconds
            
        Returns:
            bool: True if successful, False if all retries failed
            
        Raises:
            GPIOVerificationError: For critical pins that fail verification
        """
        last_exception = None
        criticality = self.PIN_CRITICALITY.get(pin_name, PinType.AUXILIARY)
        
        with self._lock:
            for attempt in range(retries + 1):
                try:
                    # Perform the hardware operation
                    self.GPIO.output(pin, self.GPIO.HIGH if value else self.GPIO.LOW)
                    
                    # Store intended state
                    self._pin_states[pin] = value
                    
                    # Verify the result for critical/important pins
                    if self._verification_enabled and criticality != PinType.AUXILIARY:
                        time.sleep(0.01)  # Small delay for hardware to settle
                        
                        # Read back the value
                        read_back_value = bool(self.GPIO.input(pin))
                        
                        if read_back_value == value:
                            # Success!
                            logger.debug(f"GPIO {pin_name} (pin {pin}) set to {value} successfully")
                            self._failure_counts[pin_name] = 0  # Reset failure count
                            return True
                        
                        # Verification failed
                        raise GPIOVerificationError(
                            f"Verification failed for {pin_name} (pin {pin}). "
                            f"Wrote {value}, read back {read_back_value}."
                        )
                    else:
                        # No verification needed/possible
                        logger.debug(f"GPIO {pin_name} (pin {pin}) set to {value} (unverified)")
                        return True
                        
                except (IOError, GPIOVerificationError) as e:
                    last_exception = e
                    self._failure_counts[pin_name] = self._failure_counts.get(pin_name, 0) + 1
                    
                    if attempt < retries:
                        logger.warning(
                            f"GPIO {pin_name} attempt {attempt + 1}/{retries + 1} failed: {e}. "
                            f"Retrying in {retry_delay_ms}ms..."
                        )
                        time.sleep(retry_delay_ms / 1000)
                    else:
                        logger.error(
                            f"GPIO {pin_name} failed after {retries + 1} attempts: {e}"
                        )
                except Exception as e:
                    # Catch any other exception and convert to HardwareError
                    last_exception = HardwareError(f"GPIO hardware failure: {e}")
                    self._failure_counts[pin_name] = self._failure_counts.get(pin_name, 0) + 1
                    
                    if attempt < retries:
                        logger.warning(
                            f"GPIO {pin_name} attempt {attempt + 1}/{retries + 1} failed: {e}. "
                            f"Retrying in {retry_delay_ms}ms..."
                        )
                        time.sleep(retry_delay_ms / 1000)
                    else:
                        logger.error(
                            f"GPIO {pin_name} failed after {retries + 1} attempts: {e}"
                        )
            
            # All retries exhausted
            if criticality == PinType.CRITICAL:
                # Critical pins must work - raise exception
                raise last_exception
            else:
                # Non-critical pins - log and continue
                logger.error(f"Non-critical GPIO {pin_name} operation failed but continuing")
                return False
    
    def safe_read(self, pin: int, pin_name: str = "") -> Optional[bool]:
        """
        Safely read a GPIO pin value with error handling.
        
        Args:
            pin: GPIO pin number
            pin_name: Configuration name for logging
            
        Returns:
            bool: Pin state, or None if read failed
        """
        with self._lock:
            try:
                value = bool(self.GPIO.input(pin))
                logger.debug(f"GPIO {pin_name} (pin {pin}) read as {value}")
                return value
            except Exception as e:
                logger.error(f"Failed to read GPIO {pin_name} (pin {pin}): {e}")
                return None
    
    def emergency_all_off(self, pin_config: Dict[str, int]) -> Dict[str, bool]:
        """
        Best-effort emergency shutdown of all pins.
        
        Args:
            pin_config: Dictionary mapping pin names to pin numbers
            
        Returns:
            Dict[str, bool]: Results of each pin operation
        """
        results = {}
        priority_order = [
            # Most critical first - stop engine
            'IGN_ON', 'IGN_START',
            # Then close valves to prevent water waste
            'MAIN_VALVE', 'REFILL_VALVE', 'PRIMING_VALVE',
            # Finally auxiliary controls
            'RPM_REDUCE', 'IGN_OFF'
        ]
        
        with self._lock:
            logger.critical("EMERGENCY SHUTDOWN - Attempting to turn off all pins")
            
            # Process in priority order
            for pin_name in priority_order:
                if pin_name in pin_config:
                    pin = pin_config[pin_name]
                    try:
                        # Special handling for IGN_OFF (active high to stop)
                        value = True if pin_name == 'IGN_OFF' else False
                        
                        # Single attempt only in emergency
                        self.GPIO.output(pin, self.GPIO.HIGH if value else self.GPIO.LOW)
                        self._pin_states[pin] = value
                        results[pin_name] = True
                        logger.info(f"Emergency: {pin_name} set to {value}")
                        
                    except Exception as e:
                        results[pin_name] = False
                        logger.error(f"Emergency: Failed to control {pin_name}: {e}")
                        # Continue with other pins regardless
            
            # Pulse IGN_OFF if it was set high
            if 'IGN_OFF' in results and results['IGN_OFF']:
                try:
                    time.sleep(1.0)  # Hold for 1 second
                    self.GPIO.output(pin_config['IGN_OFF'], self.GPIO.LOW)
                    logger.info("Emergency: IGN_OFF pulse completed")
                except Exception as e:
                    logger.error(f"Emergency: Failed to complete IGN_OFF pulse: {e}")
            
            logger.critical(f"EMERGENCY SHUTDOWN complete. Results: {results}")
            return results
    
    def get_failure_stats(self) -> Dict[str, int]:
        """Get failure counts for monitoring"""
        with self._lock:
            return self._failure_counts.copy()
    
    def reset_failure_stats(self):
        """Reset failure statistics"""
        with self._lock:
            self._failure_counts.clear()

class ThreadSafeStateMachine:
    """Base class for thread-safe state machines"""
    
    def __init__(self):
        self._state = None
        self._state_lock = threading.RLock()
        self._transition_callbacks = []
        
    def get_state(self):
        """Get current state atomically"""
        with self._state_lock:
            return self._state
    
    def transition_to(self, new_state, validation_func=None, action_func=None):
        """
        Atomic state transition with optional validation and action.
        
        Args:
            new_state: Target state
            validation_func: Optional function to validate transition
            action_func: Optional function to execute during transition
            
        Returns:
            bool: True if transition succeeded
        """
        with self._state_lock:
            old_state = self._state
            
            # Validate transition if validator provided
            if validation_func and not validation_func(old_state, new_state):
                logger.warning(f"State transition {old_state} -> {new_state} blocked by validator")
                return False
            
            # Execute action if provided
            if action_func:
                try:
                    if not action_func():
                        logger.error(f"State transition {old_state} -> {new_state} action failed")
                        return False
                except Exception as e:
                    logger.error(f"State transition {old_state} -> {new_state} action raised: {e}")
                    return False
            
            # Perform transition
            self._state = new_state
            logger.info(f"State transition: {old_state} -> {new_state}")
            
            # Notify callbacks
            for callback in self._transition_callbacks:
                try:
                    callback(old_state, new_state)
                except Exception as e:
                    logger.error(f"State transition callback error: {e}")
            
            return True
    
    def add_transition_callback(self, callback):
        """Add a callback for state transitions"""
        self._transition_callbacks.append(callback)

class SafeTimerManager:
    """Thread-safe timer management with automatic cleanup"""
    
    def __init__(self):
        self._timers: Dict[str, threading.Timer] = {}
        self._lock = threading.RLock()
        
    def schedule(self, name: str, func, delay: float, error_handler=None):
        """Schedule a timer, canceling any existing timer with same name"""
        with self._lock:
            self.cancel(name)
            
            def wrapped_func():
                with self._lock:
                    self._timers.pop(name, None)
                try:
                    func()
                except Exception as e:
                    logger.error(f"Timer '{name}' failed: {e}")
                    if error_handler:
                        try:
                            error_handler(name, e)
                        except Exception as handler_error:
                            logger.error(f"Timer error handler failed: {handler_error}")
            
            timer = threading.Timer(delay, wrapped_func)
            timer.daemon = True
            timer.start()
            self._timers[name] = timer
            logger.debug(f"Scheduled timer '{name}' for {delay}s")
    
    def cancel(self, name: str) -> bool:
        """Cancel a timer if it exists"""
        with self._lock:
            timer = self._timers.pop(name, None)
            if timer and timer.is_alive():
                timer.cancel()
                logger.debug(f"Cancelled timer '{name}'")
                return True
            return False
    
    def cancel_all(self):
        """Cancel all active timers"""
        with self._lock:
            for name in list(self._timers.keys()):
                self.cancel(name)
    
    def get_active_timers(self) -> list:
        """Get list of active timer names"""
        with self._lock:
            return [name for name, timer in self._timers.items() if timer.is_alive()]