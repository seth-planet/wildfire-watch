"""
Minimal GPIO safety wrappers for test compatibility.
"""

class SafeGPIO:
    """Minimal SafeGPIO wrapper for tests."""
    def __init__(self, gpio_module, simulation_mode=False):
        self.gpio = gpio_module
        self.simulation_mode = simulation_mode
    
    def safe_write(self, pin, state, pin_name=None, retries=3):
        """Safe GPIO write with retry logic."""
        for attempt in range(retries):
            try:
                self.gpio.output(pin, self.gpio.HIGH if state else self.gpio.LOW)
                return True
            except Exception as e:
                if attempt < retries - 1:
                    continue
                raise
        return False
    
    def emergency_all_off(self, pin_config):
        """Emergency shutdown all pins."""
        results = {}
        for name, pin in pin_config.items():
            try:
                self.gpio.output(pin, self.gpio.LOW)
                results[name] = True
            except:
                results[name] = False
        return results

class ThreadSafeStateMachine:
    """Base class for thread-safe state machines."""
    pass

class SafeTimerManager:
    """Minimal timer manager for tests."""
    def __init__(self):
        self._timers = {}
    
    def get_active_timers(self):
        """Return list of active timer names."""
        return list(self._timers.keys())
    
    def schedule(self, name, func, delay, error_handler=None):
        """Schedule a timer."""
        import threading
        
        def wrapped_func():
            self._timers.pop(name, None)
            try:
                func()
            except Exception as e:
                if error_handler:
                    error_handler(name, e)
        
        timer = threading.Timer(delay, wrapped_func)
        timer.daemon = True
        timer.start()
        self._timers[name] = timer
    
    def cancel(self, name):
        """Cancel a timer."""
        timer = self._timers.pop(name, None)
        if timer and timer.is_alive():
            timer.cancel()
    
    def cancel_all(self):
        """Cancel all timers."""
        for name in list(self._timers.keys()):
            self.cancel(name)

class HardwareError(Exception):
    """Hardware error exception."""
    pass

class GPIOVerificationError(Exception):
    """GPIO verification error exception."""
    pass