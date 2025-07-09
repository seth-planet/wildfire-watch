#!/usr/bin/env python3.12
"""Thread management utilities for Wildfire Watch services.

This module provides thread-safe timer management and background thread
coordination to prevent timer leaks and ensure proper cleanup.
"""

import threading
import logging
import time
from typing import Dict, Callable, Optional, List, Set
from contextlib import contextmanager


class SafeTimerManager:
    """Thread-safe timer management with automatic cleanup.
    
    Prevents timer leaks and ensures proper cleanup on shutdown.
    Provides a centralized way to manage all timers in a service.
    
    Attributes:
        logger: Logger instance for this manager
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize timer manager.
        
        Args:
            logger: Optional logger instance
        """
        self._timers: Dict[str, threading.Timer] = {}
        self._lock = threading.Lock()
        self.logger = logger or logging.getLogger(__name__)
        self._shutdown = False
        
    def schedule(self, name: str, func: Callable, delay: float,
                error_handler: Optional[Callable[[str, Exception], None]] = None) -> None:
        """Schedule a timer with automatic cleanup.
        
        Args:
            name: Unique name for this timer
            func: Function to call when timer expires
            delay: Delay in seconds before calling function
            error_handler: Optional error handler (name, exception) -> None
        """
        with self._lock:
            # Check shutdown status inside lock to prevent race condition
            if self._shutdown:
                self.logger.debug(f"Ignoring timer schedule '{name}' - manager shutting down")
                return
            
            # Cancel existing timer with same name (now inside lock)
            timer = self._timers.pop(name, None)
            if timer and timer.is_alive():
                timer.cancel()
                self.logger.debug(f"Cancelled existing timer '{name}'")
            
            def wrapped_func():
                """Wrapper that handles cleanup and errors."""
                # Remove from active timers
                with self._lock:
                    self._timers.pop(name, None)
                
                # Execute function with error handling
                try:
                    func()
                except Exception as e:
                    self.logger.error(f"Timer '{name}' failed: {e}")
                    if error_handler:
                        try:
                            error_handler(name, e)
                        except Exception as handler_error:
                            self.logger.error(f"Error handler for timer '{name}' failed: {handler_error}")
            
            # Create and start timer
            timer = threading.Timer(delay, wrapped_func)
            timer.daemon = True
            timer.start()
            self._timers[name] = timer
            self.logger.debug(f"Scheduled timer '{name}' for {delay}s")
    
    def cancel(self, name: str) -> bool:
        """Cancel a timer if it exists.
        
        Args:
            name: Name of timer to cancel
            
        Returns:
            True if timer was cancelled, False if not found
        """
        with self._lock:
            timer = self._timers.pop(name, None)
            if timer and timer.is_alive():
                timer.cancel()
                self.logger.debug(f"Cancelled timer '{name}'")
                return True
            return False
    
    def cancel_all(self) -> int:
        """Cancel all active timers.
        
        Returns:
            Number of timers cancelled
        """
        with self._lock:
            cancelled = 0
            for name, timer in self._timers.items():
                if timer.is_alive():
                    timer.cancel()
                    cancelled += 1
            self._timers.clear()
            
        if cancelled > 0:
            self.logger.info(f"Cancelled {cancelled} active timers")
        return cancelled
    
    def has_timer(self, name: str) -> bool:
        """Check if a timer is scheduled and active.
        
        Args:
            name: Timer name to check
            
        Returns:
            True if timer exists and is active
        """
        with self._lock:
            timer = self._timers.get(name)
            return timer is not None and timer.is_alive()
    
    def get_active_timers(self) -> List[str]:
        """Get list of active timer names.
        
        Returns:
            List of timer names that are currently active
        """
        with self._lock:
            return [name for name, timer in self._timers.items() 
                   if timer.is_alive()]
    
    def shutdown(self) -> None:
        """Shutdown manager and cancel all timers."""
        self._shutdown = True
        
        # Get list of active timers before cancelling
        active_timers = []
        with self._lock:
            active_timers = [(name, timer) for name, timer in self._timers.items() if timer.is_alive()]
        
        # Cancel all timers
        self.cancel_all()
        
        # Wait for timers to finish (with timeout to prevent hanging)
        for name, timer in active_timers:
            try:
                timer.join(timeout=2.0)  # 2 second timeout per timer
                if timer.is_alive():
                    self.logger.warning(f"Timer '{name}' did not finish within timeout")
            except Exception as e:
                self.logger.error(f"Error joining timer '{name}': {e}")
        
        self.logger.debug("Timer manager shutdown complete")


class ThreadSafeService:
    """Base class for services with background threads.
    
    Provides thread management, graceful shutdown, and state coordination.
    """
    
    def __init__(self, service_name: str, logger: Optional[logging.Logger] = None):
        """Initialize thread-safe service.
        
        Args:
            service_name: Name of this service
            logger: Optional logger instance
        """
        self.service_name = service_name
        self.logger = logger or logging.getLogger(service_name)
        
        # Thread management
        self._threads: Dict[str, threading.Thread] = {}
        self._thread_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        
        # Timer management
        self.timer_manager = SafeTimerManager(self.logger)
        
        # Service state
        self._state_lock = threading.Lock()
        self._state = "initialized"
        
    def start_thread(self, name: str, target: Callable, daemon: bool = True) -> None:
        """Start a managed background thread.
        
        Args:
            name: Unique name for this thread
            target: Function to run in thread
            daemon: Whether thread should be daemon
        """
        # Stop existing thread with same name (before acquiring lock)
        self.stop_thread(name)
        
        with self._thread_lock:
            # Create wrapper that logs exceptions
            def thread_wrapper():
                try:
                    self.logger.debug(f"Thread '{name}' started")
                    target()
                    self.logger.debug(f"Thread '{name}' completed")
                except Exception as e:
                    self.logger.error(f"Thread '{name}' crashed: {e}", exc_info=True)
                finally:
                    with self._thread_lock:
                        self._threads.pop(name, None)
            
            # Create and start thread
            thread = threading.Thread(target=thread_wrapper, name=f"{self.service_name}-{name}")
            thread.daemon = daemon
            thread.start()
            self._threads[name] = thread
            
    def stop_thread(self, name: str, timeout: float = 5.0) -> bool:
        """Stop a managed thread.
        
        Args:
            name: Name of thread to stop
            timeout: Maximum time to wait for thread to stop
            
        Returns:
            True if thread stopped, False if timeout
        """
        with self._thread_lock:
            thread = self._threads.pop(name, None)
            
        if thread and thread.is_alive():
            # Signal shutdown to thread
            self._shutdown_event.set()
            
            # Wait for thread to stop
            thread.join(timeout)
            
            if thread.is_alive():
                self.logger.warning(f"Thread '{name}' did not stop within {timeout}s")
                return False
            else:
                self.logger.debug(f"Thread '{name}' stopped")
                return True
        
        return True
    
    def stop_all_threads(self, timeout: float = 10.0) -> int:
        """Stop all managed threads.
        
        Args:
            timeout: Maximum time to wait for all threads
            
        Returns:
            Number of threads that failed to stop
        """
        # Signal shutdown to all threads
        self._shutdown_event.set()
        
        # Get list of threads
        with self._thread_lock:
            threads = list(self._threads.items())
        
        # Wait for threads to stop
        failed = 0
        start_time = time.time()
        for name, thread in threads:
            remaining = max(0, timeout - (time.time() - start_time))
            if thread.is_alive():
                thread.join(remaining)
                if thread.is_alive():
                    self.logger.warning(f"Thread '{name}' still running after shutdown")
                    failed += 1
        
        # Clear thread dict
        with self._thread_lock:
            self._threads.clear()
        
        return failed
    
    @property
    def is_shutting_down(self) -> bool:
        """Check if service is shutting down."""
        return self._shutdown_event.is_set()
    
    def wait_for_shutdown(self, timeout: Optional[float] = None) -> bool:
        """Wait for shutdown signal.
        
        Args:
            timeout: Maximum time to wait
            
        Returns:
            True if shutdown signaled, False if timeout
        """
        return self._shutdown_event.wait(timeout)
    
    @contextmanager
    def state_lock(self):
        """Context manager for thread-safe state access."""
        self._state_lock.acquire()
        try:
            yield
        finally:
            self._state_lock.release()
    
    def get_state(self) -> str:
        """Get current service state thread-safely."""
        with self._state_lock:
            return self._state
    
    def set_state(self, new_state: str) -> None:
        """Set service state thread-safely."""
        with self._state_lock:
            old_state = self._state
            self._state = new_state
            self.logger.info(f"State changed: {old_state} -> {new_state}")
    
    def shutdown(self) -> None:
        """Gracefully shutdown the service."""
        self.logger.info(f"Shutting down {self.service_name}")
        
        # Set shutdown flag
        self._shutdown_event.set()
        
        # Cancel all timers
        self.timer_manager.shutdown()
        
        # Stop all threads
        failed = self.stop_all_threads()
        if failed > 0:
            self.logger.warning(f"{failed} threads did not stop cleanly")
        
        self.logger.info(f"{self.service_name} shutdown complete")


class BackgroundTaskRunner:
    """Manages periodic background tasks with error recovery."""
    
    def __init__(self, name: str, interval: float, task: Callable,
                 logger: Optional[logging.Logger] = None):
        """Initialize background task runner.
        
        Args:
            name: Name of this task
            interval: Seconds between task executions
            task: Function to execute periodically
            logger: Optional logger instance
        """
        self.name = name
        self.interval = interval
        self.task = task
        self.logger = logger or logging.getLogger(name)
        
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._error_count = 0
        self._max_errors = 10
        
    def start(self) -> None:
        """Start the background task."""
        if self._thread and self._thread.is_alive():
            self.logger.warning(f"Task '{self.name}' is already running")
            return
        
        self._stop_event.clear()
        self._error_count = 0
        
        self._thread = threading.Thread(target=self._run, name=f"task-{self.name}")
        self._thread.daemon = True
        self._thread.start()
        self.logger.info(f"Started background task '{self.name}'")
    
    def stop(self, timeout: float = 5.0) -> bool:
        """Stop the background task.
        
        Args:
            timeout: Maximum time to wait for task to stop
            
        Returns:
            True if stopped, False if timeout
        """
        if not self._thread or not self._thread.is_alive():
            return True
        
        self.logger.info(f"Stopping background task '{self.name}'")
        self._stop_event.set()
        
        self._thread.join(timeout)
        
        if self._thread.is_alive():
            self.logger.warning(f"Task '{self.name}' did not stop within {timeout}s")
            return False
        
        self.logger.info(f"Background task '{self.name}' stopped")
        return True
    
    def _run(self) -> None:
        """Main task loop with error recovery."""
        while not self._stop_event.is_set():
            try:
                # Execute task
                self.task()
                
                # Reset error count on success
                self._error_count = 0
                
            except Exception as e:
                self._error_count += 1
                self.logger.error(f"Task '{self.name}' error #{self._error_count}: {e}")
                
                # Stop if too many errors
                if self._error_count >= self._max_errors:
                    self.logger.error(f"Task '{self.name}' stopping after {self._error_count} errors")
                    break
                
                # Brief delay before retry
                self._stop_event.wait(min(self._error_count, 10))
            
            # Wait for next execution
            if not self._stop_event.wait(self.interval):
                continue