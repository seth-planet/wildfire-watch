#!/usr/bin/env python3.12
"""Safe logging utilities for Wildfire Watch services and tests.

This module provides utilities for safe logging during shutdown and test scenarios
where loggers or their handlers might be closed or unavailable. This helps prevent
"I/O operation on closed file" errors that can occur during parallel test execution
or service shutdown.

This module combines production service utilities with test-specific utilities
to provide a single source of truth for all safe logging functionality.
"""

import logging
import sys
import threading
import weakref
from typing import Optional, Dict, Set, Callable


# ============================================================================
# Core Safe Logging Functionality (Production & Tests)
# ============================================================================

# Thread-local storage to prevent recursive logging during shutdown
_thread_local = threading.local()


def _is_handler_usable(handler: logging.Handler) -> bool:
    """Check if a logging handler is usable and not closed.
    
    Args:
        handler: The logging handler to check
        
    Returns:
        True if the handler can be safely used, False otherwise
    """
    try:
        # StreamHandler and FileHandler
        if hasattr(handler, 'stream'):
            stream = getattr(handler, 'stream', None)
            if stream is None:
                return False
            if hasattr(stream, 'closed') and stream.closed:
                return False
            # Check if stream is writable
            if hasattr(stream, 'writable') and not stream.writable():
                return False
            # Check for specific stream types
            if stream in (sys.stdout, sys.stderr):
                # Always consider stdout/stderr usable unless explicitly closed
                if hasattr(stream, 'closed') and stream.closed:
                    return False
            return True
            
        # SocketHandler
        elif hasattr(handler, 'sock'):
            sock = getattr(handler, 'sock', None)
            if sock is None:
                return False
            # Check if socket is closed
            try:
                # This will fail if socket is closed
                sock.getpeername()
                return True
            except:
                return False
                
        # SMTPHandler
        elif hasattr(handler, 'mailhost'):
            # SMTP handlers are generally stateless, consider usable
            return True
            
        # SysLogHandler
        elif hasattr(handler, 'socket'):
            socket = getattr(handler, 'socket', None)
            if socket is None:
                # Unix socket syslog, consider usable
                return True
            # Network syslog, check socket
            try:
                socket.getpeername()
                return True
            except:
                return False
                
        # QueueHandler
        elif hasattr(handler, 'queue'):
            queue = getattr(handler, 'queue', None)
            if queue is None:
                return False
            # Check if queue is full (non-blocking check)
            try:
                return not queue.full()
            except:
                return True  # If we can't check, assume usable
                
        # MemoryHandler
        elif hasattr(handler, 'buffer'):
            # Memory handlers are generally always usable
            return True
            
        # NullHandler
        elif isinstance(handler, logging.NullHandler):
            return True
            
        # HTTPHandler
        elif hasattr(handler, 'host'):
            # HTTP handlers are stateless, consider usable
            return True
            
        # Unknown handler type - assume usable if it has emit method
        else:
            return hasattr(handler, 'emit') and callable(handler.emit)
            
    except Exception:
        # If any check fails, consider handler unusable
        return False


def safe_log(logger: Optional[logging.Logger], level: str, message: str, 
             exc_info: bool = False) -> None:
    """Safely log a message with comprehensive checks.
    
    This function performs safe logging by checking if the logger and its handlers
    are still available and not closed before attempting to log. This prevents
    exceptions during shutdown or test teardown.
    
    Args:
        logger: Logger instance to use (or None)
        level: Log level as string (e.g., 'info', 'debug', 'error')
        message: Message to log
        exc_info: Whether to include exception information
    """
    # Prevent recursive logging attempts during shutdown
    if getattr(_thread_local, 'in_safe_log', False):
        return
        
    _thread_local.in_safe_log = True
    
    try:
        # Check if logger exists and is not None
        if not logger:
            return
            
        # Verify logger is still a valid Logger instance
        if not isinstance(logger, logging.Logger):
            return
            
        # Check if logger has been disabled
        if hasattr(logger, 'disabled') and logger.disabled:
            return
            
        # Check if logging level would actually log this message
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        if not logger.isEnabledFor(numeric_level):
            return
            
        # Check if logger has handlers and they're not closed
        if not hasattr(logger, 'handlers') or not logger.handlers:
            return
            
        # Check each handler to ensure it's not closed
        usable_handler_found = False
        for handler in logger.handlers:
            try:
                if _is_handler_usable(handler):
                    usable_handler_found = True
                    break
            except Exception:
                # Handler check failed, skip this handler
                continue
                
        if not usable_handler_found:
            return
                    
        # Get the log method and check it exists
        log_method = getattr(logger, level.lower(), None)
        if log_method and callable(log_method):
            log_method(message, exc_info=exc_info)
            
    except (ValueError, AttributeError, OSError, RuntimeError, KeyError):
        # Silently ignore all logging errors during shutdown
        # ValueError: I/O operation on closed file
        # RuntimeError: dictionary changed size during iteration
        # KeyError: logger might have been removed from logging registry
        pass
    except Exception:
        # Catch any other unexpected exceptions during logging
        pass
    finally:
        _thread_local.in_safe_log = False


class SafeLoggingMixin:
    """Mixin class that provides safe logging capability to any class.
    
    Classes using this mixin should have a 'logger' attribute that is a
    logging.Logger instance. The mixin provides a _safe_log method that
    can be used throughout the class for safe logging.
    
    Example:
        class MyService(SafeLoggingMixin):
            def __init__(self):
                self.logger = logging.getLogger(__name__)
                
            def some_method(self):
                self._safe_log('info', 'This is a safe log message')
    """
    
    def _safe_log(self, level: str, message: str, exc_info: bool = False) -> None:
        """Safely log a message with comprehensive checks.
        
        Args:
            level: Log level as string (e.g., 'info', 'debug', 'error')
            message: Message to log
            exc_info: Whether to include exception information
        """
        logger = getattr(self, 'logger', None)
        safe_log(logger, level, message, exc_info)


def safe_log_for_class(obj: any, level: str, message: str, exc_info: bool = False) -> None:
    """Safely log a message for any object that might have a logger attribute.
    
    This is a convenience function for classes that don't inherit from SafeLoggingMixin
    but still need safe logging capabilities.
    
    Args:
        obj: Object that might have a 'logger' attribute
        level: Log level as string
        message: Message to log
        exc_info: Whether to include exception information
    """
    logger = getattr(obj, 'logger', None)
    safe_log(logger, level, message, exc_info)


def check_logger_health(logger: Optional[logging.Logger]) -> bool:
    """Check if a logger is healthy and can be used safely.
    
    Args:
        logger: Logger to check
        
    Returns:
        True if logger is safe to use, False otherwise
    """
    if not logger:
        return False
        
    if not hasattr(logger, 'handlers') or not logger.handlers:
        return False
        
    # Check if any handler is usable
    for handler in logger.handlers:
        try:
            if _is_handler_usable(handler):
                return True
        except Exception:
            continue
                
    return False


class LoggerGuard:
    """Context manager for safe logger state management.
    
    This context manager stores the state of a logger on entry and restores
    it on exit, ensuring that any modifications during the context are
    properly cleaned up.
    
    Example:
        with LoggerGuard('my_logger') as logger:
            # Use logger safely
            logger.info('Test message')
        # Logger state is restored here
    """
    
    def __init__(self, logger_name: str):
        """Initialize the guard with a logger name.
        
        Args:
            logger_name: Name of the logger to guard
        """
        self.logger_name = logger_name
        self.logger = None
        self.original_handlers = []
        self.original_level = None
        self.original_propagate = None
        self.original_disabled = None
        
    def __enter__(self) -> logging.Logger:
        """Enter the context and store logger state.
        
        Returns:
            The logger instance
        """
        self.logger = logging.getLogger(self.logger_name)
        
        # Store original state
        self.original_handlers = self.logger.handlers.copy()
        self.original_level = self.logger.level
        self.original_propagate = self.logger.propagate
        self.original_disabled = getattr(self.logger, 'disabled', False)
        
        return self.logger
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and restore logger state.
        
        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        if self.logger is None:
            return
            
        try:
            # Remove any handlers added during the context
            for handler in self.logger.handlers[:]:
                if handler not in self.original_handlers:
                    try:
                        self.logger.removeHandler(handler)
                        if hasattr(handler, 'close'):
                            handler.close()
                    except Exception:
                        pass
            
            # Restore original handlers
            self.logger.handlers.clear()
            for handler in self.original_handlers:
                try:
                    self.logger.addHandler(handler)
                except Exception:
                    pass
            
            # Restore other properties
            self.logger.setLevel(self.original_level)
            self.logger.propagate = self.original_propagate
            if hasattr(self.logger, 'disabled'):
                self.logger.disabled = self.original_disabled
                
        except Exception:
            # Silently ignore any errors during restoration
            pass


# Global registry for cleanup callbacks
_cleanup_registry: Set[weakref.ref] = set()
_cleanup_lock = threading.Lock()


def register_logger_for_cleanup(logger: logging.Logger, cleanup_callback: Optional[Callable] = None):
    """Register a logger for cleanup during shutdown.
    
    Args:
        logger: Logger to register
        cleanup_callback: Optional callback to call during cleanup
    """
    with _cleanup_lock:
        # Create weak reference to avoid keeping logger alive
        weak_logger = weakref.ref(logger, lambda ref: _cleanup_registry.discard(ref))
        _cleanup_registry.add(weak_logger)
        
        # Store cleanup callback if provided
        if cleanup_callback:
            # Store callback in logger itself (which weak ref points to)
            logger._cleanup_callback = cleanup_callback


def cleanup_all_registered_loggers():
    """Clean up all registered loggers.
    
    This should be called during test teardown or service shutdown.
    """
    with _cleanup_lock:
        # Copy set to avoid modification during iteration
        refs = list(_cleanup_registry)
        
        for weak_logger in refs:
            logger = weak_logger()
            if logger is not None:
                try:
                    # Call custom cleanup callback if provided
                    if hasattr(logger, '_cleanup_callback'):
                        logger._cleanup_callback(logger)
                    
                    # Standard cleanup
                    for handler in logger.handlers[:]:
                        try:
                            logger.removeHandler(handler)
                            if hasattr(handler, 'close'):
                                handler.close()
                        except Exception:
                            pass
                            
                    # Clear handlers
                    logger.handlers.clear()
                    
                    # Disable logger
                    logger.disabled = True
                    
                except Exception:
                    pass
        
        # Clear registry
        _cleanup_registry.clear()


# ============================================================================
# Test-Specific Safe Logging Components
# ============================================================================

class SafeStreamHandler(logging.StreamHandler):
    """Thread-safe stream handler that prevents I/O on closed files."""
    
    def __init__(self, stream=None):
        """Initialize with a safe stream."""
        if stream is None:
            stream = sys.stderr
        super().__init__(stream)
        self._lock = threading.RLock()
        self._closed = False
    
    def emit(self, record):
        """Emit a record, safely handling closed streams."""
        if self._closed:
            return
            
        with self._lock:
            try:
                if self.stream and hasattr(self.stream, 'closed') and self.stream.closed:
                    return
                super().emit(record)
            except (ValueError, OSError) as e:
                # Handle I/O operation on closed file
                if "I/O operation on closed file" in str(e):
                    self._closed = True
                    return
                # Other errors might be important
                pass
            except Exception:
                # Catch all other exceptions to prevent test failures
                self.handleError(record)
    
    def close(self):
        """Close the handler safely."""
        with self._lock:
            self._closed = True
            try:
                super().close()
            except Exception:
                pass


class SafeNullHandler(logging.Handler):
    """A handler that does nothing, for complete safety."""
    
    def emit(self, record):
        """Do nothing."""
        pass
    
    def handle(self, record):
        """Do nothing."""
        pass
    
    def createLock(self):
        """No lock needed."""
        self.lock = None


class TestLoggerManager:
    """Manages loggers for safe test execution."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the logger manager."""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._original_handlers: Dict[str, list] = {}
            self._test_handlers: Dict[str, logging.Handler] = {}
    
    def setup_safe_logging(self, logger_name: str, level: int = logging.INFO) -> logging.Logger:
        """Setup safe logging for a specific logger.
        
        Args:
            logger_name: Name of the logger to configure
            level: Logging level to set
            
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(logger_name)
        
        # Store original handlers
        if logger_name not in self._original_handlers:
            self._original_handlers[logger_name] = logger.handlers.copy()
        
        # Remove all existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Add safe handler
        safe_handler = SafeStreamHandler(sys.stderr)
        safe_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(safe_handler)
        logger.setLevel(level)
        
        # Store for cleanup
        self._test_handlers[logger_name] = safe_handler
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        return logger
    
    def cleanup_logger(self, logger_name: str):
        """Cleanup a logger after tests."""
        logger = logging.getLogger(logger_name)
        
        # Remove test handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            if isinstance(handler, SafeStreamHandler):
                handler.close()
        
        # Restore original handlers if any
        if logger_name in self._original_handlers:
            for handler in self._original_handlers[logger_name]:
                logger.addHandler(handler)
            del self._original_handlers[logger_name]
        
        # Remove from test handlers
        if logger_name in self._test_handlers:
            del self._test_handlers[logger_name]
    
    def cleanup_all(self):
        """Cleanup all managed loggers."""
        logger_names = list(self._test_handlers.keys())
        for logger_name in logger_names:
            self.cleanup_logger(logger_name)
    
    def disable_problem_loggers(self):
        """Disable known problematic loggers that cause I/O errors."""
        problem_loggers = [
            'super_gradients',
            'torch.distributed',
            'torch.nn.parallel',
            'matplotlib',
            'PIL',
            'urllib3',
            'docker',
            'paho.mqtt',
        ]
        
        for logger_name in problem_loggers:
            logger = logging.getLogger(logger_name)
            logger.handlers.clear()
            logger.addHandler(SafeNullHandler())
            logger.propagate = False


# ============================================================================
# Test Utility Functions
# ============================================================================

# Global instance
_manager = TestLoggerManager()


def get_safe_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a safe logger for testing.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Safe logger instance
    """
    return _manager.setup_safe_logging(name, level)


def cleanup_test_logging():
    """Cleanup all test logging."""
    _manager.cleanup_all()


def disable_problem_loggers():
    """Disable known problematic loggers."""
    _manager.disable_problem_loggers()


# Auto-disable problem loggers on import (for tests)
# This is safe because production services don't import these functions
if 'pytest' in sys.modules:
    disable_problem_loggers()