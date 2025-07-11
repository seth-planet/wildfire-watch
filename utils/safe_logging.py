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
from typing import Optional, Dict


# ============================================================================
# Core Safe Logging Functionality (Production & Tests)
# ============================================================================

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
    try:
        # Check if logger exists and has handlers
        if not logger or not hasattr(logger, 'handlers') or not logger.handlers:
            return
            
        # Check if logger is disabled
        if hasattr(logger, 'disabled') and logger.disabled:
            return
            
        # Check if we have at least one usable handler
        usable_handler_found = False
        for handler in logger.handlers:
            if hasattr(handler, 'stream'):
                # StreamHandler and FileHandler
                stream = getattr(handler, 'stream', None)
                if stream and hasattr(stream, 'closed') and not stream.closed:
                    usable_handler_found = True
                    break
            else:
                # Non-stream handlers (SocketHandler, QueueHandler, etc.) are assumed usable
                usable_handler_found = True
                break
                
        if not usable_handler_found:
            return
                    
        # Get the log method and check it exists
        log_method = getattr(logger, level.lower(), None)
        if log_method and callable(log_method):
            log_method(message, exc_info=exc_info)
            
    except (ValueError, AttributeError, OSError, RuntimeError, KeyError):
        # Silently ignore all logging errors during shutdown
        # RuntimeError can occur if dict changes during iteration
        # KeyError can occur if logger dict is modified during shutdown
        pass


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
        
    # Check if any handler is closed
    for handler in logger.handlers:
        if hasattr(handler, 'stream') and hasattr(handler.stream, 'closed'):
            if handler.stream.closed:
                return False
                
    return True


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