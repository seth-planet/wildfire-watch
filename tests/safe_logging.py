#!/usr/bin/env python3.12
"""Safe logging configuration for tests to prevent I/O on closed file errors.

This module provides a safe logging setup that prevents I/O operations on closed
file descriptors during pytest teardown, especially in parallel test execution.
"""

import logging
import sys
import io
import threading
from typing import Optional, Dict


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


# Auto-disable problem loggers on import
disable_problem_loggers()