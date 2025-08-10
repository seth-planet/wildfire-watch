"""
Safe logging utilities for test suite to prevent I/O errors during parallel execution.

This module now imports from the unified safe_logging module in utils/ to avoid
code duplication and ensure consistency across production and test code.
"""
import os
import sys

# Add parent directory to path to import from utils
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import everything from the unified safe logging module
from utils.safe_logging import (
    # Core functions
    safe_log,
    check_logger_health,
    register_logger_for_cleanup,
    cleanup_all_registered_loggers,
    
    # Mixins and classes
    SafeLoggingMixin,
    LoggerGuard,
    SafeStreamHandler,
    SafeNullHandler,
    
    # Test-specific functions
    get_safe_logger,
    cleanup_test_logging,
    disable_problem_loggers,
    
    # Test-specific classes
    TestLoggerManager,
)

# Re-export safe_log_for_class for convenience
from utils.safe_logging import safe_log_for_class

# Ensure disable_problem_loggers is called for tests
disable_problem_loggers()