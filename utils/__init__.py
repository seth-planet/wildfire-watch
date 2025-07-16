"""
Utility modules for Wildfire Watch services
"""

# Export safe logging utilities
from .safe_logging import (
    safe_log, SafeLoggingMixin, safe_log_for_class, check_logger_health,
    # Test utilities
    get_safe_logger, cleanup_test_logging, disable_problem_loggers
)

# Export logging configuration utilities
from .logging_config import (
    setup_logging, get_logger, configure_module_logging, 
    apply_common_filters, setup_file_logging,
    DEFAULT_LOG_FORMAT, HYPHEN_LOG_FORMAT
)

__all__ = [
    # Safe logging utilities
    'safe_log', 'SafeLoggingMixin', 'safe_log_for_class', 'check_logger_health',
    'get_safe_logger', 'cleanup_test_logging', 'disable_problem_loggers',
    # Logging configuration
    'setup_logging', 'get_logger', 'configure_module_logging',
    'apply_common_filters', 'setup_file_logging',
    'DEFAULT_LOG_FORMAT', 'HYPHEN_LOG_FORMAT'
]

# Export GPIO test helpers (only when running tests)
import os
if os.environ.get('PYTEST_CURRENT_TEST'):
    from .gpio_test_helpers import (
        wait_for_state, wait_for_any_state, verify_all_pins_low,
        verify_pin_states, assert_pin_state, setup_sensor_pin,
        simulate_button_press, simulate_sensor_state, PinMonitor,
        wait_for_pin_state, get_all_pin_states, verify_safe_shutdown_state,
        trigger_hardware_validation_failure, wait_for_stable_state
    )
    
    __all__.extend([
        'wait_for_state', 'wait_for_any_state', 'verify_all_pins_low',
        'verify_pin_states', 'assert_pin_state', 'setup_sensor_pin',
        'simulate_button_press', 'simulate_sensor_state', 'PinMonitor',
        'wait_for_pin_state', 'get_all_pin_states', 'verify_safe_shutdown_state',
        'trigger_hardware_validation_failure', 'wait_for_stable_state'
    ])