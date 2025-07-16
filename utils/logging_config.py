#!/usr/bin/env python3
"""Standardized logging configuration for Wildfire Watch services.

This module provides a centralized logging configuration to ensure consistent
logging across all microservices in the Wildfire Watch system.

Example:
    Basic usage in a service::

        from utils.logging_config import setup_logging
        
        logger = setup_logging("camera_detector")
        logger.info("Service started")

    Using with custom log level::

        # Set LOG_LEVEL=DEBUG in environment
        logger = setup_logging("gpio_trigger")
        logger.debug("Detailed debug information")

Attributes:
    DEFAULT_LOG_FORMAT: Standard logging format used across all services
    DEFAULT_LOG_LEVEL: Default logging level if not specified in environment
"""

import os
import sys
import logging
from typing import Optional, List
from logging import Logger, Handler, StreamHandler


# Standard logging format for all services
DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# Alternative format with hyphen separators (used by some services)
HYPHEN_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Default log level
DEFAULT_LOG_LEVEL = "INFO"


def setup_logging(
    service_name: str,
    log_level: Optional[str] = None,
    format_string: Optional[str] = None,
    force_unbuffered: bool = True,
    handlers: Optional[List[Handler]] = None
) -> Logger:
    """Set up standardized logging configuration for a service.
    
    This function configures logging with consistent formatting and behavior
    across all Wildfire Watch services. It reads the LOG_LEVEL from the
    environment if not explicitly provided and ensures proper output buffering
    for containerized environments.
    
    Args:
        service_name: Name of the service (e.g., "camera_detector", "gpio_trigger")
        log_level: Override log level. If None, reads from LOG_LEVEL env var
        format_string: Custom format string. If None, uses DEFAULT_LOG_FORMAT
        force_unbuffered: Force stdout to be unbuffered for real-time logs
        handlers: Custom handlers. If None, uses StreamHandler(sys.stdout)
    
    Returns:
        Logger: Configured logger instance for the service
        
    Note:
        The function automatically:
        - Reads LOG_LEVEL from environment (default: INFO)
        - Forces unbuffered output for Docker environments
        - Sets both root and service loggers to the same level
        - Uses stdout for compatibility with container logging
        
    Example:
        >>> logger = setup_logging("my_service")
        >>> logger.info("Service initialized")
        2024-01-15 10:30:45 [INFO] my_service: Service initialized
    """
    # Get log level from environment or parameter
    if log_level is None:
        log_level = os.environ.get('LOG_LEVEL', DEFAULT_LOG_LEVEL).upper()
    else:
        log_level = log_level.upper()
    
    # Validate log level
    numeric_level = getattr(logging, log_level, None)
    if not isinstance(numeric_level, int):
        print(f"Warning: Invalid log level '{log_level}', using INFO", file=sys.stderr)
        numeric_level = logging.INFO
        log_level = "INFO"
    
    # Use default format if not provided
    if format_string is None:
        format_string = DEFAULT_LOG_FORMAT
    
    # Create handlers if not provided
    if handlers is None:
        # Use stdout for container compatibility
        handler = StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(format_string))
        handlers = [handler]
    
    # Configure basic logging
    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    # Set root logger level
    logging.getLogger().setLevel(numeric_level)
    
    # Get service logger
    logger = logging.getLogger(service_name)
    logger.setLevel(numeric_level)
    
    # Force stdout to be unbuffered for real-time logging in containers
    if force_unbuffered:
        if hasattr(sys.stdout, 'reconfigure'):
            try:
                sys.stdout.reconfigure(line_buffering=True)
            except Exception:
                # Ignore errors on systems that don't support reconfigure
                pass
        # Also set environment variable for Python buffering
        os.environ['PYTHONUNBUFFERED'] = '1'
    
    # Log initial setup information
    logger.info(f"Logging configured for {service_name} at {log_level} level")
    
    return logger


def get_logger(name: str) -> Logger:
    """Get a logger instance with the current configuration.
    
    This is a convenience function for getting additional loggers after
    initial setup. The logger will inherit the configuration from the
    root logger.
    
    Args:
        name: Logger name (typically __name__ or module name)
        
    Returns:
        Logger: Logger instance
        
    Example:
        >>> # In main service file
        >>> logger = setup_logging("my_service")
        >>> 
        >>> # In another module
        >>> from utils.logging_config import get_logger
        >>> logger = get_logger(__name__)
    """
    return logging.getLogger(name)


def configure_module_logging(
    module_name: str,
    level: Optional[str] = None
) -> None:
    """Configure logging level for a specific module.
    
    This is useful for adjusting verbosity of third-party libraries or
    specific modules without affecting the entire application.
    
    Args:
        module_name: Name of the module (e.g., "paho.mqtt", "urllib3")
        level: Log level for the module. If None, inherits from root
        
    Example:
        >>> # Reduce MQTT client verbosity
        >>> configure_module_logging("paho.mqtt", "WARNING")
        >>> 
        >>> # Silence urllib3 warnings
        >>> configure_module_logging("urllib3", "ERROR")
    """
    logger = logging.getLogger(module_name)
    
    if level is not None:
        numeric_level = getattr(logging, level.upper(), None)
        if isinstance(numeric_level, int):
            logger.setLevel(numeric_level)
        else:
            logging.warning(f"Invalid log level '{level}' for module {module_name}")


# Common module configurations to reduce noise
def apply_common_filters() -> None:
    """Apply common logging filters to reduce noise from third-party libraries.
    
    This function sets appropriate log levels for commonly noisy modules
    to keep logs focused on application-specific messages.
    """
    # Reduce verbosity of common libraries
    configure_module_logging("paho.mqtt", "WARNING")
    configure_module_logging("urllib3", "WARNING")
    configure_module_logging("requests", "WARNING")
    configure_module_logging("docker", "WARNING")
    configure_module_logging("scapy", "ERROR")  # Very noisy during network scans
    

def setup_file_logging(
    service_name: str,
    log_dir: str = "/var/log/wildfire-watch",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> Logger:
    """Set up logging with both console and file output.
    
    This function sets up rotating file handlers in addition to console output,
    useful for services that need persistent logs.
    
    Args:
        service_name: Name of the service
        log_dir: Directory for log files
        max_bytes: Maximum size of each log file
        backup_count: Number of backup files to keep
        
    Returns:
        Logger: Configured logger instance
        
    Note:
        Requires write permissions to log_dir. Falls back to console-only
        if file logging fails.
    """
    from logging.handlers import RotatingFileHandler
    
    # First set up console logging
    logger = setup_logging(service_name)
    
    # Try to add file handler
    try:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{service_name}.log")
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
        
        # Add to both root and service logger
        logging.getLogger().addHandler(file_handler)
        logger.addHandler(file_handler)
        
        logger.info(f"File logging enabled at {log_file}")
        
    except Exception as e:
        logger.warning(f"Could not set up file logging: {e}")
    
    return logger