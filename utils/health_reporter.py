#!/usr/bin/env python3.12
"""Base health reporter class for Wildfire Watch services.

This module provides a base class for standardized health reporting across
all services, with periodic publishing and service-specific metrics.
"""

import time
import threading
import platform
import psutil
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable

from .mqtt_service import MQTTService


class HealthReporter(ABC):
    """Base class for service health reporting.
    
    Provides periodic health status publishing with common metrics
    and service-specific metrics via override.
    
    Attributes:
        mqtt_service: The MQTT service instance for publishing
        interval: Health report interval in seconds
    """
    
    def __init__(self, mqtt_service: MQTTService, interval: float = 60.0):
        """Initialize health reporter.
        
        Args:
            mqtt_service: MQTT service instance for publishing
            interval: Seconds between health reports
        """
        self.mqtt_service = mqtt_service
        self.interval = interval
        self._health_timer: Optional[threading.Timer] = None
        self._start_time = time.time()
        self._shutdown = False
        self._lock = threading.Lock()
        
        # System resource monitoring
        self._process = psutil.Process()
        
    def start_health_reporting(self) -> None:
        """Start periodic health reporting."""
        self._shutdown = False
        self.mqtt_service.logger.info(f"[HEALTH DEBUG] Starting health reporting for {self.mqtt_service.service_name} with interval {self.interval}s")
        self.mqtt_service.logger.info(f"[HEALTH DEBUG] MQTT connected: {self.mqtt_service.is_connected}")
        self.mqtt_service.logger.info(f"[HEALTH DEBUG] Calling _publish_health immediately")
        self._publish_health()
        
    def stop_health_reporting(self) -> None:
        """Stop health reporting and cleanup."""
        self._shutdown = True
        with self._lock:
            if self._health_timer:
                self._health_timer.cancel()
                # Wait for timer thread to finish to prevent logging after cleanup
                if self._health_timer.is_alive():
                    self._health_timer.join(timeout=2.0)
                self._health_timer = None
    
    def _safe_log(self, level: str, message: str, exc_info: bool = False) -> None:
        """Safely log a message with comprehensive checks."""
        try:
            logger = getattr(self.mqtt_service, 'logger', None)
            if not logger:
                return
                
            # Check if logger has handlers and they're not closed
            if not hasattr(logger, 'handlers') or not logger.handlers:
                return
                
            # Check each handler to ensure it's not closed
            for handler in logger.handlers:
                if hasattr(handler, 'stream') and hasattr(handler.stream, 'closed'):
                    if handler.stream.closed:
                        return
                        
            # Log the message
            getattr(logger, level.lower())(message, exc_info=exc_info)
        except (ValueError, AttributeError, OSError):
            # Silently ignore logging errors during shutdown
            pass

    def _publish_health(self) -> None:
        """Publish health status and reschedule."""
        if self._shutdown:
            return
        
        try:
            self._safe_log('info', f"[HEALTH DEBUG] _publish_health called for {self.mqtt_service.service_name}")
            
            # Check MQTT connection first
            if not self.mqtt_service.is_connected:
                self._safe_log('warning', f"[HEALTH DEBUG] MQTT not connected for {self.mqtt_service.service_name}, skipping health publish")
                # Still reschedule to try again later
            else:
                # Gather health data
                health_data = self._get_base_health_data()
                
                # Add service-specific health data
                service_health = self.get_service_health()
                if service_health:
                    health_data.update(service_health)
                
                # Publish health status
                topic = f"system/{self.mqtt_service.service_name}/health"
                self._safe_log('info', f"[HEALTH DEBUG] Publishing health to {topic} with data keys: {list(health_data.keys())}")
                result = self.mqtt_service.publish_message(
                    topic,
                    health_data,
                    retain=True
                )
                if result:
                    self._safe_log('info', f"[HEALTH DEBUG] Health published successfully to {topic}")
                else:
                    self._safe_log('error', f"[HEALTH DEBUG] Failed to publish health to {topic}")
            
        except Exception as e:
            self._safe_log('error', f"[HEALTH DEBUG] Error publishing health: {e}", exc_info=True)
        
        # Reschedule next health report
        if not self._shutdown:
            with self._lock:
                self._safe_log('info', f"[HEALTH DEBUG] Scheduling next health report in {self.interval}s")
                self._health_timer = threading.Timer(self.interval, self._publish_health)
                # Set daemon=False so thread can be properly joined
                self._health_timer.daemon = False
                self._health_timer.start()
    
    def _get_base_health_data(self) -> Dict[str, Any]:
        """Get base health metrics common to all services.
        
        Returns:
            Dictionary of base health metrics
        """
        uptime = time.time() - self._start_time
        
        # Get system resources
        try:
            cpu_percent = self._process.cpu_percent(interval=0.1)
            memory_info = self._process.memory_info()
            memory_percent = self._process.memory_percent()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            cpu_percent = 0.0
            memory_info = None
            memory_percent = 0.0
        
        health_data = {
            'timestamp': time.time(),
            'uptime': uptime,
            'uptime_hours': round(uptime / 3600, 2),
            'service': self.mqtt_service.service_name,
            'version': getattr(self.mqtt_service.config, 'version', 'unknown'),
            'mqtt_connected': self.mqtt_service.is_connected,
            'hostname': platform.node(),
            'pid': self._process.pid,
            'resources': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_mb': memory_info.rss / 1024 / 1024 if memory_info else 0,
            }
        }
        
        # Add system-wide resources if available
        try:
            health_data['system'] = {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
            }
        except Exception:
            pass  # Optional system metrics
        
        return health_data
    
    @abstractmethod
    def get_service_health(self) -> Dict[str, Any]:
        """Get service-specific health metrics.
        
        This method must be implemented by each service to provide
        service-specific health information.
        
        Returns:
            Dictionary of service-specific health metrics
        """
        pass
    
    def force_health_update(self) -> None:
        """Force an immediate health update."""
        # Cancel current timer
        with self._lock:
            if self._health_timer:
                self._health_timer.cancel()
        
        # Publish immediately
        self._publish_health()


class ServiceHealthReporter(HealthReporter):
    """Generic health reporter for services without special metrics."""
    
    def __init__(self, mqtt_service: MQTTService, 
                 service_state_getter: Optional[Callable] = None,
                 interval: float = 60.0):
        """Initialize generic health reporter.
        
        Args:
            mqtt_service: MQTT service instance
            service_state_getter: Optional function to get service state
            interval: Health report interval
        """
        super().__init__(mqtt_service, interval)
        self._state_getter = service_state_getter
    
    def get_service_health(self) -> Dict[str, Any]:
        """Get generic service health."""
        health = {
            'status': 'healthy',
            'last_activity': time.time(),
        }
        
        # Add service state if available
        if self._state_getter:
            try:
                state = self._state_getter()
                if isinstance(state, dict):
                    health.update(state)
                else:
                    health['state'] = str(state)
            except Exception as e:
                health['state_error'] = str(e)
        
        return health