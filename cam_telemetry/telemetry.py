#!/usr/bin/env python3
"""System health monitoring service for Wildfire Watch edge devices (Refactored).

This is a refactored version of the telemetry service that uses the new
base classes for reduced code duplication and improved maintainability.

Key Improvements:
1. Uses MQTTService base class for connection management
2. Uses HealthReporter base class for health monitoring
3. Uses ThreadSafeService for thread management
4. Configuration management via ConfigBase
5. Cleaner separation of concerns
"""

import os
import sys
import time
import json
import socket
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from utils.safe_logging import safe_log

# Import base classes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.mqtt_service import MQTTService
from utils.health_reporter import HealthReporter
from utils.thread_manager import ThreadSafeService
from utils.config_base import ConfigBase, ConfigSchema
from utils.logging_config import setup_logging

# Try to import psutil for system metrics
try:
    import psutil
except ImportError:
    psutil = None

load_dotenv()


# ─────────────────────────────────────────────────────────────
# Configuration using ConfigBase
# ─────────────────────────────────────────────────────────────
class TelemetryConfig(ConfigBase):
    """Configuration for Telemetry service."""
    
    SCHEMA = {
        # Service identification
        'camera_id': ConfigSchema(
            str,
            default=socket.gethostname(),
            description="Unique camera identifier"
        ),
        
        # Telemetry settings
        'telemetry_interval': ConfigSchema(
            int,
            default=60,
            min=10,
            max=3600,
            description="Seconds between telemetry updates"
        ),
        'telemetry_topic': ConfigSchema(
            str,
            default='system/telemetry',
            description="MQTT topic for telemetry data"
        ),
        
        # Service configuration (for reporting)
        'rtsp_stream_url': ConfigSchema(
            str,
            default='',
            description="RTSP stream URL being monitored"
        ),
        'model_path': ConfigSchema(
            str,
            default='',
            description="AI model path being used"
        ),
        'detector': ConfigSchema(
            str,
            default='unknown',
            description="Detector backend type"
        ),
        
        # MQTT settings
        'mqtt_broker': ConfigSchema(str, required=True, default='mqtt_broker'),
        'mqtt_port': ConfigSchema(int, default=1883, min=1, max=65535),
        'mqtt_tls': ConfigSchema(bool, default=False),
        'mqtt_username': ConfigSchema(str, default=''),
        'mqtt_password': ConfigSchema(str, default=''),
        'topic_prefix': ConfigSchema(str, default='', description="MQTT topic prefix"),
    }
    
    def __init__(self):
        super().__init__()
        
    def validate(self):
        """Validate telemetry configuration."""
        # No additional validation needed for telemetry
        pass


# ─────────────────────────────────────────────────────────────
# Telemetry Health Reporter
# ─────────────────────────────────────────────────────────────
class TelemetryHealthReporter(HealthReporter):
    """Health reporter for telemetry service."""
    
    def __init__(self, telemetry_service):
        self.telemetry = telemetry_service
        # Telemetry publishes its own health, so we use the same interval
        super().__init__(telemetry_service, telemetry_service.config.telemetry_interval)
        
    def get_service_health(self) -> Dict[str, Any]:
        """Get telemetry-specific health metrics."""
        metrics = self._get_system_metrics()
        
        health = {
            'camera_id': self.telemetry.config.camera_id,
            'status': 'online',
            'backend': self.telemetry.config.detector,
            'config': {
                'rtsp_url': self.telemetry.config.rtsp_stream_url,
                'model_path': self.telemetry.config.model_path
            }
        }
        
        # Add system metrics both as structured field and top-level
        if metrics:
            health['system_metrics'] = metrics
            # Also add some metrics at top level for backward compatibility
            for key in ['cpu_percent', 'memory_percent', 'free_disk_mb', 'total_disk_mb', 'uptime_seconds']:
                if key in metrics:
                    health[key] = metrics[key]
                    
        return health
    
    def _publish_health(self) -> None:
        """Override to publish to telemetry topic instead of health topic."""
        if self._shutdown:
            return
        
        try:
            self.telemetry._safe_log('info', f"[HEALTH DEBUG] _publish_health called for {self.mqtt_service.service_name}")
            
            # Check MQTT connection first
            if not self.mqtt_service.is_connected:
                self.telemetry._safe_log('warning', f"[HEALTH DEBUG] MQTT not connected for {self.mqtt_service.service_name}, skipping health publish")
                # Still reschedule to try again later
            else:
                # Gather health data
                health_data = self._get_base_health_data()
                
                # Add service-specific health data
                service_health = self.get_service_health()
                if service_health:
                    health_data.update(service_health)
                
                # Publish to telemetry topic instead of health topic
                topic = self.telemetry.config.telemetry_topic
                self.telemetry._safe_log('info', f"[HEALTH DEBUG] Publishing telemetry to {topic} with data keys: {list(health_data.keys())}")
                result = self.mqtt_service.publish_message(
                    topic,
                    health_data,
                    retain=False,  # Telemetry messages should not be retained
                    qos=1
                )
                if result:
                    self.telemetry._safe_log('info', f"[HEALTH DEBUG] Telemetry published successfully to {topic}")
                else:
                    self.telemetry._safe_log('error', f"[HEALTH DEBUG] Failed to publish telemetry to {topic}")
            
        except Exception as e:
            self.telemetry._safe_log('error', f"[HEALTH DEBUG] Error publishing telemetry: {e}", exc_info=True)
        
        # Reschedule next health report
        if not self._shutdown:
            with self._lock:
                self.telemetry._safe_log('info', f"[HEALTH DEBUG] Scheduling next telemetry report in {self.interval}s")
                self._health_timer = threading.Timer(self.interval, self._publish_health)
                self._health_timer.daemon = True
                self._health_timer.start()
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Collect system health metrics using psutil."""
        metrics = {}
        if psutil:
            try:
                du = psutil.disk_usage("/")
                vm = psutil.virtual_memory()
                
                metrics = {
                    "cpu_percent": psutil.cpu_percent(interval=None),
                    "memory_percent": vm.percent,
                    "disk_usage": {
                        "free": du.free,
                        "total": du.total,
                        "used": du.used,
                        "percent": round((du.used / du.total) * 100, 1)
                    },
                    "uptime_hours": round((time.time() - psutil.boot_time()) / 3600, 2),
                    # Legacy fields for backward compatibility
                    "free_disk_mb": round(du.free / 1024 / 1024, 2),
                    "total_disk_mb": round(du.total / 1024 / 1024, 2),
                    "uptime_seconds": int(time.time() - psutil.boot_time())
                }
            except Exception as e:
                self.telemetry._safe_log('error', f"Error collecting system metrics: {e}")
        return metrics


# ─────────────────────────────────────────────────────────────
# Refactored Telemetry Service
# ─────────────────────────────────────────────────────────────
class TelemetryService(MQTTService, ThreadSafeService):
    """Refactored telemetry service using base classes.
    
    This implementation reduces code duplication by:
    1. Using MQTTService for all MQTT handling
    2. Using ThreadSafeService for thread management
    3. Using HealthReporter for health monitoring
    4. Configuration via ConfigBase
    """
    
    def __init__(self):
        # Load configuration
        self.config = TelemetryConfig()
        
        # Initialize base classes
        MQTTService.__init__(self, "telemetry", self.config)
        ThreadSafeService.__init__(self, "telemetry", logging.getLogger(__name__))
        
        # Setup health reporter with custom implementation
        self.health_reporter = TelemetryHealthReporter(self)
        
        # Setup MQTT (no subscriptions for telemetry)
        self.setup_mqtt(
            on_connect=self._on_connect,
            on_message=None,
            subscriptions=[]
        )
        
        self._safe_log('info', f"Telemetry Service configured: {self.config.camera_id}")
        
        # Connect to MQTT after everything is initialized
        # This prevents race conditions during startup
        self.connect()
        self._safe_log('info', f"Telemetry Service fully initialized and connected: {self.config.camera_id}")
        
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback."""
        self._safe_log('info', "MQTT connected, starting telemetry reporting")
        # Start health reporting on connection
        self._start_telemetry_reporting()
        
    def _start_telemetry_reporting(self):
        """Start the telemetry reporting cycle."""
        # Start health reporting which will use our overridden _publish_health
        self.health_reporter.start_health_reporting()
        
    def cleanup(self):
        """Clean shutdown of service."""
        self._safe_log('info', "Shutting down Telemetry Service")
        
        # Stop health reporting
        if hasattr(self, 'health_reporter'):
            self.health_reporter.stop_health_reporting()
            
        # Shutdown base services
        ThreadSafeService.shutdown(self)
        MQTTService.shutdown(self)
        
        self._safe_log('info', "Telemetry Service shutdown complete")


# ─────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────


def main():
    """Main entry point for telemetry service."""
    # Setup logging using standardized configuration
    logger = setup_logging("cam_telemetry")
    
    try:
        telemetry = TelemetryService()
        
        # Wait for shutdown
        telemetry.wait_for_shutdown()
        
    except KeyboardInterrupt:
        safe_log(logger, 'info', "Received interrupt signal")
    except Exception as e:
        safe_log(logger, 'error', f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if 'telemetry' in locals():
            telemetry.cleanup()
            

if __name__ == "__main__":
    main()