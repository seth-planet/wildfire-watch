#!/usr/bin/env python3
"""Fire Consensus Service (Refactored) - Multi-camera fire detection validation.

This is a refactored version of the fire consensus service that uses the new
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
import threading
import logging
import hashlib
import math
from datetime import datetime, timedelta, timezone
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Set
import numpy as np

from dotenv import load_dotenv

# Import base classes
# Add parent directory to path if not already there
parent_dir = os.path.join(os.path.dirname(__file__), '..')
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
from utils.mqtt_service import MQTTService
from utils.health_reporter import HealthReporter
from utils.thread_manager import ThreadSafeService, SafeTimerManager
from utils.config_base import ConfigBase, ConfigSchema
from utils.safe_logging import SafeLoggingMixin
from utils.logging_config import setup_logging

load_dotenv()

# ─────────────────────────────────────────────────────────────
# Configuration using ConfigBase
# ─────────────────────────────────────────────────────────────
class FireConsensusConfig(ConfigBase):
    """Configuration for Fire Consensus service."""
    
    SCHEMA = {
        # Service identification
        'service_id': ConfigSchema(
            str,
            default=f"fire_consensus_{socket.gethostname()}",
            description="Unique service identifier"
        ),
        
        # Core consensus parameters
        'consensus_threshold': ConfigSchema(
            int,
            default=2,
            min=1,
            max=10,
            description="Number of cameras required for consensus"
        ),
        'single_camera_trigger': ConfigSchema(
            bool,
            default=False,
            description="Allow single camera to trigger (overrides threshold)"
        ),
        'detection_window': ConfigSchema(
            float,
            default=30.0,
            min=10.0,
            max=300.0,
            description="Time window for detection history (seconds)"
        ),
        'cooldown_period': ConfigSchema(
            float,
            default=300.0,
            min=0.0,
            max=3600.0,
            description="Minimum time between triggers (seconds)"
        ),
        
        # Detection filtering
        'min_confidence': ConfigSchema(
            float,
            default=0.7,
            min=0.0,
            max=1.0,
            description="Minimum ML confidence score"
        ),
        'min_area_ratio': ConfigSchema(
            float,
            default=0.001,
            min=0.0,
            max=1.0,
            description="Minimum fire area as fraction of frame"
        ),
        'max_area_ratio': ConfigSchema(
            float,
            default=0.8,
            min=0.0,
            max=1.0,
            description="Maximum fire area as fraction of frame"
        ),
        
        # Growth analysis
        'area_increase_ratio': ConfigSchema(
            float,
            default=1.2,
            min=1.0,
            max=5.0,
            description="Required growth ratio (1.2 = 20% growth)"
        ),
        'moving_average_window': ConfigSchema(
            int,
            default=3,
            min=1,
            max=10,
            description="Detections for moving average smoothing"
        ),
        
        # System health
        'camera_timeout': ConfigSchema(
            float,
            default=60.0,
            min=10.0,
            max=600.0,
            description="Seconds before marking camera offline"
        ),
        'health_interval': ConfigSchema(
            int,
            default=30,
            min=10,
            max=300,
            description="Health report frequency (seconds)"
        ),
        'memory_cleanup_interval': ConfigSchema(
            int,
            default=300,
            min=60,
            max=3600,
            description="State cleanup frequency (seconds)"
        ),
        
        # Zone mapping
        'zone_activation': ConfigSchema(
            bool,
            default=False,
            description="Enable zone-based sprinkler control"
        ),
        'zone_mapping': ConfigSchema(
            dict,
            default={},
            description="Camera ID to zone list mapping"
        ),
        
        # MQTT settings
        'mqtt_broker': ConfigSchema(str, required=True, default='mqtt_broker'),
        'mqtt_port': ConfigSchema(int, default=1883, min=1, max=65535),
        'mqtt_tls': ConfigSchema(bool, default=False),
        'mqtt_username': ConfigSchema(str, default=''),
        'mqtt_password': ConfigSchema(str, default=''),
        'topic_prefix': ConfigSchema(str, default='', description="MQTT topic prefix"),
        
        # TLS settings
        'tls_ca_path': ConfigSchema(str, default='/mnt/data/certs/ca.crt'),
    }
    
    def __init__(self):
        super().__init__()
        
    def validate(self):
        """Validate fire consensus configuration."""
        # If single camera trigger enabled, consensus threshold doesn't matter
        if self.single_camera_trigger and self.consensus_threshold > 1:
            logging.warning("single_camera_trigger=true overrides consensus_threshold")
            
        # Ensure min area < max area
        if self.min_area_ratio >= self.max_area_ratio:
            raise ValueError("min_area_ratio must be less than max_area_ratio")
            
        # Detection window must be long enough for moving average
        min_window = self.moving_average_window * 2
        if self.detection_window < min_window:
            logging.warning(f"detection_window ({self.detection_window}s) may be too short "
                          f"for moving_average_window ({self.moving_average_window})")


# ─────────────────────────────────────────────────────────────
# Data Models (reuse from original)
# ─────────────────────────────────────────────────────────────
class Detection:
    """Single fire detection from a camera."""
    def __init__(self, confidence: float, area: float, object_id: str = "0"):
        self.confidence = confidence
        self.area = area
        self.object_id = object_id
        self.timestamp = time.time()

class CameraState:
    """State tracking for a single camera."""
    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self.detections: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.last_seen = time.time()
        self.is_online = True
        self.total_detections = 0
        self.last_detection_time = 0


# ─────────────────────────────────────────────────────────────
# Consensus Health Reporter
# ─────────────────────────────────────────────────────────────
class ConsensusHealthReporter(HealthReporter):
    """Health reporter for fire consensus service."""
    
    def __init__(self, consensus):
        self.consensus = consensus
        # Pass the consensus instance as mqtt_service since FireConsensus extends MQTTService
        super().__init__(consensus, consensus.config.health_interval)
        
    def get_service_health(self) -> Dict[str, any]:
        """Get consensus-specific health metrics."""
        with self.consensus.lock:
            cameras = list(self.consensus.cameras.values())
            online_cameras = [c for c in cameras if c.is_online]
            
            # Calculate detection rates
            now = time.time()
            recent_detections = sum(
                1 for c in cameras 
                if now - c.last_detection_time < 60
            )
            
            # Get consensus state
            consensus_state = "idle"
            if self.consensus.last_trigger_time > 0:
                time_since_trigger = now - self.consensus.last_trigger_time
                if time_since_trigger < self.consensus.config.cooldown_period:
                    consensus_state = "cooldown"
                    
            health = {
                'healthy': True,  # Service is healthy if we get this far
                'consensus_state': consensus_state,
                'trigger_count': self.consensus.trigger_count,
                'last_trigger_ago': int(now - self.consensus.last_trigger_time) if self.consensus.last_trigger_time > 0 else -1,
                'cameras_total': len(cameras),
                'cameras_online': len(online_cameras),
                'cameras_detecting': recent_detections,
                'consensus_threshold': self.consensus.config.consensus_threshold,
                'single_camera_mode': self.consensus.config.single_camera_trigger,
                'recent_consensus_events': len(self.consensus.consensus_events),
            }
            
            # Add per-camera stats
            camera_stats = {}
            for cam in cameras:
                total_objects = sum(len(detections) for detections in cam.detections.values())
                camera_stats[cam.camera_id] = {
                    'online': cam.is_online,
                    'total_detections': cam.total_detections,
                    'active_objects': total_objects,
                    'last_seen_ago': int(now - cam.last_seen)
                }
            health['cameras'] = camera_stats
            
        return health


# ─────────────────────────────────────────────────────────────
# Refactored Fire Consensus Service
# ─────────────────────────────────────────────────────────────
class FireConsensus(MQTTService, ThreadSafeService, SafeLoggingMixin):
    """Refactored fire consensus service using base classes.
    
    This implementation reduces code duplication by:
    1. Using MQTTService for all MQTT handling
    2. Using ThreadSafeService for thread management
    3. Using HealthReporter for health monitoring
    4. Using SafeTimerManager for timer management
    """
    
    def __init__(self, auto_connect=True):
        # Load configuration
        self.config = FireConsensusConfig()
        
        # Initialize base classes - ThreadSafeService first to set up timer_manager
        ThreadSafeService.__init__(self, "fire_consensus", logging.getLogger(__name__))
        MQTTService.__init__(self, "fire_consensus", self.config)
        
        # Core state
        self.cameras: Dict[str, CameraState] = {}
        self.last_trigger_time = 0
        self.trigger_count = 0
        self.consensus_events = deque(maxlen=1000)
        self.lock = threading.RLock()
        self.health_reporter = None  # Initialize to None, will be set in _initialize_mqtt_connection
        
        # Setup MQTT with subscriptions
        subscriptions = [
            "fire/detection",
            "fire/detection/+",
            "frigate/events",
            "system/camera_telemetry"
        ]
        
        self.setup_mqtt(
            on_connect=self._on_connect,
            on_message=self._on_message,
            subscriptions=subscriptions
        )
        
        # Enable offline queue for resilience
        self.enable_offline_queue(max_size=100)
        
        try:
            # Start background tasks
            self._safe_log('info', "Starting background tasks...")
            self._start_background_tasks()
            self._safe_log('info', "Background tasks started successfully")
        except Exception as e:
            self._safe_log('error', f"Failed to start background tasks: {e}", exc_info=True)
            raise
        
        self._safe_log('info', f"Fire Consensus configured: {self.config.service_id}")
        
        # Only connect to MQTT if auto_connect is True (default behavior)
        # This allows tests to create the service without immediate connection
        if auto_connect:
            self._initialize_mqtt_connection()
            
    def _initialize_mqtt_connection(self):
        """Initialize MQTT connection and health reporting.
        
        This is separated from __init__ to allow tests to set up environment
        variables before connection is attempted.
        """
        # Connect to MQTT before creating health reporter
        # This ensures MQTT is ready for health messages
        try:
            self._safe_log('info', "Connecting to MQTT...")
            self.connect()
            self._safe_log('info', "MQTT connection initiated")
        except Exception as e:
            self._safe_log('error', f"Failed to connect to MQTT: {e}", exc_info=True)
            raise
        
        # Setup health reporter AFTER MQTT connection is initiated
        self._safe_log('info', "Creating ConsensusHealthReporter...")
        self.health_reporter = ConsensusHealthReporter(self)
        self._safe_log('info', "ConsensusHealthReporter created successfully")
        
        # Start health reporting
        try:
            self._safe_log('info', f"Starting health reporting with interval: {self.config.health_interval} seconds")
            self.health_reporter.start_health_reporting()
            self._safe_log('info', "Health reporting started")
        except Exception as e:
            self._safe_log('error', f"Failed to start health reporting: {e}", exc_info=True)
            raise
        
        self._safe_log('info', f"Fire Consensus fully initialized: {self.config.service_id}")
        
        # Publish initial health status after connection
        try:
            self._safe_log('info', "Publishing initial health status...")
            initial_health = {
                'healthy': True,
                'status': 'starting',
                'service': 'fire_consensus',
                'timestamp': time.time()
            }
            if self.publish_message("system/fire_consensus/health", initial_health, retain=True):
                self._safe_log('info', "Initial health status published successfully")
            else:
                self._safe_log('error', "Failed to publish initial health status")
        except Exception as e:
            self._safe_log('error', f"Failed to publish initial health: {e}", exc_info=True)
        
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback."""
        self._safe_log('info', "MQTT connected, ready for fire detection messages")
        
        # Restart health reporting after reconnection
        if hasattr(self, 'health_reporter') and self.health_reporter:
            self._safe_log('info', "Restarting health reporting after reconnection")
            try:
                # Publish immediate health status
                initial_health = {
                    'healthy': True,
                    'status': 'online',
                    'service': 'fire_consensus',
                    'timestamp': time.time()
                }
                if self.publish_message("system/fire_consensus/health", initial_health, retain=True):
                    self._safe_log('info', "Published health status after reconnection")
                else:
                    self._safe_log('error', "Failed to publish health status after reconnection")
                    
                # Force an immediate health report
                self.health_reporter._publish_health()
            except Exception as e:
                self._safe_log('error', f"Failed to restart health reporting: {e}", exc_info=True)
        
    def _on_message(self, topic: str, payload: any):
        """Handle incoming MQTT messages."""
        try:
            # Debug: Log all received messages
            self._safe_log('debug', f"Received message on topic '{topic}': {str(payload)[:100]}...")
            
            # Route messages based on topic
            if topic.startswith("fire/detection"):
                self._safe_log('debug', f"Processing fire detection message on topic '{topic}'")
                self._handle_fire_detection(topic, payload)
            elif topic == "frigate/events":
                self._safe_log('debug', f"Processing frigate event message on topic '{topic}'")
                self._handle_frigate_event(payload)
            elif topic == "system/camera_telemetry":
                self._safe_log('debug', f"Processing camera telemetry message on topic '{topic}'")
                self._handle_camera_telemetry(payload)
            else:
                self._safe_log('debug', f"Ignoring message on topic '{topic}' (no handler)")
                
        except Exception as e:
            self._safe_log('error', f"Error processing message on {topic}: {e}", exc_info=True)
            
    def _start_background_tasks(self):
        """Start background tasks using timer manager."""
        try:
            # Memory cleanup task
            self._safe_log('debug', f"Scheduling memory cleanup with interval {self.config.memory_cleanup_interval}s")
            self.timer_manager.schedule(
                "memory_cleanup",
                self._cleanup_old_data,
                self.config.memory_cleanup_interval
            )
            self._safe_log('info', "Memory cleanup task scheduled successfully")
        except Exception as e:
            self._safe_log('error', f"Failed to schedule memory cleanup: {e}", exc_info=True)
            # Don't raise - memory cleanup is not critical
        
    def _handle_fire_detection(self, topic: str, payload: Dict):
        """Process fire detection from cameras."""
        try:
            # Extract camera ID
            parts = topic.split('/')
            camera_id = parts[2] if len(parts) > 2 else payload.get('camera_id', 'unknown')
            
            # Validate camera_id
            if not camera_id or camera_id == 'unknown':
                return
                
            # Validate detection
            confidence = float(payload.get('confidence', 0))
            if confidence < self.config.min_confidence:
                return
                
            # Calculate area - support both 'bbox' and 'bounding_box' for compatibility
            bbox = payload.get('bbox') or payload.get('bounding_box') or []
            if bbox is None or len(bbox) != 4:
                return  # Invalid bbox
            
            area = self._calculate_area(bbox)
            self._safe_log('debug', f"Calculated area: {area}, min: {self.config.min_area_ratio}, max: {self.config.max_area_ratio}")
            if not (self.config.min_area_ratio <= area <= self.config.max_area_ratio):
                self._safe_log('debug', f"Rejecting detection due to invalid area: {area}")
                return
                
            # Create detection
            object_id = str(payload.get('object_id', '0'))
            detection = Detection(confidence, area, object_id)
            
            # Add detection and check consensus
            self._add_detection(camera_id, detection)
            
        except Exception as e:
            self._safe_log('error', f"Error handling fire detection: {e}")
            
    def _handle_frigate_event(self, payload: Dict):
        """Process Frigate NVR events."""
        try:
            # Filter for fire/smoke events
            label = payload.get('after', {}).get('label', '')
            if label not in ['fire', 'smoke']:
                return
                
            # Extract data
            camera_id = payload.get('after', {}).get('camera', 'unknown')
            confidence = float(payload.get('after', {}).get('top_score', 0))
            
            if confidence < self.config.min_confidence:
                return
                
            # Calculate area from box
            box = payload.get('after', {}).get('box', [])
            if len(box) == 4:
                # Use the helper to get normalized area
                area = self._calculate_area(box)
                if not (self.config.min_area_ratio <= area <= self.config.max_area_ratio):
                    return
            else:
                area = 0.01
                
            # Create detection
            object_id = str(payload.get('after', {}).get('id', '0'))
            detection = Detection(confidence, area, object_id)
            
            # Add detection
            self._add_detection(camera_id, detection)
            
        except Exception as e:
            self._safe_log('error', f"Error handling Frigate event: {e}")
            
    def _handle_camera_telemetry(self, payload: Dict):
        """Update camera online status from telemetry."""
        try:
            camera_id = payload.get('camera_id')
            if not camera_id:
                return
                
            with self.lock:
                if camera_id not in self.cameras:
                    self.cameras[camera_id] = CameraState(camera_id)
                    
                camera = self.cameras[camera_id]
                camera.last_seen = time.time()
                camera.is_online = True
                
        except Exception as e:
            self._safe_log('error', f"Error handling camera telemetry: {e}")
            
    def _add_detection(self, camera_id: str, detection: Detection):
        """Add detection and check for consensus."""
        with self.lock:
            # Ensure camera exists
            if camera_id not in self.cameras:
                self.cameras[camera_id] = CameraState(camera_id)
                
            camera = self.cameras[camera_id]
            camera.detections[detection.object_id].append(detection)
            camera.total_detections += 1
            camera.last_detection_time = time.time()
            camera.last_seen = time.time()
            camera.is_online = True
            
            detection_count = len(camera.detections[detection.object_id])
            self._safe_log('debug', f"[DEBUG] Added detection from {camera_id}: "
                            f"confidence={detection.confidence:.2f}, "
                            f"area={detection.area:.4f}, "
                            f"object_id={detection.object_id}, "
                            f"total_for_object={detection_count}")
            
        # Check consensus
        self._check_consensus()
        
    def _check_consensus(self):
        """Check if consensus conditions are met."""
        with self.lock:
            # Check cooldown
            current_time = time.time()
            if current_time - self.last_trigger_time < self.config.cooldown_period:
                return
                
            # Get growing fires from all cameras
            cameras_with_fires = []
            fire_locations = []
            
            for camera_id, camera in self.cameras.items():
                if not camera.is_online:
                    continue
                    
                growing_fires = self.get_growing_fires(camera_id)
                if growing_fires:
                    cameras_with_fires.append(camera_id)
                    fire_locations.extend([
                        (camera_id, obj_id) for obj_id in growing_fires
                    ])
                    
            # Check consensus threshold
            consensus_met = False
            if self.config.single_camera_trigger and len(cameras_with_fires) >= 1:
                consensus_met = True
            elif len(cameras_with_fires) >= self.config.consensus_threshold:
                consensus_met = True
                
            if consensus_met:
                # CRITICAL SAFETY FIX: Update trigger time ATOMICALLY before firing trigger
                # to prevent race condition where multiple threads trigger simultaneously
                self.last_trigger_time = current_time
                
                self._safe_log('warning', f"CONSENSUS REACHED! {len(cameras_with_fires)} cameras "
                                  f"detecting growing fires: {cameras_with_fires}")
                self._trigger_fire_response(cameras_with_fires, fire_locations)
                
                # Record consensus event using the atomic timestamp
                self.consensus_events.append({
                    'timestamp': self.last_trigger_time,
                    'cameras': cameras_with_fires,
                    'fire_count': len(fire_locations)
                })
                
    def get_growing_fires(self, camera_id: str) -> List[str]:
        """Get list of growing fire object IDs for a camera."""
        camera = self.cameras.get(camera_id)
        if not camera:
            self._safe_log('debug', f"[DEBUG] get_growing_fires: No camera found for {camera_id}")
            return []
            
        growing_fires = []
        current_time = time.time()
        
        for object_id, detections in camera.detections.items():
            # Filter recent detections
            recent = [d for d in detections 
                     if current_time - d.timestamp <= self.config.detection_window]
            
            self._safe_log('debug', f"[DEBUG] Camera {camera_id}, Object {object_id}: {len(recent)} recent detections out of {len(detections)} total")
            
            if len(recent) < self.config.moving_average_window * 2:
                self._safe_log('debug', f"[DEBUG] Not enough recent detections: {len(recent)} < {self.config.moving_average_window * 2}")
                continue
                
            # Calculate moving averages using median for noise robustness
            # SAFETY FIX: Replace np.mean() with np.median() for better robustness against sensor noise
            areas = [d.area for d in recent]
            early_avg = np.median(areas[:self.config.moving_average_window])
            recent_avg = np.median(areas[-self.config.moving_average_window:])
            
            growth_ratio = recent_avg / early_avg if early_avg > 0 else 0
            self._safe_log('debug', f"[DEBUG] Growth analysis: early_avg={early_avg:.4f}, recent_avg={recent_avg:.4f}, ratio={growth_ratio:.2f}, required={self.config.area_increase_ratio}")
            
            # Check growth
            if recent_avg >= early_avg * self.config.area_increase_ratio:
                growing_fires.append(object_id)
                self._safe_log('debug', f"[DEBUG] GROWING FIRE DETECTED: {object_id} on camera {camera_id}")
                
        return growing_fires
        
    def _trigger_fire_response(self, cameras: List[str], fire_locations: List[Tuple[str, str]]):
        """Send fire suppression trigger command."""
        # Note: last_trigger_time is now set atomically in _check_consensus
        # to prevent race conditions
        self.trigger_count += 1
        
        # Build trigger payload
        payload = {
            'action': 'trigger',
            'confidence': 'high',
            'timestamp': self.last_trigger_time,
            'trigger_count': self.trigger_count,
            'consensus_cameras': cameras,
            'fire_locations': fire_locations,
            'service_id': self.config.service_id
        }
        
        # Add zone information if enabled
        if self.config.zone_activation:
            zones = set()
            for camera_id in cameras:
                camera_zones = self.config.zone_mapping.get(camera_id, [])
                zones.update(camera_zones)
            payload['zones'] = list(zones)
            
        # Publish with QoS 2 (exactly once)
        self.publish_message("fire/trigger", payload, qos=2)
        
        self._safe_log('critical', f"FIRE TRIGGER SENT! Cameras: {cameras}")
        
    def _cleanup_old_data(self):
        """Periodic cleanup of old detection data."""
        with self.lock:
            current_time = time.time()
            
            for camera in self.cameras.values():
                # Clean old detections
                for object_id, detections in list(camera.detections.items()):
                    # Remove old detections
                    while detections and current_time - detections[0].timestamp > self.config.detection_window * 2:
                        detections.popleft()
                        
                    # Remove empty deques
                    if not detections:
                        del camera.detections[object_id]
        
                # Update online status
                if current_time - camera.last_seen > self.config.camera_timeout:
                    camera.is_online = False
        
        # Reschedule for next cleanup
        if not getattr(self, '_shutdown', False):
            self.timer_manager.schedule(
                "memory_cleanup",
                self._cleanup_old_data,
                self.config.memory_cleanup_interval
            )
        
    def _calculate_area(self, bbox: List[float], camera_resolution: Optional[Tuple[int, int]] = None) -> float:
        """Calculate normalized area from bounding box.
        
        Args:
            bbox: Bounding box in format [x1, y1, x2, y2]
            camera_resolution: Optional (width, height) tuple. If None, uses 1920x1080 default.
        
        Returns:
            Area as fraction of frame (0.0 to 1.0)
        """
        try:
            # Validate bbox values
            for val in bbox:
                if math.isnan(val) or math.isinf(val) or val < 0:
                    return 0
            
            # Handle both pixel coordinates and normalized coordinates
            if all(0 <= val <= 1 for val in bbox):
                # Normalized coordinates (0-1) - already correct
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                
                # Check for negative dimensions
                if width <= 0 or height <= 0:
                    return 0
                    
                area = width * height
            else:
                # Pixel coordinates - need to normalize
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                
                # Use provided resolution or default to 1920x1080
                if camera_resolution:
                    frame_width, frame_height = camera_resolution
                else:
                    frame_width, frame_height = 1920, 1080
                
                # Check for negative dimensions
                if width <= 0 or height <= 0:
                    return 0
                    
                area = (width * height) / (frame_width * frame_height)
            
            # Final validation
            if math.isnan(area) or math.isinf(area) or area < 0:
                return 0
                
            return area
        except (ValueError, TypeError):
            return 0
        

    def cleanup(self):
        """Clean shutdown of service."""
        self._safe_log('info', "Shutting down Fire Consensus")
        
        # Stop health reporting
        if hasattr(self, 'health_reporter') and self.health_reporter is not None:
            self.health_reporter.stop_health_reporting()
            
        # Cancel all timers
        self.timer_manager.cancel_all()
        
        # Shutdown base services
        ThreadSafeService.shutdown(self)
        MQTTService.shutdown(self)
        
        self._safe_log('info', "Fire Consensus shutdown complete")
        

# ─────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────


def main():
    """Main entry point for fire consensus service."""
    # Setup logging using standardized configuration
    logger = setup_logging("fire_consensus")
    
    try:
        # Create with auto_connect=True (default) for normal operation
        consensus = FireConsensus()
        
        # Wait for shutdown
        consensus.wait_for_shutdown()
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if 'consensus' in locals():
            consensus.cleanup()
            

if __name__ == "__main__":
    main()