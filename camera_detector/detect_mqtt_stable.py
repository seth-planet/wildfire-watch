#!/usr/bin/env python3.12
"""Camera Detector with Stable MQTT Implementation

This is a modified version of detect.py that uses the StableMQTTHandler
to prevent disconnections during long-running network operations.
"""

# First, we'll create a patch file that can be applied to detect.py
# This shows the specific changes needed

import json
from datetime import datetime, timezone
import logging

# Import the stable MQTT handler
from mqtt_stability_fix import StableMQTTHandler

# Create a mixin class that can be used to patch CameraDetector
class StableMQTTMixin:
    """Mixin to add stable MQTT functionality to CameraDetector"""
    
    def _setup_mqtt(self):
        """Setup stable MQTT handler"""
        # Create handler with reduced keepalive
        self.mqtt_handler = StableMQTTHandler(
            broker=self.config.MQTT_BROKER,
            port=self.config.MQTT_PORT,
            client_id=self.config.SERVICE_ID,
            keepalive=30,  # Reduced from 60 for faster detection
            tls_enabled=self.config.MQTT_TLS,
            ca_cert_path=self.config.TLS_CA_PATH if self.config.MQTT_TLS else None
        )
        
        # Keep the old mqtt_client reference for compatibility
        self.mqtt_client = self.mqtt_handler.client
        
        # Set callbacks that match original signature
        self.mqtt_handler.on_connect_callback = lambda: self._on_mqtt_connect(None, None, None, 0)
        self.mqtt_handler.on_disconnect_callback = lambda: self._on_mqtt_disconnect(None, None, 1)
        
        # Set LWT
        lwt_topic = f"{self.config.TOPIC_HEALTH}/{self.config.NODE_ID}/lwt"
        lwt_payload = json.dumps({
            'node_id': self.config.NODE_ID,
            'service': 'camera_detector',
            'status': 'offline',
            'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        })
        self.mqtt_handler.set_will(lwt_topic, lwt_payload, qos=1, retain=True)
        
        # Start handler
        self.mqtt_handler.start()
        
        # Wait for initial connection
        if not self.mqtt_handler.wait_for_connection(timeout=10.0):
            safe_log("Initial MQTT connection timeout - running in degraded mode", logging.WARNING)
        
        # Update connection status
        self.mqtt_connected = self.mqtt_handler.is_connected()
    
    def mqtt_publish(self, topic: str, payload, qos: int = 0, retain: bool = False) -> bool:
        """Thread-safe publish to MQTT"""
        if not hasattr(self, 'mqtt_handler'):
            safe_log("MQTT handler not initialized", logging.ERROR)
            return False
        
        # Convert payload to JSON if needed
        if not isinstance(payload, str):
            payload = json.dumps(payload)
        
        success = self.mqtt_handler.publish(topic, payload, qos, retain)
        
        if not success:
            safe_log(f"Failed to queue message for topic: {topic}", logging.WARNING)
        
        return success
    
    def _publish_camera(self, camera):
        """Publish camera discovery with thread-safe MQTT"""
        topic = f"{self.config.TOPIC_DISCOVERY}/{camera.id}"
        payload = camera.to_mqtt_payload()
        
        self.mqtt_publish(topic, payload, qos=1, retain=True)
        safe_log(f"Published camera discovery: {camera.name} ({camera.id})")
    
    def _publish_camera_status(self, camera, status: str):
        """Publish camera status update with thread-safe MQTT"""
        topic = f"{self.config.TOPIC_STATUS}/{camera.id}"
        payload = {
            'camera_id': camera.id,
            'status': status,
            'timestamp': datetime.now(timezone.utc).isoformat() + 'Z',
            'online': camera.online,
            'stream_active': camera.stream_active
        }
        
        self.mqtt_publish(topic, payload, qos=0, retain=False)
    
    def _publish_frigate_config(self):
        """Publish Frigate configuration with thread-safe MQTT"""
        with self.lock:
            # Get online cameras
            online_cameras = [c for c in self.cameras.values() if c.stream_active]
            
            if not online_cameras:
                safe_log("No active camera streams for Frigate config")
                return
            
            # Generate config
            config = self.frigate_generator.generate_config(online_cameras)
            
            # Publish as YAML string
            import yaml
            config_yaml = yaml.dump(config, default_flow_style=False)
            
            self.mqtt_publish(
                self.config.TOPIC_FRIGATE_CONFIG,
                config_yaml,
                qos=1,
                retain=True
            )
            
            # Trigger Frigate reload
            self.mqtt_publish(
                self.config.FRIGATE_RELOAD_TOPIC,
                json.dumps({'command': 'reload'}),
                qos=1,
                retain=False
            )
            
            safe_log(f"Published Frigate config for {len(online_cameras)} cameras")
    
    def _publish_health(self):
        """Publish health status with thread-safe MQTT"""
        with self.lock:
            online_cameras = [c for c in self.cameras.values() if c.online]
            streaming_cameras = [c for c in self.cameras.values() if c.stream_active]
            
            payload = {
                'node_id': self.config.NODE_ID,
                'service': 'camera_detector',
                'timestamp': datetime.now(timezone.utc).isoformat() + 'Z',
                'status': 'online' if self.mqtt_handler.is_connected() else 'degraded',
                'stats': {
                    'total_cameras': len(self.cameras),
                    'online_cameras': len(online_cameras),
                    'streaming_cameras': len(streaming_cameras),
                    'discovery_interval': self.config.DISCOVERY_INTERVAL,
                    'mac_tracking_enabled': self.config.MAC_TRACKING_ENABLED,
                    'mqtt_queue_size': self.mqtt_handler.message_queue.qsize() if hasattr(self, 'mqtt_handler') else 0,
                },
                'cameras': {
                    cam.id: {
                        'name': cam.name,
                        'ip': cam.ip,
                        'mac': cam.mac,
                        'online': cam.online,
                        'stream_active': cam.stream_active,
                        'last_seen': cam.last_seen.isoformat() + 'Z' if cam.last_seen else None
                    }
                    for cam in self.cameras.values()
                }
            }
            
            self.mqtt_publish(
                self.config.TOPIC_HEALTH,
                payload,
                qos=0,
                retain=True
            )
    
    def cleanup(self):
        """Clean shutdown with proper MQTT cleanup"""
        safe_log("Cleaning up Camera Detector Service")
        
        # Stop all background threads
        self._running = False
        
        # Cancel all active futures
        with self._executor_lock:
            for future in list(self._active_futures):
                future.cancel()
            self._active_futures.clear()
        
        # Stop the health report timer
        if hasattr(self, '_health_report_timer'):
            try:
                self._health_report_timer.cancel()
            except:
                pass
        
        # Wait for background threads to finish
        for thread in self._background_threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
        
        # Give threads a moment to finish logging
        time.sleep(0.5)
        
        # Shutdown executors with proper wait
        safe_log("Shutting down thread executor...")
        if hasattr(self, '_thread_executor') and self._thread_executor:
            try:
                self._thread_executor.shutdown(wait=True, cancel_futures=True)
            except Exception as e:
                safe_log(f"Error shutting down thread executor: {e}", logging.WARNING)
        
        safe_log("Shutting down process executor...")
        if hasattr(self, '_process_executor') and self._process_executor:
            try:
                self._process_executor.shutdown(wait=True, cancel_futures=True)
            except Exception as e:
                safe_log(f"Error shutting down process executor: {e}", logging.WARNING)
        
        # Mark all cameras offline
        with self.lock:
            for camera in self.cameras.values():
                if camera.online:
                    camera.online = False
                    try:
                        self._publish_camera_status(camera, "offline")
                    except:
                        pass  # Ignore errors during shutdown
        
        # Publish offline status
        try:
            if hasattr(self, 'mqtt_handler'):
                lwt_payload = json.dumps({
                    'node_id': self.config.NODE_ID,
                    'service': 'camera_detector',
                    'status': 'offline',
                    'timestamp': datetime.now(timezone.utc).isoformat() + 'Z'
                })
                self.mqtt_publish(
                    f"{self.config.TOPIC_HEALTH}/{self.config.NODE_ID}/lwt",
                    lwt_payload,
                    qos=1,
                    retain=True
                )
                
                # Give time for final messages to be sent
                time.sleep(1.0)
                
                # Stop MQTT handler
                self.mqtt_handler.stop()
        except Exception as e:
            safe_log(f"Error during MQTT cleanup: {e}")
        
        safe_log("Camera Detector Service cleanup complete")


# Monkey patch function to apply the stable MQTT implementation
def apply_stable_mqtt_patch():
    """Apply stable MQTT implementation to CameraDetector
    
    This function should be called before instantiating CameraDetector
    to replace the MQTT methods with stable implementations.
    """
    import detect
    
    # Store original methods for potential rollback
    original_methods = {
        '_setup_mqtt': detect.CameraDetector._setup_mqtt,
        '_publish_camera': detect.CameraDetector._publish_camera,
        '_publish_camera_status': detect.CameraDetector._publish_camera_status,
        '_publish_frigate_config': detect.CameraDetector._publish_frigate_config,
        '_publish_health': detect.CameraDetector._publish_health,
        'cleanup': detect.CameraDetector.cleanup,
    }
    
    # Apply the mixin methods
    detect.CameraDetector._setup_mqtt = StableMQTTMixin._setup_mqtt
    detect.CameraDetector.mqtt_publish = StableMQTTMixin.mqtt_publish
    detect.CameraDetector._publish_camera = StableMQTTMixin._publish_camera
    detect.CameraDetector._publish_camera_status = StableMQTTMixin._publish_camera_status
    detect.CameraDetector._publish_frigate_config = StableMQTTMixin._publish_frigate_config
    detect.CameraDetector._publish_health = StableMQTTMixin._publish_health
    detect.CameraDetector.cleanup = StableMQTTMixin.cleanup
    
    # Return original methods for rollback if needed
    return original_methods


# Alternative: Create a new class that inherits from CameraDetector
def create_stable_camera_detector():
    """Create CameraDetector class with stable MQTT
    
    Returns a new class that inherits from CameraDetector but uses
    the stable MQTT implementation.
    """
    import detect
    
    class StableCameraDetector(detect.CameraDetector, StableMQTTMixin):
        """Camera Detector with stable MQTT implementation"""
        
        def __init__(self):
            # Call parent init
            super().__init__()
            
        # The mixin methods will override the parent methods
    
    return StableCameraDetector


if __name__ == "__main__":
    # Example usage: Apply the patch and run
    import detect
    import time
    
    # Import safe_log into global namespace
    safe_log = detect.safe_log
    
    # Apply the stable MQTT patch
    apply_stable_mqtt_patch()
    
    # Now run the detector with stable MQTT
    detector = detect.CameraDetector()
    detector.run()