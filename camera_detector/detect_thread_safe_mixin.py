#!/usr/bin/env python3.12
"""Thread Safety Mixin for Camera Detector

This module provides a mixin class that adds thread safety to the existing
CameraDetector class without requiring a complete rewrite.
"""

import threading
import time
import logging
from typing import Dict, Set, Optional, Tuple, List, Any
from functools import wraps

logger = logging.getLogger(__name__)

class ThreadSafeCameraDetectorMixin:
    """Mixin to add thread safety to CameraDetector
    
    This mixin overrides methods that access shared state to ensure
    thread-safe operations. It should be mixed in with the existing
    CameraDetector class.
    
    Usage:
        class ThreadSafeCameraDetector(ThreadSafeCameraDetectorMixin, CameraDetector):
            pass
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize thread safety additions"""
        # Call parent __init__ first
        super().__init__(*args, **kwargs)
        
        # Ensure we have the lock (parent may not have initialized it)
        if not hasattr(self, 'lock'):
            self.lock = threading.RLock()
        
        # Add thread-safe wrappers for camera operations
        self._wrap_camera_methods()
        
        logger.info("Thread safety mixin initialized")
    
    def _wrap_camera_methods(self):
        """Wrap methods that need thread safety"""
        # Methods that modify shared state
        unsafe_methods = [
            '_add_camera',
            '_remove_camera', 
            '_update_camera_status',
            '_update_discovery_state'
        ]
        
        for method_name in unsafe_methods:
            if hasattr(self, method_name):
                original = getattr(self, method_name)
                wrapped = self._make_thread_safe(original)
                setattr(self, method_name, wrapped)
    
    def _make_thread_safe(self, func):
        """Create thread-safe wrapper for a method"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.lock:
                return func(*args, **kwargs)
        return wrapper
    
    # ─────────────────────────────────────────────────────────────
    # Thread-Safe Camera Management
    # ─────────────────────────────────────────────────────────────
    
    def add_camera(self, camera) -> bool:
        """Thread-safe camera addition
        
        Args:
            camera: Camera object to add
            
        Returns:
            True if camera was new, False if updated
        """
        with self.lock:
            is_new = camera.id not in self.cameras
            self.cameras[camera.id] = camera
            self.known_camera_ips.add(camera.ip)
            
            # Update MAC tracker
            if camera.mac and camera.ip:
                self.mac_tracker.update(camera.mac, camera.ip)
            
            return is_new
    
    def remove_camera(self, camera_id: str) -> bool:
        """Thread-safe camera removal
        
        Args:
            camera_id: ID of camera to remove
            
        Returns:
            True if camera was removed
        """
        with self.lock:
            camera = self.cameras.pop(camera_id, None)
            if camera:
                self.known_camera_ips.discard(camera.ip)
                logger.info(f"Removed camera: {camera_id}")
                return True
            return False
    
    def get_camera(self, camera_id: str) -> Optional[Any]:
        """Thread-safe camera retrieval
        
        Args:
            camera_id: ID of camera to get
            
        Returns:
            Camera object or None
        """
        with self.lock:
            return self.cameras.get(camera_id)
    
    def get_camera_by_ip(self, ip: str) -> Optional[Any]:
        """Thread-safe camera retrieval by IP
        
        Args:
            ip: IP address to search
            
        Returns:
            Camera object or None
        """
        with self.lock:
            for camera in self.cameras.values():
                if camera.ip == ip:
                    return camera
            return None
    
    def get_camera_by_mac(self, mac: str) -> Optional[Any]:
        """Thread-safe camera retrieval by MAC
        
        Args:
            mac: MAC address to search
            
        Returns:
            Camera object or None
        """
        with self.lock:
            for camera in self.cameras.values():
                if camera.mac == mac:
                    return camera
            return None
    
    def get_all_cameras(self) -> List[Any]:
        """Thread-safe retrieval of all cameras
        
        Returns:
            List of all camera objects
        """
        with self.lock:
            return list(self.cameras.values())
    
    def get_online_cameras(self) -> List[Any]:
        """Thread-safe retrieval of online cameras
        
        Returns:
            List of online camera objects
        """
        with self.lock:
            return [c for c in self.cameras.values() if c.online]
    
    def update_camera_status(self, camera_id: str, online: bool, 
                           stream_active: Optional[bool] = None) -> bool:
        """Thread-safe camera status update
        
        Args:
            camera_id: Camera to update
            online: Online status
            stream_active: Optional stream status
            
        Returns:
            True if camera was found and updated
        """
        with self.lock:
            camera = self.cameras.get(camera_id)
            if camera:
                camera.online = online
                camera.last_seen = time.time()
                if stream_active is not None:
                    camera.stream_active = stream_active
                return True
            return False
    
    # ─────────────────────────────────────────────────────────────
    # Thread-Safe Discovery State
    # ─────────────────────────────────────────────────────────────
    
    def increment_discovery_count(self) -> int:
        """Thread-safe discovery count increment
        
        Returns:
            New discovery count
        """
        with self.lock:
            self.discovery_count += 1
            return self.discovery_count
    
    def update_discovery_state(self, camera_count: int) -> Tuple[bool, int]:
        """Thread-safe discovery state update
        
        Args:
            camera_count: Current camera count
            
        Returns:
            Tuple of (is_steady_state, stable_count)
        """
        with self.lock:
            if camera_count == self.last_camera_count:
                self.stable_count += 1
            else:
                self.stable_count = 0
            
            self.last_camera_count = camera_count
            
            # Enter steady state after stable discoveries
            if self.stable_count >= 3 and not self.is_steady_state:
                self.is_steady_state = True
                logger.info("Entering steady state discovery mode")
            
            return self.is_steady_state, self.stable_count
    
    def reset_discovery_state(self):
        """Thread-safe reset of discovery state"""
        with self.lock:
            self.discovery_count = 0
            self.stable_count = 0
            self.is_steady_state = False
            self.last_full_discovery = 0
    
    def get_discovery_stats(self) -> Dict[str, Any]:
        """Thread-safe retrieval of discovery statistics
        
        Returns:
            Dictionary of discovery stats
        """
        with self.lock:
            return {
                'discovery_count': self.discovery_count,
                'camera_count': len(self.cameras),
                'online_count': sum(1 for c in self.cameras.values() if c.online),
                'is_steady_state': self.is_steady_state,
                'stable_count': self.stable_count,
                'known_ips': len(self.known_camera_ips)
            }
    
    # ─────────────────────────────────────────────────────────────
    # Thread-Safe Discovery Methods  
    # ─────────────────────────────────────────────────────────────
    
    def _discovery_loop(self):
        """Thread-safe discovery loop override"""
        while self._running:
            try:
                # Increment discovery count atomically
                count = self.increment_discovery_count()
                logger.info(f"Starting discovery cycle #{count}")
                
                # Get current camera count before discovery
                with self.lock:
                    initial_count = len(self.cameras)
                
                # Run discovery (I/O outside lock)
                self._perform_discovery()
                
                # Update discovery state
                with self.lock:
                    final_count = len(self.cameras)
                
                is_steady, stable = self.update_discovery_state(final_count)
                
                # Determine sleep interval
                interval = self.config.DISCOVERY_INTERVAL
                if self.config.SMART_DISCOVERY_ENABLED and is_steady:
                    interval *= 2  # Double interval in steady state
                
                logger.info(f"Discovery complete. Cameras: {initial_count} -> {final_count}. "
                          f"Sleeping {interval}s")
                
                # Check for shutdown during sleep
                for _ in range(int(interval)):
                    if not self._running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Discovery loop error: {e}", exc_info=True)
                time.sleep(30)
    
    def _health_check_loop(self):
        """Thread-safe health check loop override"""
        while self._running:
            try:
                current_time = time.time()
                
                # Get cameras to check (snapshot to avoid holding lock)
                with self.lock:
                    cameras_to_check = [
                        (cam.id, cam.ip, cam.last_seen) 
                        for cam in self.cameras.values()
                    ]
                
                # Check each camera (I/O outside lock)
                for camera_id, ip, last_seen in cameras_to_check:
                    if current_time - last_seen > self.config.OFFLINE_THRESHOLD:
                        # Update status atomically
                        if self.update_camera_status(camera_id, online=False, stream_active=False):
                            logger.warning(f"Camera {camera_id} at {ip} is offline")
                            self._publish_camera_status_safe(camera_id, "offline")
                
                # Sleep with shutdown check
                for _ in range(30):
                    if not self._running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Health check error: {e}", exc_info=True)
                time.sleep(30)
    
    def _publish_camera_status_safe(self, camera_id: str, status: str):
        """Thread-safe camera status publishing"""
        try:
            # Get camera data safely
            with self.lock:
                camera = self.cameras.get(camera_id)
                if not camera:
                    return
                
                # Create status message
                status_msg = {
                    'camera_id': camera_id,
                    'ip': camera.ip,
                    'mac': camera.mac,
                    'status': status,
                    'timestamp': time.time()
                }
            
            # Publish outside lock
            if self.mqtt_connected:
                topic = f"{self.config.TOPIC_STATUS}/{camera_id}"
                self.mqtt_client.publish(topic, json.dumps(status_msg), retain=False)
                
        except Exception as e:
            logger.error(f"Failed to publish camera status: {e}")
    
    def _perform_discovery(self):
        """Perform actual discovery (called with I/O outside lock)"""
        # This method would contain the actual discovery logic
        # It's a placeholder showing where discovery happens
        logger.debug("Performing camera discovery...")
        
        # The parent class's discovery methods would be called here
        # They should acquire the lock only when updating shared state