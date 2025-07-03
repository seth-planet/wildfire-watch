#!/usr/bin/env python3.12
"""
Test Isolation Fixes for Wildfire Watch
Provides proper cleanup and isolation between tests
"""
import os
import time
import threading
import weakref
import logging
import psutil
import pytest
from typing import Set, Optional

logger = logging.getLogger(__name__)

class ThreadMonitor:
    """Monitor and cleanup threads between tests"""
    
    def __init__(self):
        self._initial_threads: Set[int] = set()
        self._test_threads: Set[int] = set()
        
    def start(self):
        """Record initial thread state"""
        self._initial_threads = {t.ident for t in threading.enumerate()}
        
    def cleanup(self, timeout: float = 5.0):
        """Clean up threads created during test"""
        current_threads = threading.enumerate()
        test_threads = [t for t in current_threads 
                       if t.ident not in self._initial_threads and t.is_alive()]
        
        # Try graceful shutdown first
        for thread in test_threads:
            if hasattr(thread, 'stop') and callable(thread.stop):
                thread.stop()
            elif hasattr(thread, 'cancel') and callable(thread.cancel):
                thread.cancel()
                
        # Wait for threads to finish
        start_time = time.time()
        while test_threads and time.time() - start_time < timeout:
            test_threads = [t for t in test_threads if t.is_alive()]
            if test_threads:
                time.sleep(0.1)
                
        # Log any remaining threads
        if test_threads:
            logger.warning(f"Failed to cleanup {len(test_threads)} threads: "
                         f"{[t.name for t in test_threads]}")
                         
        return len(test_threads) == 0

class ResourceTracker:
    """Track and cleanup system resources"""
    
    def __init__(self):
        self._process = psutil.Process()
        self._initial_fds = None
        self._initial_threads = None
        self._initial_connections = None
        
    def start(self):
        """Record initial resource state"""
        try:
            self._initial_fds = self._process.num_fds()
        except:
            self._initial_fds = None
            
        self._initial_threads = self._process.num_threads()
        
        try:
            self._initial_connections = len(self._process.connections())
        except:
            self._initial_connections = None
            
    def check_leaks(self) -> dict:
        """Check for resource leaks"""
        leaks = {}
        
        try:
            if self._initial_fds is not None:
                current_fds = self._process.num_fds()
                if current_fds > self._initial_fds:
                    leaks['file_descriptors'] = current_fds - self._initial_fds
        except:
            pass
            
        current_threads = self._process.num_threads()
        if current_threads > self._initial_threads:
            leaks['threads'] = current_threads - self._initial_threads
            
        try:
            if self._initial_connections is not None:
                current_connections = len(self._process.connections())
                if current_connections > self._initial_connections:
                    leaks['connections'] = current_connections - self._initial_connections
        except:
            pass
            
        return leaks

class ServiceStateManager:
    """Manage service state between tests"""
    
    def __init__(self):
        self._services = weakref.WeakSet()
        
    def register(self, service):
        """Register a service for cleanup"""
        self._services.add(service)
        
    def cleanup_all(self):
        """Clean up all registered services"""
        for service in list(self._services):
            try:
                # Stop MQTT client first
                if hasattr(service, "_mqtt_client"):
                    if hasattr(service._mqtt_client, 'loop_stop'):
                        service.mqtt_client.loop_stop()
                    if hasattr(service._mqtt_client, 'disconnect'):
                        service.mqtt_client.disconnect()
                        
                # Stop the service
                if hasattr(service, 'cleanup'):
                    service.cleanup()
                elif hasattr(service, 'stop'):
                    service.stop()
                elif hasattr(service, "_shutdown"):
                    service.shutdown()
                    
                # Clear any state
                if hasattr(service, 'cameras'):
                    service.cameras.clear()
                if hasattr(service, 'detections'):
                    service.detections.clear()
                if hasattr(service, '_shutdown'):
                    service._shutdown = True
                    
            except Exception as e:
                logger.error(f"Error cleaning up service: {e}")
                
        self._services.clear()

# Pytest fixtures
@pytest.fixture
def thread_monitor():
    """Monitor and cleanup threads"""
    monitor = ThreadMonitor()
    monitor.start()
    yield monitor
    monitor.cleanup()

@pytest.fixture
def resource_tracker():
    """Track system resources"""
    tracker = ResourceTracker()
    tracker.start()
    yield tracker
    leaks = tracker.check_leaks()
    if leaks:
        logger.warning(f"Resource leaks detected: {leaks}")

@pytest.fixture
def service_state_manager():
    """Manage service state"""
    manager = ServiceStateManager()
    yield manager
    manager.cleanup_all()

@pytest.fixture(autouse=True)
def ensure_clean_state(thread_monitor, resource_tracker, service_state_manager):
    """Ensure clean state between tests"""
    yield
    
    # Additional cleanup
    import gc
    gc.collect()
    
    # Clear any module-level state
    import sys
    for module in list(sys.modules.keys()):
        if module.startswith('fire_consensus') or module.startswith('camera_detector'):
            if hasattr(sys.modules[module], '_instances'):
                sys.modules[module]._instances.clear()
            if hasattr(sys.modules[module], '_state'):
                sys.modules[module]._state.clear()

# Improved fixtures for services
@pytest.fixture
def consensus_service_isolated(test_mqtt_broker, monkeypatch, service_state_manager, thread_monitor):
    """Create isolated FireConsensus service"""
    # Import here to avoid module-level state
    import sys
    if 'fire_consensus.consensus' in sys.modules:
        del sys.modules['fire_consensus.consensus']
    
    # Now import fresh
    from fire_consensus.consensus import FireConsensus
    
    # Get connection parameters
    conn_params = test_mqtt_broker.get_connection_params()
    
    # Set test environment
    monkeypatch.setenv("CONSENSUS_THRESHOLD", "2")
    monkeypatch.setenv("CAMERA_WINDOW", "10")
    monkeypatch.setenv("INCREASE_COUNT", "3")
    monkeypatch.setenv("DETECTION_COOLDOWN", "0.5")
    monkeypatch.setenv("MIN_CONFIDENCE", "0.7")
    monkeypatch.setenv("TELEMETRY_INTERVAL", "3600")  # Disable health timer
    monkeypatch.setenv("CLEANUP_INTERVAL", "3600")    # Disable cleanup timer
    monkeypatch.setenv("MQTT_BROKER", conn_params['host'])
    monkeypatch.setenv("MQTT_PORT", str(conn_params['port']))
    monkeypatch.setenv("MQTT_KEEPALIVE", "60")
    monkeypatch.setenv("MQTT_TLS", "false")
    
    # Create service
    service = FireConsensus()
    service_state_manager.register(service)
    
    # Wait for connection
    start_time = time.time()
    while not service._mqtt_connected and time.time() - start_time < 10:
        time.sleep(0.1)
    
    assert service._mqtt_connected, "Service must connect to MQTT broker"
    
    yield service
    
    # Comprehensive cleanup
    try:
        service._shutdown = True
        
        # Cancel timers first
        if hasattr(service, '_health_timer') and service._health_timer:
            service._health_timer.cancel()
        if hasattr(service, '_cleanup_timer') and service._cleanup_timer:
            service._cleanup_timer.cancel()
            
        # Stop MQTT
        if hasattr(service, "_mqtt_client"):
            service.mqtt_client.loop_stop()
            service.mqtt_client.disconnect()
            
        # Wait for threads
        time.sleep(0.2)
        
        # Final cleanup
        service.cleanup()
        
    except Exception as e:
        logger.error(f"Error during consensus cleanup: {e}")

@pytest.fixture
def camera_detector_isolated(test_mqtt_broker, monkeypatch, service_state_manager, thread_monitor):
    """Create isolated CameraDetector"""
    # Import fresh
    import sys
    if 'camera_detector.detect' in sys.modules:
        del sys.modules['camera_detector.detect']
    
    from camera_detector.detect import CameraDetector
    
    # Get connection parameters
    conn_params = test_mqtt_broker.get_connection_params()
    
    # Set test environment
    monkeypatch.setenv("MQTT_BROKER", conn_params['host'])
    monkeypatch.setenv("MQTT_PORT", str(conn_params['port']))
    monkeypatch.setenv("DISCOVERY_INTERVAL", "3600")  # Disable discovery
    monkeypatch.setenv("HEALTH_CHECK_INTERVAL", "3600")  # Disable health checks
    
    # Create detector
    detector = CameraDetector()
    service_state_manager.register(detector)
    
    # Stop background tasks immediately
    detector.stop_background_tasks()
    
    # Clear any initial state
    detector.cameras.clear()
    detector.mac_tracker.mac_to_ip.clear()
    
    # Wait for connection
    start_time = time.time()
    while not detector._mqtt_connected and time.time() - start_time < 10:
        time.sleep(0.1)
    
    assert detector._mqtt_connected, "Detector must connect to MQTT broker"
    
    yield detector
    
    # Cleanup
    try:
        detector.stop_background_tasks()
        
        if hasattr(detector, "_mqtt_client"):
            detector.mqtt_client.loop_stop()
            detector.mqtt_client.disconnect()
            
        if hasattr(detector, 'executor'):
            detector._executor.shutdown(wait=False)
            
        time.sleep(0.2)
        
    except Exception as e:
        logger.error(f"Error during detector cleanup: {e}")