#!/usr/bin/env python3.12
"""
Enhanced Test Isolation with Garbage Collection and Thread Cleanup
Provides comprehensive isolation between tests to prevent resource leaks
"""
import os
import gc
import time
import threading
import weakref
import logging
import psutil
import pytest
import uuid
from typing import Set, Optional, Dict, Any, List
import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Enhanced Thread Management with Kill Support
# ─────────────────────────────────────────────────────────────

class EnhancedThreadManager:
    """Enhanced thread management with automatic cleanup and kill support"""
    
    def __init__(self):
        self._initial_threads: Set[int] = set()
        self._test_threads: Set[threading.Thread] = weakref.WeakSet()
        self._lock = threading.Lock()
        self._thread_start_count = 0
        self._thread_stop_count = 0
        
    def start(self):
        """Record initial thread state"""
        self._initial_threads = {t.ident for t in threading.enumerate()}
        self._thread_start_count = threading.active_count()
        logger.debug(f"Initial threads: {len(self._initial_threads)}")
        
    def register_thread(self, thread: threading.Thread):
        """Register a thread for tracking"""
        with self._lock:
            self._test_threads.add(thread)
            
    def get_orphaned_threads(self) -> List[threading.Thread]:
        """Get list of threads created during the test"""
        current_threads = threading.enumerate()
        orphaned = [t for t in current_threads 
                   if t.ident not in self._initial_threads and t.is_alive()]
        return orphaned
            
    def cleanup(self, timeout: float = 5.0, force_kill: bool = True):
        """Clean up all test threads with optional force kill"""
        orphaned_threads = self.get_orphaned_threads()
        
        if not orphaned_threads:
            return True
            
        logger.debug(f"Cleaning up {len(orphaned_threads)} orphaned threads")
        
        # First try graceful shutdown
        for thread in orphaned_threads:
            if hasattr(thread, 'stop') and callable(thread.stop):
                try:
                    thread.stop()
                except Exception as e:
                    logger.debug(f"Error stopping thread {thread.name}: {e}")
            elif hasattr(thread, 'cancel') and callable(thread.cancel):
                try:
                    thread.cancel()
                except Exception as e:
                    logger.debug(f"Error cancelling thread {thread.name}: {e}")
                    
            # Set stop flags if available
            if hasattr(thread, '_stop_event') and hasattr(thread._stop_event, 'set'):
                thread._stop_event.set()
            if hasattr(thread, 'shutdown') and callable(thread.shutdown):
                try:
                    thread.shutdown()
                except Exception:
                    pass
        
        # Wait for threads to finish
        start_time = time.time()
        while orphaned_threads and time.time() - start_time < timeout:
            orphaned_threads = [t for t in orphaned_threads if t.is_alive()]
            if orphaned_threads:
                time.sleep(0.1)
                
        # Force kill if requested and threads still alive
        if force_kill and orphaned_threads:
            logger.warning(f"Force killing {len(orphaned_threads)} stubborn threads: "
                         f"{[t.name for t in orphaned_threads]}")
            # Note: Python doesn't support true thread killing, but we can
            # try to interrupt blocking operations
            for thread in orphaned_threads:
                if hasattr(thread, '_target') and thread._target:
                    # Try to close any file descriptors or sockets
                    try:
                        frame = thread._target.__code__
                        for var_name in frame.co_varnames:
                            if 'socket' in var_name.lower() or 'file' in var_name.lower():
                                # This is a heuristic approach
                                pass
                    except Exception:
                        pass
                        
        final_orphaned = self.get_orphaned_threads()
        if final_orphaned:
            logger.error(f"Failed to cleanup {len(final_orphaned)} threads: "
                        f"{[t.name for t in final_orphaned]}")
                         
        return len(final_orphaned) == 0
        
    def report_thread_leaks(self) -> Dict[str, Any]:
        """Generate thread leak report"""
        orphaned = self.get_orphaned_threads()
        return {
            'initial_count': len(self._initial_threads),
            'current_count': threading.active_count(),
            'orphaned_count': len(orphaned),
            'orphaned_threads': [
                {
                    'name': t.name,
                    'daemon': t.daemon,
                    'ident': t.ident,
                    'target': str(getattr(t, '_target', 'unknown'))
                }
                for t in orphaned
            ]
        }

# ─────────────────────────────────────────────────────────────
# Resource Monitoring and Cleanup
# ─────────────────────────────────────────────────────────────

class ResourceMonitor:
    """Monitor system resources and detect leaks"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.initial_state = {}
        
    def capture_state(self):
        """Capture current resource state"""
        self.initial_state = {
            'memory': self.process.memory_info().rss,
            'threads': self.process.num_threads(),
            'fds': len(self.process.open_files()) if hasattr(self.process, 'open_files') else 0,
            'connections': len(self.process.connections()) if hasattr(self.process, 'connections') else 0
        }
        
    def get_leaks(self) -> Dict[str, Any]:
        """Detect resource leaks"""
        current_state = {
            'memory': self.process.memory_info().rss,
            'threads': self.process.num_threads(),
            'fds': len(self.process.open_files()) if hasattr(self.process, 'open_files') else 0,
            'connections': len(self.process.connections()) if hasattr(self.process, 'connections') else 0
        }
        
        leaks = {}
        if current_state['memory'] > self.initial_state['memory'] * 1.5:  # 50% increase
            leaks['memory'] = {
                'initial': self.initial_state['memory'],
                'current': current_state['memory'],
                'increase_pct': ((current_state['memory'] - self.initial_state['memory']) 
                               / self.initial_state['memory'] * 100)
            }
            
        for resource in ['threads', 'fds', 'connections']:
            if current_state[resource] > self.initial_state[resource]:
                leaks[resource] = {
                    'initial': self.initial_state[resource],
                    'current': current_state[resource],
                    'leaked': current_state[resource] - self.initial_state[resource]
                }
                
        return leaks

# ─────────────────────────────────────────────────────────────
# Comprehensive Test Isolation Fixture
# ─────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def comprehensive_test_isolation():
    """
    Comprehensive test isolation with:
    - Thread monitoring and cleanup
    - Forced garbage collection
    - Resource leak detection
    - MQTT client cleanup
    - Timer cleanup
    """
    thread_manager = EnhancedThreadManager()
    resource_monitor = ResourceMonitor()
    
    # Capture initial state
    thread_manager.start()
    resource_monitor.capture_state()
    
    # Force initial garbage collection
    gc.collect()
    
    yield
    
    # Post-test cleanup
    try:
        # 1. Stop all telemetry timers
        try:
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cam_telemetry"))
            import telemetry
            if hasattr(telemetry, 'shutdown_telemetry'):
                telemetry.shutdown_telemetry()
        except ImportError:
            pass
            
        # 2. Cleanup all Timer threads
        for thread in threading.enumerate():
            if isinstance(thread, threading.Timer) and thread.is_alive():
                thread.cancel()
                
        # 3. Force close all MQTT clients
        for obj in gc.get_objects():
            if isinstance(obj, mqtt.Client):
                try:
                    if obj.is_connected():
                        obj.disconnect()
                    obj.loop_stop()
                except Exception:
                    pass
                    
        # 4. Clean up orphaned threads
        thread_manager.cleanup(timeout=5.0, force_kill=True)
        
        # 5. Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
            time.sleep(0.1)
            
        # 6. Report any leaks
        thread_leaks = thread_manager.report_thread_leaks()
        if thread_leaks['orphaned_count'] > 0:
            logger.warning(f"Thread leaks detected: {thread_leaks}")
            
        resource_leaks = resource_monitor.get_leaks()
        if resource_leaks:
            logger.warning(f"Resource leaks detected: {resource_leaks}")
            
    except Exception as e:
        logger.error(f"Error during test cleanup: {e}")

# ─────────────────────────────────────────────────────────────
# Service-Specific Cleanup Helpers
# ─────────────────────────────────────────────────────────────

def cleanup_camera_detector():
    """Clean up CameraDetector resources"""
    try:
        import sys
        if 'camera_detector.detect' in sys.modules:
            module = sys.modules['camera_detector.detect']
            if hasattr(module, 'CameraDetector'):
                # Find all instances via garbage collector
                for obj in gc.get_objects():
                    if type(obj).__name__ == 'CameraDetector':
                        if hasattr(obj, 'cleanup'):
                            obj.cleanup()
    except Exception as e:
        logger.debug(f"Error cleaning up CameraDetector: {e}")

def cleanup_fire_consensus():
    """Clean up FireConsensus resources"""
    try:
        import sys
        if 'fire_consensus.consensus' in sys.modules:
            module = sys.modules['fire_consensus.consensus']
            if hasattr(module, 'FireConsensus'):
                for obj in gc.get_objects():
                    if type(obj).__name__ == 'FireConsensus':
                        if hasattr(obj, 'cleanup'):
                            obj.cleanup()
    except Exception as e:
        logger.debug(f"Error cleaning up FireConsensus: {e}")

def cleanup_gpio_trigger():
    """Clean up GPIO trigger resources"""
    try:
        import sys
        if 'gpio_trigger.trigger' in sys.modules:
            module = sys.modules['gpio_trigger.trigger']
            if hasattr(module, 'PumpController'):
                for obj in gc.get_objects():
                    if type(obj).__name__ == 'PumpController':
                        if hasattr(obj, 'cleanup'):
                            obj.cleanup()
    except Exception as e:
        logger.debug(f"Error cleaning up GPIO trigger: {e}")

# ─────────────────────────────────────────────────────────────
# Enhanced Fixture for Service Cleanup
# ─────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def cleanup_services():
    """Ensure all services are properly cleaned up after each test"""
    yield
    
    # Clean up specific services
    cleanup_camera_detector()
    cleanup_fire_consensus()
    cleanup_gpio_trigger()
    
    # Force module reloading for next test
    modules_to_reload = [
        'camera_detector.detect',
        'fire_consensus.consensus',
        'gpio_trigger.trigger',
        'cam_telemetry.telemetry'
    ]
    
    import sys
    for module_name in modules_to_reload:
        if module_name in sys.modules:
            del sys.modules[module_name]