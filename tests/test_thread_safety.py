#!/usr/bin/env python3.12
"""Thread Safety Tests for Wildfire Watch Services

This module tests the thread safety implementations to ensure
proper synchronization and prevention of race conditions.
"""

import pytest
import threading
import time
import random
from unittest.mock import Mock, patch
from collections import defaultdict

from camera_detector.thread_safety import (
    ThreadSafeDict, ThreadSafeSet, ThreadSafeCounter,
    ThreadSafeCameraRegistry, PeriodicTask
)
from fire_consensus.thread_safety import (
    ThreadSafeDetectionHistory, ThreadSafeObjectTracker,
    ThreadSafeCameraStates, ThreadSafeConsensusState
)

class TestThreadSafeDict:
    """Test ThreadSafeDict implementation"""
    
    def test_concurrent_updates(self):
        """Test concurrent dictionary updates"""
        ts_dict = ThreadSafeDict()
        errors = []
        
        def update_dict(start, count):
            try:
                for i in range(count):
                    key = f"key_{start + i}"
                    ts_dict.set(key, i)
                    # Simulate some work
                    time.sleep(random.uniform(0, 0.001))
                    
                    # Read and verify
                    value = ts_dict.get(key)
                    assert value == i
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=update_dict, args=(i * 100, 100))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Check results
        assert len(errors) == 0
        assert len(ts_dict) == 1000
    
    def test_atomic_pop(self):
        """Test atomic pop operation"""
        ts_dict = ThreadSafeDict()
        
        # Add items
        for i in range(100):
            ts_dict.set(f"key_{i}", i)
        
        popped_values = []
        
        def pop_items():
            for i in range(10):
                key = f"key_{random.randint(0, 99)}"
                value = ts_dict.pop(key, None)
                if value is not None:
                    popped_values.append(value)
        
        # Concurrent pops
        threads = [threading.Thread(target=pop_items) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Each value should be popped at most once
        assert len(popped_values) == len(set(popped_values))

class TestThreadSafeSet:
    """Test ThreadSafeSet implementation"""
    
    def test_concurrent_adds(self):
        """Test concurrent set additions"""
        ts_set = ThreadSafeSet()
        
        def add_items(start, count):
            for i in range(count):
                ts_set.add(start + i)
        
        threads = []
        for i in range(10):
            t = threading.Thread(target=add_items, args=(i * 100, 100))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # All items should be added
        assert len(ts_set) == 1000
        
        # Verify all items present
        for i in range(1000):
            assert i in ts_set

class TestThreadSafeCounter:
    """Test ThreadSafeCounter implementation"""
    
    def test_concurrent_increments(self):
        """Test concurrent counter increments"""
        counter = ThreadSafeCounter()
        
        def increment_many(count):
            for _ in range(count):
                counter.increment()
        
        # Create threads that increment
        threads = []
        increments_per_thread = 1000
        num_threads = 10
        
        for _ in range(num_threads):
            t = threading.Thread(target=increment_many, args=(increments_per_thread,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Counter should equal total increments
        assert counter.get() == increments_per_thread * num_threads

class TestThreadSafeCameraRegistry:
    """Test ThreadSafeCameraRegistry implementation"""
    
    def test_concurrent_camera_operations(self):
        """Test concurrent camera add/remove/update"""
        registry = ThreadSafeCameraRegistry()
        
        # Mock camera class
        class MockCamera:
            def __init__(self, id, ip, mac=None):
                self.id = id
                self.ip = ip
                self.mac = mac
                self.online = True
        
        def add_cameras(start_id, count):
            for i in range(count):
                camera = MockCamera(
                    f"cam_{start_id + i}",
                    f"192.168.1.{start_id + i}",
                    f"00:11:22:33:44:{start_id + i:02x}"
                )
                registry.add_camera(camera)
        
        def remove_cameras(start_id, count):
            for i in range(count):
                registry.remove_camera(f"cam_{start_id + i}")
        
        # Add cameras concurrently
        add_threads = []
        for i in range(5):
            t = threading.Thread(target=add_cameras, args=(i * 20, 20))
            add_threads.append(t)
            t.start()
        
        for t in add_threads:
            t.join()
        
        # Verify all added
        assert registry.count() == 100
        
        # Remove some concurrently
        remove_threads = []
        for i in range(5):
            t = threading.Thread(target=remove_cameras, args=(i * 10, 5))
            remove_threads.append(t)
            t.start()
        
        for t in remove_threads:
            t.join()
        
        # Verify correct count
        assert registry.count() == 75
    
    def test_ip_update_atomicity(self):
        """Test atomic IP updates"""
        registry = ThreadSafeCameraRegistry()
        
        class MockCamera:
            def __init__(self, id, ip):
                self.id = id
                self.ip = ip
                self.mac = None
                self.online = True
        
        # Add camera
        camera = MockCamera("cam1", "192.168.1.100")
        registry.add_camera(camera)
        
        success_count = [0]
        
        def try_update_ip(old_ip, new_ip):
            if registry.update_ip("cam1", old_ip, new_ip):
                success_count[0] += 1
        
        # Multiple threads try to update
        threads = []
        threads.append(threading.Thread(target=try_update_ip, args=("192.168.1.100", "192.168.1.101")))
        threads.append(threading.Thread(target=try_update_ip, args=("192.168.1.100", "192.168.1.102")))
        threads.append(threading.Thread(target=try_update_ip, args=("192.168.1.100", "192.168.1.103")))
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Only one should succeed
        assert success_count[0] == 1

class TestThreadSafeDetectionHistory:
    """Test ThreadSafeDetectionHistory implementation"""
    
    def test_concurrent_detection_adds(self):
        """Test concurrent detection additions"""
        history = ThreadSafeDetectionHistory()
        
        # Mock detection
        class MockDetection:
            def __init__(self, camera_id, timestamp):
                self.camera_id = camera_id
                self.timestamp = timestamp
                self.confidence = 0.9
                self.object_type = 'fire'
        
        def add_detections(camera_id, count):
            for i in range(count):
                detection = MockDetection(camera_id, time.time())
                history.add_detection(camera_id, detection)
                time.sleep(0.001)  # Small delay
        
        # Multiple threads add detections
        threads = []
        for i in range(5):
            camera_id = f"cam_{i}"
            t = threading.Thread(target=add_detections, args=(camera_id, 20))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Verify all detections added
        assert history.get_camera_count() == 5
        assert history.get_total_detections() == 100
    
    def test_cleanup_during_additions(self):
        """Test cleanup while detections are being added"""
        history = ThreadSafeDetectionHistory(_time_window=1.0)  # 1 second window
        
        class MockDetection:
            def __init__(self, timestamp):
                self.timestamp = timestamp
        
        stop_flag = threading.Event()
        
        def add_detections():
            while not stop_flag.is_set():
                detection = MockDetection(time.time())
                history.add_detection("cam1", detection)
                time.sleep(0.01)
        
        def cleanup_loop():
            while not stop_flag.is_set():
                history.cleanup_old_detections(time.time())
                time.sleep(0.1)
        
        # Start threads
        add_thread = threading.Thread(target=add_detections)
        cleanup_thread = threading.Thread(target=cleanup_loop)
        
        add_thread.start()
        cleanup_thread.start()
        
        # Run for 2 seconds
        time.sleep(2)
        stop_flag.set()
        
        add_thread.join()
        cleanup_thread.join()
        
        # Should only have recent detections
        recent = history.get_recent_detections("cam1", time.time())
        for detection in recent:
            assert time.time() - detection.timestamp < 1.0

class TestThreadSafeConsensusState:
    """Test ThreadSafeConsensusState implementation"""
    
    def test_cooldown_enforcement(self):
        """Test that cooldown is properly enforced"""
        consensus = ThreadSafeConsensusState(threshold=2, cooldown=1.0)
        
        current_time = time.time()
        
        # First trigger should be allowed
        assert consensus.can_trigger(current_time)
        consensus.record_trigger(current_time)
        
        # Immediate second trigger should be blocked
        assert not consensus.can_trigger(current_time + 0.5)
        
        # After cooldown should be allowed
        assert consensus.can_trigger(current_time + 1.1)
    
    def test_concurrent_trigger_checks(self):
        """Test concurrent trigger checks"""
        consensus = ThreadSafeConsensusState(threshold=2, cooldown=1.0)
        
        trigger_count = [0]
        
        def try_trigger():
            current = time.time()
            if consensus.can_trigger(current):
                consensus.record_trigger(current)
                trigger_count[0] += 1
        
        # Multiple threads try to trigger
        threads = []
        for _ in range(10):
            t = threading.Thread(target=try_trigger)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Only one should succeed
        assert trigger_count[0] == 1
        assert consensus.get_stats()['triggers_sent'] == 1

class TestPeriodicTask:
    """Test PeriodicTask implementation"""
    
    def test_periodic_execution(self):
        """Test that task executes periodically"""
        execution_count = [0]
        
        def task_function():
            execution_count[0] += 1
        
        task = PeriodicTask(interval=0.1, function=task_function)
        task.start()
        
        # Wait for multiple executions
        time.sleep(0.55)
        task.stop()
        
        # Should have executed ~5 times
        assert 4 <= execution_count[0] <= 6
    
    def test_stop_prevents_execution(self):
        """Test that stop prevents further execution"""
        execution_times = []
        
        def task_function():
            execution_times.append(time.time())
        
        task = PeriodicTask(interval=0.1, function=task_function)
        task.start()
        
        time.sleep(0.25)
        stop_time = time.time()
        task.stop()
        
        time.sleep(0.25)
        
        # No executions should occur after stop
        for exec_time in execution_times:
            assert exec_time < stop_time + 0.05  # Small margin for timing

class TestRaceConditionScenarios:
    """Test specific race condition scenarios"""
    
    def test_camera_discovery_race(self):
        """Test race condition in camera discovery"""
        registry = ThreadSafeCameraRegistry()
        
        class MockCamera:
            def __init__(self, id, ip):
                self.id = id
                self.ip = ip
                self.mac = None
                self.online = True
        
        def discover_camera(ip):
            # Simulate discovery delay
            time.sleep(random.uniform(0, 0.01))
            
            # Check if already discovered
            existing = registry.get_by_ip(ip)
            if not existing:
                camera = MockCamera(f"cam_{ip}", ip)
                registry.add_camera(camera)
        
        # Multiple threads discover same camera
        ip = "192.168.1.100"
        threads = []
        for _ in range(10):
            t = threading.Thread(target=discover_camera, args=(ip,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Should only have one camera
        assert registry.count() == 1
    
    def test_detection_consensus_race(self):
        """Test race condition in consensus checking"""
        history = ThreadSafeDetectionHistory()
        consensus_state = ThreadSafeConsensusState(threshold=2, cooldown=10.0)
        
        class MockDetection:
            def __init__(self, camera_id):
                self.camera_id = camera_id
                self.timestamp = time.time()
        
        trigger_count = [0]
        
        def process_detection(camera_id):
            # Add detection
            detection = MockDetection(camera_id)
            history.add_detection(camera_id, detection)
            
            # Check consensus
            current_time = time.time()
            all_detections = history.get_all_recent_detections(current_time)
            
            if len(all_detections) >= 2:  # Threshold met
                if consensus_state.can_trigger(current_time):
                    consensus_state.record_trigger(current_time)
                    trigger_count[0] += 1
        
        # Multiple cameras detect fire simultaneously
        threads = []
        for i in range(5):
            t = threading.Thread(target=process_detection, args=(f"cam_{i}",))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Should trigger exactly once
        assert trigger_count[0] == 1

def test_thread_safety_mixin_integration():
    """Test integration of thread safety mixins"""
    from camera_detector.detect import CameraDetector
    from camera_detector.detect_thread_safe_mixin import ThreadSafeCameraDetectorMixin
    
    # This would test the actual integration
    # For now, just verify imports work
    assert ThreadSafeCameraDetectorMixin is not None