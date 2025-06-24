#!/usr/bin/env python3.12
"""Test cases for RTSP validation timeout handling.

This test suite verifies that the camera detector properly handles
hanging RTSP connections that could freeze the service.
"""

import unittest
import threading
import time
import socket
from unittest.mock import patch, MagicMock
import sys
import os
import cv2

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from camera_detector.detect import CameraDetector, Config


class TestRTSPValidationTimeout(unittest.TestCase):
    """Test RTSP validation timeout handling."""
    
    def setUp(self):
        """Set up test fixtures."""

        # Suppress OpenCV/FFMPEG warnings
        os.environ['OPENCV_FFMPEG_LOGLEVEL'] = 'quiet'
        os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'loglevel;quiet'
        self.test_camera_name = "TestCamera"
        self.test_rtsp_url = "rtsp://192.168.1.100:554/stream"
        
        # Create camera detector instance with all background tasks disabled
        with patch('camera_detector.detect.CameraDetector._setup_mqtt'), \
             patch('camera_detector.detect.CameraDetector._start_background_tasks'):
            self.detector = CameraDetector()
            # Override the timeout for testing
            self.detector.config.RTSP_TIMEOUT = 10  # 10 second timeout
            # Ensure no background threads are running
            self.detector._running = False
            
    def tearDown(self):
        """Clean up after tests."""
        # Ensure detector is stopped
        if hasattr(self, 'detector'):
            self.detector._running = False
            # Clean up any executors that might have been created
            if hasattr(self.detector, '_executor') and hasattr(self.detector._executor, 'shutdown'):
                try:
                    self.detector._executor.shutdown(wait=False)
                except:
                    pass
            if hasattr(self.detector, '_safe_executor') and hasattr(self.detector._safe_executor, 'shutdown'):
                try:
                    self.detector._safe_executor.shutdown(wait=False)
                except:
                    pass
        
    def test_hanging_rtsp_connection(self):
        """Test that validate_rtsp_stream doesn't hang indefinitely.
        
        This test simulates a hanging cv2.VideoCapture call that would
        normally freeze the entire service. The current implementation
        is vulnerable to this issue.
        """
        # Create a mock that simulates a hanging VideoCapture
        with patch('cv2.VideoCapture') as mock_video_capture:
            # Create a mock capture object
            mock_cap = MagicMock()
            
            # Simulate a hanging call by making VideoCapture sleep indefinitely
            def hanging_video_capture(url):
                # This simulates a network hang
                time.sleep(60)  # Sleep for 60 seconds (longer than any reasonable timeout)
                return mock_cap
            
            mock_video_capture.side_effect = hanging_video_capture
            
            # Record start time
            start_time = time.time()
            
            # This SHOULD timeout, but with current implementation it will hang
            # We'll run it in a thread and check if it completes in reasonable time
            result = [None]
            exception = [None]
            
            def run_validation():
                try:
                    result[0] = self.detector._validate_rtsp_stream(self.test_rtsp_url)
                except Exception as e:
                    exception[0] = e
            
            validation_thread = threading.Thread(target=run_validation)
            validation_thread.daemon = True
            validation_thread.start()
            
            # Wait for up to 15 seconds for the validation to complete
            validation_thread.join(timeout=15)
            
            # Check if thread is still alive (indicating a hang)
            if validation_thread.is_alive():
                self.fail("validate_rtsp_stream hung indefinitely - this is the bug we need to fix!")
            
            # If we get here, the validation completed (good!)
            elapsed_time = time.time() - start_time
            
            # Validation should have failed (returned False) due to timeout
            self.assertFalse(result[0], "Validation should have failed for hanging stream")
            
            # And it should have completed within reasonable time (< 15 seconds)
            self.assertLess(elapsed_time, 15, "Validation took too long to timeout")
            
    def test_normal_rtsp_validation(self):
        """Test that normal RTSP validation still works correctly."""
        # Test with a localhost URL that will fail quickly
        # Since we can't mock inside ProcessPoolExecutor, we test the actual behavior
        test_url = "rtsp://127.0.0.1:9999/stream"  # Non-existent local port
        
        # This should fail quickly (connection refused)
        start_time = time.time()
        result = self.detector._validate_rtsp_stream(test_url)
        elapsed = time.time() - start_time
        
        self.assertFalse(result, "Non-existent stream validation should fail")
        self.assertLess(elapsed, 5, "Validation should fail quickly for connection refused")
            
    def test_failed_rtsp_validation(self):
        """Test that failed RTSP connections are handled properly."""
        # Test with an invalid URL format
        invalid_url = "not-a-valid-rtsp-url"
        
        # This should fail quickly
        start_time = time.time()
        result = self.detector._validate_rtsp_stream(invalid_url)
        elapsed = time.time() - start_time
        
        self.assertFalse(result, "Invalid URL validation should return False")
        self.assertLess(elapsed, 5, "Invalid URL should fail quickly")
            
    def test_rtsp_validation_with_network_error(self):
        """Test RTSP validation with network errors."""
        with patch('camera_detector.detect._rtsp_validation_worker') as mock_worker:
            # Simulate a network error in the worker
            mock_worker.side_effect = Exception("Network unreachable")
            
            # This should handle the exception gracefully
            result = self.detector._validate_rtsp_stream(self.test_rtsp_url)
            
            self.assertFalse(result, "Network error should result in validation failure")


class TestRTSPValidationIntegration(unittest.TestCase):
    """Integration tests for RTSP validation with real network behavior."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_port = 55444  # Non-standard port to avoid conflicts
        
        # Create camera detector instance with all background tasks disabled
        with patch('camera_detector.detect.CameraDetector._setup_mqtt'), \
             patch('camera_detector.detect.CameraDetector._start_background_tasks'):
            self.detector = CameraDetector()
            # Override the timeout for testing
            self.detector.config.RTSP_TIMEOUT = 10  # 10 second timeout
            # Ensure no background threads are running
            self.detector._running = False
            
    def tearDown(self):
        """Clean up after tests."""
        # Ensure detector is stopped
        if hasattr(self, 'detector'):
            self.detector._running = False
            # Clean up any executors that might have been created
            if hasattr(self.detector, '_executor') and hasattr(self.detector._executor, 'shutdown'):
                try:
                    self.detector._executor.shutdown(wait=False)
                except:
                    pass
            if hasattr(self.detector, '_safe_executor') and hasattr(self.detector._safe_executor, 'shutdown'):
                try:
                    self.detector._safe_executor.shutdown(wait=False)
                except:
                    pass
        
    def test_connection_to_non_existent_host(self):
        """Test validation against a non-existent host.
        
        This should timeout quickly, not hang indefinitely.
        """
        # Use a non-routable IP address
        non_existent_url = "rtsp://192.0.2.1:554/stream"  # TEST-NET-1 (RFC 5737)
        
        start_time = time.time()
        result = self.detector._validate_rtsp_stream(non_existent_url)
        elapsed_time = time.time() - start_time
        
        self.assertFalse(result, "Connection to non-existent host should fail")
        self.assertLess(elapsed_time, 30, "Timeout should occur within 30 seconds")
        
    def test_connection_to_closed_port(self):
        """Test validation against a closed port.
        
        This should fail quickly with connection refused.
        """
        # Use localhost with a port we know is closed
        closed_port_url = f"rtsp://127.0.0.1:{self.test_port}/stream"
        
        # Ensure port is actually closed
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', self.test_port))
        sock.close()
        self.assertNotEqual(result, 0, f"Port {self.test_port} should be closed")
        
        start_time = time.time()
        result = self.detector._validate_rtsp_stream(closed_port_url)
        elapsed_time = time.time() - start_time
        
        self.assertFalse(result, "Connection to closed port should fail")
        self.assertLess(elapsed_time, 5, "Closed port should fail quickly")


if __name__ == '__main__':
    unittest.main()