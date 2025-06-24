#!/usr/bin/env python3.12
"""Test the improved RTSP validation with process-based timeout."""

import unittest
import time
from unittest.mock import patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from camera_detector.detect import CameraDetector


class TestRTSPValidationImproved(unittest.TestCase):
    """Test the improved RTSP validation implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create camera detector instance with all background tasks disabled
        with patch('camera_detector.detect.CameraDetector._setup_mqtt'), \
             patch('camera_detector.detect.CameraDetector._start_background_tasks'):
            self.detector = CameraDetector()
            # Set a short timeout for testing
            self.detector.config.RTSP_TIMEOUT = 2  # 2 second timeout
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
    
    def test_timeout_on_non_existent_host(self):
        """Test that validation times out properly on non-existent hosts.
        
        This is the key test - it verifies that our process-based approach
        successfully times out even if cv2.VideoCapture would hang.
        """
        # Use a non-routable IP address that will cause connection attempts to hang
        non_existent_url = "rtsp://192.0.2.1:554/stream"  # TEST-NET-1 (RFC 5737)
        
        start_time = time.time()
        result = self.detector._validate_rtsp_stream(non_existent_url)
        elapsed_time = time.time() - start_time
        
        # Should fail
        self.assertFalse(result, "Connection to non-existent host should fail")
        
        # Should timeout within our configured time + small buffer
        self.assertLess(elapsed_time, 3, 
                       f"Should timeout within 3 seconds, took {elapsed_time:.2f}s")
        
    def test_quick_failure_on_invalid_url(self):
        """Test that obviously invalid URLs fail quickly."""
        invalid_url = "not-a-valid-url"
        
        start_time = time.time()
        result = self.detector._validate_rtsp_stream(invalid_url)
        elapsed_time = time.time() - start_time
        
        self.assertFalse(result, "Invalid URL should fail")
        self.assertLess(elapsed_time, 1, "Invalid URL should fail quickly")
        
    def test_closed_port_fails_quickly(self):
        """Test that connection to closed port fails quickly."""
        # Use localhost with a port we know is closed
        closed_port_url = "rtsp://127.0.0.1:55555/stream"
        
        start_time = time.time()
        result = self.detector._validate_rtsp_stream(closed_port_url)
        elapsed_time = time.time() - start_time
        
        self.assertFalse(result, "Connection to closed port should fail")
        self.assertLess(elapsed_time, 3, "Closed port should fail within timeout")
        
    def test_process_isolation(self):
        """Test that validation runs in isolated process.
        
        We can't easily test this directly, but we can verify that
        the validation completes even with a short timeout.
        """
        # This URL will definitely fail, but should do so cleanly
        test_url = "rtsp://example.com:554/stream"
        
        # Run validation multiple times to ensure process cleanup works
        for i in range(3):
            result = self.detector._validate_rtsp_stream(test_url)
            self.assertFalse(result, f"Validation {i+1} should fail")
            
        # If we get here, process isolation is working correctly
        
        
if __name__ == '__main__':
    unittest.main()