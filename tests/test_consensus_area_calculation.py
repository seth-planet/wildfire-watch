#!/usr/bin/env python3.12
"""Test cases for fire consensus area calculation with different camera resolutions.

This test suite verifies that the consensus service properly handles
area calculations for cameras with different resolutions.
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fire_consensus.consensus import FireConsensus, FireConsensusConfig


class TestConsensusAreaCalculation(unittest.TestCase):
    """Test area calculation for different camera resolutions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a test config
        self.config = FireConsensusConfig()
        
        # Create consensus instance with mocked MQTT connection
        with patch('fire_consensus.consensus.FireConsensus.connect'), \
             patch('fire_consensus.consensus.FireConsensus.setup_mqtt'):
            self.consensus = FireConsensus()
            self.consensus.config = self.config
            self.consensus._mqtt_connected = True  # Mock connected state
    
    def test_area_calculation_with_4k_camera(self):
        """Test that area calculation works correctly for 4K cameras.
        
        The current implementation assumes 1920x1080 resolution, which
        causes incorrect normalization for 4K cameras (3840x2160).
        """
        # 4K camera detection with a fire covering 10% of the frame
        # Box coordinates: top-left (384, 216), bottom-right (768, 432)
        # This is 384x216 pixels = 82,944 pixels
        # As percentage of 4K frame: 82,944 / (3840*2160) = 0.01 (1%)
        
        box = [384, 216, 768, 432]  # Frigate format: [x1, y1, x2, y2]
        
        area = self.consensus._calculate_area(box)
        
        # Current implementation incorrectly assumes 1920x1080
        # So it calculates: 82,944 / (1920*1080) = 0.04 (4%)
        # This is 4x larger than the actual percentage!
        
        # With the hardcoded assumption, this will be ~0.04
        self.assertAlmostEqual(area, 0.04, places=2,
                             msg="Area calculation assumes 1920x1080 resolution")
        
        # The actual area for a 4K camera should be ~0.01
        actual_4k_area = (384 * 216) / (3840 * 2160)
        self.assertAlmostEqual(actual_4k_area, 0.01, places=2,
                             msg="Actual area for 4K camera is much smaller")
        
        # This demonstrates the bug: a fire that covers 1% of a 4K frame
        # is treated as if it covers 4% of the frame, potentially bypassing
        # MIN_AREA_RATIO filters incorrectly
    
    def test_area_calculation_with_low_res_camera(self):
        """Test area calculation for low resolution cameras.
        
        Low resolution cameras (e.g., 640x480) will have their areas
        underestimated by the current implementation.
        """
        # 640x480 camera with fire covering 10% of frame
        # Box: 64x48 pixels = 3,072 pixels
        # Actual percentage: 3,072 / (640*480) = 0.01 (1%)
        
        box = [100, 100, 164, 148]  # 64x48 pixel box
        
        area = self.consensus._calculate_area(box)
        
        # Current implementation: 3,072 / (1920*1080) = 0.0015
        # This underestimates the area by ~7x!
        
        self.assertLess(area, 0.002,
                       msg="Low-res camera areas are underestimated")
        
        # Actual area for 640x480 camera
        actual_vga_area = (64 * 48) / (640 * 480)
        self.assertAlmostEqual(actual_vga_area, 0.01, places=2,
                             msg="Actual area for VGA camera is much larger")
    
    def test_normalized_area_calculation(self):
        """Test that normalized coordinates work correctly.
        
        Normalized coordinates (0-1) should not be affected by resolution.
        """
        # Normalized box covering 10% of frame
        normalized_box = [0.1, 0.1, 0.2, 0.2]  # 0.1 x 0.1 = 0.01 (1%)
        
        area = self.consensus._calculate_area(normalized_box)
        
        self.assertAlmostEqual(area, 0.01, places=3,
                             msg="Normalized coordinates should work correctly")
    
    def test_different_resolutions_same_fire(self):
        """Test that the same fire appears different on different resolution cameras.
        
        This demonstrates how the hardcoded resolution affects consensus.
        """
        # Same physical fire seen by three cameras
        fire_width_meters = 2.0  # 2 meter wide fire
        fire_height_meters = 1.5  # 1.5 meter tall fire
        distance_meters = 50.0  # 50 meters away
        
        # Approximate pixels based on typical camera FOV
        # These are simplified calculations for demonstration
        
        # Test 1: When camera resolution is provided, areas should be properly normalized
        # 1080p camera (1920x1080)
        fire_pixels_1080p = (192, 162)  # ~10% of width, ~15% of height
        box_1080p = [864, 459, 1056, 621]  # Center of frame
        area_1080p_with_res = self.consensus._calculate_area(box_1080p, camera_resolution=(1920, 1080))
        
        # 4K camera (3840x2160) - same fire appears as more pixels
        fire_pixels_4k = (384, 324)  # Same percentage of frame
        box_4k = [1728, 918, 2112, 1242]
        area_4k_with_res = self.consensus._calculate_area(box_4k, camera_resolution=(3840, 2160))
        
        # 720p camera (1280x720) - same fire appears as fewer pixels  
        fire_pixels_720p = (128, 108)
        box_720p = [576, 306, 704, 414]
        area_720p_with_res = self.consensus._calculate_area(box_720p, camera_resolution=(1280, 720))
        
        # When camera resolution is provided, all cameras should report similar areas
        # (same physical fire, same percentage of frame)
        self.assertAlmostEqual(area_1080p_with_res, 0.015, places=2)
        self.assertAlmostEqual(area_4k_with_res, 0.015, places=2)
        self.assertAlmostEqual(area_720p_with_res, 0.015, places=2)
        
        # Test 2: Without camera resolution, the hardcoded 1920x1080 assumption causes issues
        area_1080p_no_res = self.consensus._calculate_area(box_1080p)  # No resolution provided
        area_4k_no_res = self.consensus._calculate_area(box_4k)  # No resolution provided
        area_720p_no_res = self.consensus._calculate_area(box_720p)  # No resolution provided
        
        # Only the 1080p camera gets the "correct" area
        self.assertAlmostEqual(area_1080p_no_res, 0.015, places=2)
        
        # 4K camera's area is overestimated (appears 4x larger due to 2x pixel dimensions)
        self.assertGreater(area_4k_no_res, area_1080p_no_res * 3.5)
        
        # 720p camera's area is underestimated (appears smaller)
        self.assertLess(area_720p_no_res, area_1080p_no_res * 0.5)
        
        print(f"With resolution - Same fire calculated areas: 1080p={area_1080p_with_res:.3f}, "
              f"4K={area_4k_with_res:.3f}, 720p={area_720p_with_res:.3f}")
        print(f"Without resolution - Same fire calculated areas: 1080p={area_1080p_no_res:.3f}, "
              f"4K={area_4k_no_res:.3f}, 720p={area_720p_no_res:.3f}")


if __name__ == '__main__':
    unittest.main()