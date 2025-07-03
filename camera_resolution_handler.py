#!/usr/bin/env python3.12
"""Camera Resolution Handler for Dynamic Resolution Support

This module provides resolution-independent camera handling with proper
aspect ratio preservation and model input size adaptation.
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class Resolution:
    """Represents a resolution with aspect ratio info"""
    width: int
    height: int
    
    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio (width/height)"""
        return self.width / self.height
    
    @property
    def total_pixels(self) -> int:
        """Total pixel count"""
        return self.width * self.height
    
    def __str__(self):
        return f"{self.width}x{self.height}"

class ResolutionHandler:
    """Handles resolution adaptation between cameras and models"""
    
    # Common camera resolutions in priority order
    COMMON_RESOLUTIONS = [
        Resolution(3840, 2160),  # 4K
        Resolution(2560, 1440),  # 1440p
        Resolution(1920, 1080),  # 1080p
        Resolution(1280, 720),   # 720p
        Resolution(960, 540),    # 540p
        Resolution(640, 480),    # VGA
        Resolution(640, 360),    # 360p
    ]
    
    # Common model input sizes
    COMMON_MODEL_SIZES = [
        Resolution(640, 640),    # YOLOv5/v8 default
        Resolution(416, 416),    # YOLOv3/v4 default
        Resolution(320, 320),    # Mobile/edge optimized
        Resolution(512, 512),    # Some detection models
        Resolution(1024, 1024),  # High-res models
    ]
    
    def __init__(self, 
                 model_size: Tuple[int, int] = (640, 640),
                 maintain_aspect_ratio: bool = True,
                 interpolation: int = cv2.INTER_LINEAR):
        """Initialize resolution handler
        
        Args:
            model_size: Target model input size (width, height)
            maintain_aspect_ratio: Whether to preserve aspect ratio with letterboxing
            interpolation: OpenCV interpolation method
        """
        self.model_size = Resolution(model_size[0], model_size[1])
        self.maintain_aspect_ratio = maintain_aspect_ratio
        self.interpolation = interpolation
    
    def detect_camera_resolutions(self, camera_source: Any) -> List[Resolution]:
        """Detect supported resolutions for a camera
        
        Args:
            camera_source: Camera index, RTSP URL, or file path
            
        Returns:
            List of supported resolutions, highest first
        """
        supported = []
        
        # Try to open camera
        cap = cv2.VideoCapture(camera_source)
        if not cap.isOpened():
            logger.error(f"Failed to open camera: {camera_source}")
            return supported
        
        # Save original resolution
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Test common resolutions
        for res in self.COMMON_RESOLUTIONS:
            # Try to set resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, res.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res.height)
            
            # Check if it was actually set
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if actual_width == res.width and actual_height == res.height:
                supported.append(res)
                logger.debug(f"Camera supports {res}")
        
        # Restore original resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, original_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, original_height)
        
        cap.release()
        
        # If no standard resolutions worked, add the original
        if not supported:
            supported.append(Resolution(original_width, original_height))
        
        return supported
    
    def select_optimal_resolution(self, 
                                camera_source: Any,
                                max_pixels: Optional[int] = None,
                                prefer_aspect_ratio: Optional[float] = None) -> Optional[Resolution]:
        """Select optimal camera resolution based on constraints
        
        Args:
            camera_source: Camera source
            max_pixels: Maximum total pixels (for performance)
            prefer_aspect_ratio: Preferred aspect ratio (e.g., 16/9)
            
        Returns:
            Optimal resolution or None if camera unavailable
        """
        resolutions = self.detect_camera_resolutions(camera_source)
        if not resolutions:
            return None
        
        # Filter by max pixels if specified
        if max_pixels:
            resolutions = [r for r in resolutions if r.total_pixels <= max_pixels]
            if not resolutions:
                # Take smallest available if all exceed max
                resolutions = [min(self.detect_camera_resolutions(camera_source), 
                                 key=lambda r: r.total_pixels)]
        
        # Sort by aspect ratio similarity if preference given
        if prefer_aspect_ratio:
            resolutions.sort(key=lambda r: abs(r.aspect_ratio - prefer_aspect_ratio))
        
        # Return highest resolution (first in list)
        return resolutions[0]
    
    def calculate_letterbox_params(self, 
                                 src_size: Tuple[int, int],
                                 dst_size: Tuple[int, int]) -> Dict[str, Any]:
        """Calculate letterboxing parameters to maintain aspect ratio
        
        Args:
            src_size: Source image size (width, height)
            dst_size: Destination size (width, height)
            
        Returns:
            Dictionary with resize and padding parameters
        """
        src_w, src_h = src_size
        dst_w, dst_h = dst_size
        
        # Calculate scale to fit
        scale = min(dst_w / src_w, dst_h / src_h)
        
        # New size after scaling
        new_w = int(src_w * scale)
        new_h = int(src_h * scale)
        
        # Calculate padding
        pad_w = dst_w - new_w
        pad_h = dst_h - new_h
        
        # Split padding evenly
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        
        return {
            'scale': scale,
            'new_size': (new_w, new_h),
            'padding': {
                'left': pad_left,
                'right': pad_right,
                'top': pad_top,
                'bottom': pad_bottom
            }
        }
    
    def preprocess_image(self, 
                        image: np.ndarray,
                        normalize: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Preprocess image for model input
        
        Args:
            image: Input image (BGR format from OpenCV)
            normalize: Whether to normalize pixel values to [0,1]
            
        Returns:
            Preprocessed image and transformation info for coordinate mapping
        """
        orig_h, orig_w = image.shape[:2]
        
        if self.maintain_aspect_ratio:
            # Calculate letterbox parameters
            params = self.calculate_letterbox_params(
                (orig_w, orig_h),
                (self.model_size.width, self.model_size.height)
            )
            
            # Resize image
            resized = cv2.resize(
                image, 
                params['new_size'], 
                interpolation=self.interpolation
            )
            
            # Add padding
            padded = cv2.copyMakeBorder(
                resized,
                params['padding']['top'],
                params['padding']['bottom'],
                params['padding']['left'],
                params['padding']['right'],
                cv2.BORDER_CONSTANT,
                value=(114, 114, 114)  # Gray padding
            )
            
            transform_info = {
                'original_size': (orig_w, orig_h),
                'scale': params['scale'],
                'padding': params['padding'],
                'letterboxed': True
            }
        else:
            # Simple resize without maintaining aspect ratio
            padded = cv2.resize(
                image,
                (self.model_size.width, self.model_size.height),
                interpolation=self.interpolation
            )
            
            transform_info = {
                'original_size': (orig_w, orig_h),
                'scale_x': self.model_size.width / orig_w,
                'scale_y': self.model_size.height / orig_h,
                'letterboxed': False
            }
        
        # Normalize if requested
        if normalize:
            padded = padded.astype(np.float32) / 255.0
        
        return padded, transform_info
    
    def postprocess_bbox(self,
                        bbox: Tuple[float, float, float, float],
                        transform_info: Dict[str, Any]) -> Tuple[float, float, float, float]:
        """Convert model output bbox back to original image coordinates
        
        Args:
            bbox: Bounding box in model coordinates (x1, y1, x2, y2)
            transform_info: Transformation info from preprocessing
            
        Returns:
            Bounding box in original image coordinates
        """
        x1, y1, x2, y2 = bbox
        
        if transform_info['letterboxed']:
            # Remove padding offset
            x1 -= transform_info['padding']['left']
            x2 -= transform_info['padding']['left']
            y1 -= transform_info['padding']['top']
            y2 -= transform_info['padding']['top']
            
            # Reverse scaling
            scale = transform_info['scale']
            x1 /= scale
            x2 /= scale
            y1 /= scale
            y2 /= scale
        else:
            # Simple scaling reversal
            x1 *= transform_info['original_size'][0] / self.model_size.width
            x2 *= transform_info['original_size'][0] / self.model_size.width
            y1 *= transform_info['original_size'][1] / self.model_size.height
            y2 *= transform_info['original_size'][1] / self.model_size.height
        
        # Ensure coordinates are within image bounds
        orig_w, orig_h = transform_info['original_size']
        x1 = max(0, min(x1, orig_w))
        x2 = max(0, min(x2, orig_w))
        y1 = max(0, min(y1, orig_h))
        y2 = max(0, min(y2, orig_h))
        
        return (x1, y1, x2, y2)
    
    def calculate_detection_area(self,
                               bbox: Tuple[float, float, float, float],
                               image_size: Tuple[int, int]) -> float:
        """Calculate detection area as fraction of image
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            image_size: Image size (width, height)
            
        Returns:
            Area as fraction of total image (0.0 to 1.0)
        """
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        image_area = image_size[0] * image_size[1]
        
        return bbox_area / image_area if image_area > 0 else 0.0

class CameraConfigManager:
    """Manages camera configurations with resolution flexibility"""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize camera configuration manager
        
        Args:
            config_file: Path to camera configuration file
        """
        self.config_file = Path(config_file) if config_file else None
        self.cameras = {}
        
        if self.config_file and self.config_file.exists():
            self._load_config()
    
    def _load_config(self):
        """Load camera configurations from file"""
        import yaml
        
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
                
            for camera_id, camera_config in config.get('cameras', {}).items():
                self.cameras[camera_id] = {
                    'source': camera_config.get('source'),
                    'preferred_resolution': Resolution(
                        camera_config.get('width', 1920),
                        camera_config.get('height', 1080)
                    ),
                    'max_fps': camera_config.get('max_fps', 30),
                    'enabled': camera_config.get('enabled', True)
                }
                
        except Exception as e:
            logger.error(f"Failed to load camera config: {e}")
    
    def add_camera(self, 
                  camera_id: str,
                  source: Any,
                  preferred_resolution: Optional[Resolution] = None):
        """Add or update camera configuration
        
        Args:
            camera_id: Unique camera identifier
            source: Camera source (index, URL, path)
            preferred_resolution: Preferred resolution
        """
        self.cameras[camera_id] = {
            'source': source,
            'preferred_resolution': preferred_resolution or Resolution(1920, 1080),
            'enabled': True
        }
    
    def get_camera_config(self, camera_id: str) -> Optional[Dict]:
        """Get camera configuration
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            Camera configuration or None
        """
        return self.cameras.get(camera_id)
    
    def save_config(self):
        """Save camera configurations to file"""
        if not self.config_file:
            logger.warning("No config file specified")
            return
        
        import yaml
        
        config = {'cameras': {}}
        
        for camera_id, camera_info in self.cameras.items():
            config['cameras'][camera_id] = {
                'source': camera_info['source'],
                'width': camera_info['preferred_resolution'].width,
                'height': camera_info['preferred_resolution'].height,
                'max_fps': camera_info.get('max_fps', 30),
                'enabled': camera_info.get('enabled', True)
            }
        
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
                
            logger.info(f"Saved camera config to {self.config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save camera config: {e}")

def demo_resolution_handling():
    """Demonstrate resolution handling capabilities"""
    # Create handler for 640x640 model
    handler = ResolutionHandler(model_size=(640, 640))
    
    # Simulate different camera resolutions
    test_resolutions = [
        (1920, 1080),  # 16:9 HD
        (3840, 2160),  # 16:9 4K
        (1280, 960),   # 4:3
        (640, 480),    # 4:3 VGA
        (2560, 1440),  # 16:9 1440p
    ]
    
    print("Resolution Adaptation Demo")
    print("=" * 50)
    print(f"Model Input Size: {handler.model_size}")
    print()
    
    for width, height in test_resolutions:
        print(f"\nCamera Resolution: {width}x{height}")
        
        # Calculate letterbox params
        params = handler.calculate_letterbox_params(
            (width, height),
            (handler.model_size.width, handler.model_size.height)
        )
        
        print(f"  Scale Factor: {params['scale']:.3f}")
        print(f"  Resized Size: {params['new_size']}")
        print(f"  Padding: Top={params['padding']['top']}, "
              f"Bottom={params['padding']['bottom']}, "
              f"Left={params['padding']['left']}, "
              f"Right={params['padding']['right']}")
        
        # Calculate area preservation
        orig_area = width * height
        scaled_area = params['new_size'][0] * params['new_size'][1]
        area_ratio = scaled_area / orig_area
        print(f"  Area Preservation: {area_ratio:.1%}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_resolution_handling()