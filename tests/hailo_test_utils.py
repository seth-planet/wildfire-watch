#!/usr/bin/env python3.10
"""Hailo testing utilities for Wildfire Watch.

This module provides common utilities for testing Hailo integration including:
- Video download and caching
- RTSP stream management
- MQTT test client helpers
- Performance metric collection
- Hailo device management
"""

import os
import sys
import time
import json
import hashlib
import subprocess
import tempfile
import threading
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from urllib.request import urlretrieve
import cv2
import numpy as np
import paho.mqtt.client as mqtt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test video URLs
TEST_VIDEO_URLS = {
    "fire1.mov": "https://github.com/AlimTleuliyev/wildfire-detection/raw/main/demo-videos/fire1.mov",
    "fire2.mov": "https://github.com/AlimTleuliyev/wildfire-detection/raw/main/demo-videos/fire2.mov", 
    "fire3.mp4": "https://github.com/AlimTleuliyev/wildfire-detection/raw/main/demo-videos/fire3.mp4",
    "fire4.mp4": "https://github.com/AlimTleuliyev/wildfire-detection/raw/main/demo-videos/fire4.mp4"
}

# Expected checksums for validation
VIDEO_CHECKSUMS = {
    "fire1.mov": None,  # Will be computed on first download
    "fire2.mov": None,
    "fire3.mp4": None,
    "fire4.mp4": None
}


@dataclass
class PerformanceMetrics:
    """Container for performance measurement results."""
    inference_time_ms: float
    preprocessing_time_ms: float
    postprocessing_time_ms: float
    fps: float
    batch_size: int
    memory_usage_mb: float
    cpu_percent: float
    temperature_celsius: Optional[float] = None
    power_watts: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class VideoDownloader:
    """Downloads and caches test videos for Hailo testing."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize video downloader.
        
        Args:
            cache_dir: Directory to cache videos. Uses temp dir if None.
        """
        if cache_dir is None:
            cache_dir = Path(tempfile.gettempdir()) / "wildfire_test_videos"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def download_video(self, video_name: str, force: bool = False) -> Path:
        """Download a test video if not cached.
        
        Args:
            video_name: Name of video file (e.g., "fire1.mov")
            force: Force re-download even if cached
            
        Returns:
            Path to downloaded video file
            
        Raises:
            ValueError: If video name not recognized
            RuntimeError: If download fails
        """
        if video_name not in TEST_VIDEO_URLS:
            raise ValueError(f"Unknown video: {video_name}. Available: {list(TEST_VIDEO_URLS.keys())}")
            
        video_path = self.cache_dir / video_name
        
        # Check if already cached
        if video_path.exists() and not force:
            logger.info(f"Using cached video: {video_path}")
            return video_path
            
        # Download video
        url = TEST_VIDEO_URLS[video_name]
        logger.info(f"Downloading {video_name} from {url}...")
        
        try:
            urlretrieve(url, video_path)
            logger.info(f"Downloaded {video_name} to {video_path}")
            
            # Compute and store checksum
            checksum = self._compute_checksum(video_path)
            logger.info(f"Video checksum: {checksum}")
            
            return video_path
            
        except Exception as e:
            if video_path.exists():
                video_path.unlink()
            raise RuntimeError(f"Failed to download {video_name}: {e}")
            
    def download_all_videos(self) -> Dict[str, Path]:
        """Download all test videos.
        
        Returns:
            Dictionary mapping video names to paths
        """
        videos = {}
        for video_name in TEST_VIDEO_URLS:
            try:
                path = self.download_video(video_name)
                videos[video_name] = path
            except Exception as e:
                logger.error(f"Failed to download {video_name}: {e}")
                
        return videos
        
    def _compute_checksum(self, file_path: Path) -> str:
        """Compute SHA256 checksum of file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


class RTSPStreamServer:
    """Simple RTSP server for streaming test videos."""
    
    def __init__(self, port: int = 8554):
        """Initialize RTSP server.
        
        Args:
            port: RTSP server port
        """
        self.port = port
        self.process = None
        self.container_name = f"rtsp_server_test_{port}"
        
    def start(self, video_path: Path, stream_name: str = "stream") -> str:
        """Start RTSP server streaming a video.
        
        Args:
            video_path: Path to video file to stream
            stream_name: Name of RTSP stream
            
        Returns:
            RTSP URL for the stream
        """
        if self.process is not None:
            self.stop()
            
        # Use mediamtx (formerly rtsp-simple-server) Docker image
        cmd = [
            "docker", "run", "-d",
            "--name", self.container_name,
            "-p", f"{self.port}:8554",
            "-v", f"{video_path.parent}:/videos:ro",
            "-e", f"RTSP_PATHS_{stream_name.upper()}_SOURCE=file:///videos/{video_path.name}",
            "bluenviron/mediamtx:latest"
        ]
        
        logger.info(f"Starting RTSP server: {' '.join(cmd)}")
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        time.sleep(3)
        
        rtsp_url = f"rtsp://localhost:{self.port}/{stream_name}"
        logger.info(f"RTSP stream available at: {rtsp_url}")
        
        return rtsp_url
        
    def stop(self):
        """Stop RTSP server."""
        if self.process:
            subprocess.run(["docker", "stop", self.container_name], capture_output=True)
            subprocess.run(["docker", "rm", self.container_name], capture_output=True)
            self.process = None
            logger.info("RTSP server stopped")


class MQTTTestClient:
    """MQTT client for testing Frigate integration."""
    
    def __init__(self, host: str = "localhost", port: int = 1883):
        """Initialize MQTT test client.
        
        Args:
            host: MQTT broker host
            port: MQTT broker port
        """
        self.host = host
        self.port = port
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "hailo_test_client")
        self.messages = []
        self.connected = False
        
        # Set up callbacks
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        
    def connect(self, timeout: int = 10) -> bool:
        """Connect to MQTT broker.
        
        Args:
            timeout: Connection timeout in seconds
            
        Returns:
            True if connected successfully
        """
        try:
            self.client.connect(self.host, self.port)
            self.client.loop_start()
            
            # Wait for connection
            start_time = time.time()
            while not self.connected and time.time() - start_time < timeout:
                time.sleep(0.1)
                
            return self.connected
            
        except Exception as e:
            logger.error(f"Failed to connect to MQTT: {e}")
            return False
            
    def subscribe(self, topics: List[str]):
        """Subscribe to MQTT topics.
        
        Args:
            topics: List of topics to subscribe to
        """
        for topic in topics:
            self.client.subscribe(topic)
            logger.info(f"Subscribed to: {topic}")
            
    def wait_for_message(self, topic_filter: str, timeout: int = 30) -> Optional[Dict]:
        """Wait for a message matching topic filter.
        
        Args:
            topic_filter: Topic to match (can include wildcards)
            timeout: Maximum wait time in seconds
            
        Returns:
            Message payload as dict or None if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            for msg in self.messages:
                if self._topic_matches(topic_filter, msg['topic']):
                    return msg
            time.sleep(0.1)
            
        return None
        
    def get_messages(self, topic_filter: Optional[str] = None) -> List[Dict]:
        """Get all received messages.
        
        Args:
            topic_filter: Optional topic filter
            
        Returns:
            List of message dictionaries
        """
        if topic_filter is None:
            return self.messages.copy()
            
        return [msg for msg in self.messages if self._topic_matches(topic_filter, msg['topic'])]
        
    def clear_messages(self):
        """Clear message buffer."""
        self.messages.clear()
        
    def disconnect(self):
        """Disconnect from MQTT broker."""
        self.client.loop_stop()
        self.client.disconnect()
        self.connected = False
        
    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """MQTT connection callback."""
        if rc == 0:
            self.connected = True
            logger.info("Connected to MQTT broker")
        else:
            logger.error(f"MQTT connection failed with code: {rc}")
            
    def _on_message(self, client, userdata, msg):
        """MQTT message callback."""
        try:
            payload = json.loads(msg.payload.decode())
        except:
            payload = msg.payload.decode()
            
        message = {
            'topic': msg.topic,
            'payload': payload,
            'timestamp': time.time()
        }
        
        self.messages.append(message)
        logger.debug(f"Received: {msg.topic} = {payload}")
        
    def _topic_matches(self, filter_topic: str, actual_topic: str) -> bool:
        """Check if topic matches filter (supports wildcards)."""
        filter_parts = filter_topic.split('/')
        actual_parts = actual_topic.split('/')
        
        if len(filter_parts) != len(actual_parts):
            return False
            
        for f, a in zip(filter_parts, actual_parts):
            if f == '+':  # Single level wildcard
                continue
            elif f == '#':  # Multi-level wildcard
                return True
            elif f != a:
                return False
                
        return True


class HailoDevice:
    """Hailo device management for testing."""
    
    def __init__(self, device_id: int = 0):
        """Initialize Hailo device wrapper.
        
        Args:
            device_id: Hailo device ID (usually 0)
        """
        self.device_id = device_id
        self.device_path = f"/dev/hailo{device_id}"
        
    def is_available(self) -> bool:
        """Check if Hailo device is available."""
        return Path(self.device_path).exists()
        
    def get_temperature(self) -> Optional[float]:
        """Get Hailo device temperature.
        
        Returns:
            Temperature in Celsius or None if unavailable
        """
        # Try to read from sysfs
        temp_paths = [
            f"/sys/class/hailo_chardev/hailo{self.device_id}/device/hwmon/hwmon*/temp1_input",
            f"/sys/devices/virtual/hailo_chardev/hailo{self.device_id}/temperature"
        ]
        
        for temp_path_pattern in temp_paths:
            import glob
            temp_files = glob.glob(temp_path_pattern)
            if temp_files:
                try:
                    with open(temp_files[0], 'r') as f:
                        # Temperature is usually in millidegrees
                        temp_milli = int(f.read().strip())
                        return temp_milli / 1000.0
                except Exception as e:
                    logger.debug(f"Failed to read temperature from {temp_files[0]}: {e}")
                    
        return None
        
    def get_power_consumption(self) -> Optional[float]:
        """Get Hailo device power consumption.
        
        Returns:
            Power in watts or None if unavailable
        """
        # This would require specific power monitoring hardware
        # For now, return None
        return None
        
    def reset(self):
        """Reset Hailo device (requires sudo)."""
        try:
            subprocess.run(["sudo", "hailortcli", "reset"], check=True)
            logger.info("Hailo device reset")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to reset Hailo device: {e}")


def collect_system_metrics() -> Dict[str, float]:
    """Collect system performance metrics.
    
    Returns:
        Dictionary of metric name to value
    """
    import psutil
    
    metrics = {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_used_mb': psutil.virtual_memory().used / (1024 * 1024),
        'memory_percent': psutil.virtual_memory().percent
    }
    
    # Try to get GPU metrics if available
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            gpu_util, gpu_mem = result.stdout.strip().split(', ')
            metrics['gpu_percent'] = float(gpu_util)
            metrics['gpu_memory_mb'] = float(gpu_mem)
    except:
        pass
        
    return metrics


def measure_inference_time(func):
    """Decorator to measure inference time.
    
    Usage:
        @measure_inference_time
        def run_inference(model, input_data):
            return model.infer(input_data)
    """
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        inference_time_ms = (end_time - start_time) * 1000
        
        # Store in result if it's a dict
        if isinstance(result, dict):
            result['inference_time_ms'] = inference_time_ms
        
        return result
        
    return wrapper