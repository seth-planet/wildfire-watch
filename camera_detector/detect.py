#!/usr/bin/env python3.12
"""
Camera Discovery and Management Service - Wildfire Watch
Discovers IP cameras via ONVIF/mDNS, tracks by MAC address, integrates with Frigate
"""
import os
import sys
import time
import json
import yaml
import socket
import threading
import logging
import subprocess
import asyncio
import ipaddress
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import paho.mqtt.client as mqtt
from dotenv import load_dotenv
from onvif import ONVIFCamera
from wsdiscovery.discovery import ThreadedWSDiscovery as WSDiscovery
import netifaces
import cv2
from scapy.all import ARP, Ether, srp

load_dotenv()

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
class Config:
    # MQTT Settings
    MQTT_BROKER = os.getenv("MQTT_BROKER", "mqtt_broker")
    MQTT_PORT = max(1, min(65535, int(os.getenv("MQTT_PORT", "1883"))))
    MQTT_TLS = os.getenv("MQTT_TLS", "false").lower() == "true"
    TLS_CA_PATH = os.getenv("TLS_CA_PATH", "/mnt/data/certs/ca.crt")
    
    # Discovery Settings  
    DISCOVERY_INTERVAL = max(30, int(os.getenv("DISCOVERY_INTERVAL", "300")))  # Minimum 30 seconds
    RTSP_TIMEOUT = max(1, min(60, int(os.getenv("RTSP_TIMEOUT", "10"))))  # 1-60 seconds
    ONVIF_TIMEOUT = max(1, min(30, int(os.getenv("ONVIF_TIMEOUT", "5"))))  # 1-30 seconds
    MAC_TRACKING_ENABLED = os.getenv("MAC_TRACKING_ENABLED", "true").lower() == "true"
    
    # Smart Discovery Settings
    SMART_DISCOVERY_ENABLED = os.getenv("SMART_DISCOVERY_ENABLED", "true").lower() == "true"
    INITIAL_DISCOVERY_COUNT = int(os.getenv("INITIAL_DISCOVERY_COUNT", "3"))  # Aggressive scans at startup
    STEADY_STATE_INTERVAL = max(300, int(os.getenv("STEADY_STATE_INTERVAL", "1800")))  # 30 min in steady state
    QUICK_CHECK_INTERVAL = max(30, int(os.getenv("QUICK_CHECK_INTERVAL", "60")))  # Quick health checks
    
    # Camera Settings
    DEFAULT_USERNAME = os.getenv("CAMERA_USERNAME", "admin")
    DEFAULT_PASSWORD = os.getenv("CAMERA_PASSWORD", "")
    RTSP_PORT = int(os.getenv("RTSP_PORT", "554"))
    ONVIF_PORT = int(os.getenv("ONVIF_PORT", "80"))
    HTTP_PORT = int(os.getenv("HTTP_PORT", "80"))
    
    # Camera credentials (comma-separated user:pass pairs)
    CAMERA_CREDENTIALS = os.getenv("CAMERA_CREDENTIALS", "admin:,admin:admin,admin:12345")
    
    # Health Monitoring
    HEALTH_CHECK_INTERVAL = max(10, int(os.getenv("HEALTH_CHECK_INTERVAL", "60")))  # Minimum 10 seconds
    OFFLINE_THRESHOLD = max(60, int(os.getenv("OFFLINE_THRESHOLD", "180")))  # Minimum 1 minute
    
    # Frigate Integration
    FRIGATE_CONFIG_PATH = os.getenv("FRIGATE_CONFIG_PATH", "/config/frigate/cameras.yml")
    FRIGATE_UPDATE_ENABLED = os.getenv("FRIGATE_UPDATE_ENABLED", "true").lower() == "true"
    FRIGATE_RELOAD_TOPIC = os.getenv("FRIGATE_RELOAD_TOPIC", "frigate/config/reload")
    
    # Node Identity
    NODE_ID = os.getenv("BALENA_DEVICE_UUID", socket.gethostname())
    SERVICE_ID = f"camera-detector-{NODE_ID}"
    
    # Topics
    TOPIC_DISCOVERY = "camera/discovery"
    TOPIC_STATUS = "camera/status"
    TOPIC_HEALTH = "system/camera_detector_health"
    TOPIC_FRIGATE_CONFIG = "frigate/config/cameras"

# ─────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────
log_level = getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────
@dataclass
class CameraProfile:
    """Camera stream profile"""
    name: str
    token: str
    resolution: Optional[Tuple[int, int]] = None
    framerate: Optional[int] = None
    encoding: Optional[str] = None

@dataclass
class Camera:
    """Represents a discovered camera"""
    ip: str
    mac: str
    name: str
    manufacturer: str = "Unknown"
    model: str = "Unknown"
    serial_number: str = "Unknown"
    firmware_version: str = "Unknown"
    onvif_url: Optional[str] = None
    rtsp_urls: Dict[str, str] = field(default_factory=dict)  # profile -> url
    http_url: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    last_seen: float = 0
    last_validated: float = 0
    online: bool = False
    stream_active: bool = False
    profiles: List[CameraProfile] = field(default_factory=list)
    capabilities: Dict[str, bool] = field(default_factory=dict)
    ip_history: List[str] = field(default_factory=list)  # Track IP changes
    
    def __post_init__(self):
        self.last_seen = time.time()
        if self.ip and self.ip not in self.ip_history:
            self.ip_history.append(self.ip)
    
    @property
    def id(self) -> str:
        """Unique ID based on MAC address"""
        return self.mac.replace(":", "").lower()
    
    @property
    def primary_rtsp_url(self) -> Optional[str]:
        """Get primary RTSP URL"""
        if self.rtsp_urls:
            # Prefer main/high quality stream
            for key in ['main', 'mainstream', 'high', 'profile_1']:
                if key in self.rtsp_urls:
                    return self.rtsp_urls[key]
            # Return first available
            return list(self.rtsp_urls.values())[0]
        return None
    
    def update_ip(self, new_ip: str):
        """Update IP address and track history"""
        if new_ip != self.ip:
            logger.info(f"Camera {self.name} IP changed from {self.ip} to {new_ip}")
            self.ip = new_ip
            if new_ip not in self.ip_history:
                self.ip_history.append(new_ip)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            **asdict(self),
            'id': self.id,
            'primary_rtsp_url': self.primary_rtsp_url,
            'last_seen_iso': datetime.fromtimestamp(self.last_seen).isoformat() + 'Z',
            'last_validated_iso': datetime.fromtimestamp(self.last_validated).isoformat() + 'Z' if self.last_validated else None
        }
    
    def to_frigate_config(self) -> dict:
        """Generate Frigate camera configuration"""
        if not self.primary_rtsp_url:
            return None
        
        # Determine stream quality based on available profiles
        detect_width = 1280
        detect_height = 720
        
        for profile in self.profiles:
            if profile.resolution:
                # Use lower resolution for detection if available
                if profile.resolution[0] <= 1280:
                    detect_width = profile.resolution[0]
                    detect_height = profile.resolution[1]
                    break
        
        config = {
            self.id: {
                'ffmpeg': {
                    'inputs': []
                },
                'detect': {
                    'enabled': True,
                    'width': detect_width,
                    'height': detect_height,
                    'fps': 5,
                    'stationary': {
                        'interval': 0,
                        'threshold': 50
                    }
                },
                'objects': {
                    'track': ['fire', 'smoke', 'person'],
                    'filters': {
                        'fire': {
                            'min_area': 500,
                            'max_area': 100000,
                            'threshold': 0.7
                        },
                        'smoke': {
                            'min_area': 1000,
                            'max_area': 200000,
                            'threshold': 0.6
                        }
                    }
                },
                'record': {
                    'enabled': True,
                    'retain': {
                        'days': 7,
                        'mode': 'active_objects'
                    },
                    'events': {
                        'retain': {
                            'default': 14,
                            'mode': 'active_objects',
                            'objects': {
                                'fire': 30,
                                'smoke': 30
                            }
                        }
                    }
                },
                'snapshots': {
                    'enabled': True,
                    'timestamp': True,
                    'retain': {
                        'default': 7,
                        'objects': {
                            'fire': 30,
                            'smoke': 30
                        }
                    }
                },
                'mqtt': {
                    'enabled': True,
                    'timestamp': True,
                    'crop': True,
                    'quality': 90
                }
            }
        }
        
        # Add main stream for recording and detection
        if self.primary_rtsp_url:
            config[self.id]['ffmpeg']['inputs'].append({
                'path': self.primary_rtsp_url,
                'roles': ['detect', 'record', 'clips']
            })
        
        # Add substream if available for detection only
        for key in ['sub', 'substream', 'low', 'profile_2']:
            if key in self.rtsp_urls and self.rtsp_urls[key] != self.primary_rtsp_url:
                config[self.id]['ffmpeg']['inputs'].append({
                    'path': self.rtsp_urls[key],
                    'roles': ['detect']
                })
                break
        
        # Add camera info
        config[self.id]['ui'] = {
            'order': 0,
            'dashboard': True
        }
        
        config[self.id]['description'] = f"{self.manufacturer} {self.model} ({self.mac})"
        
        return config

# ─────────────────────────────────────────────────────────────
# MAC Address Tracker
# ─────────────────────────────────────────────────────────────
class MACTracker:
    """Tracks MAC addresses and their IP associations"""
    
    def __init__(self):
        self.mac_to_ip: Dict[str, str] = {}
        self.ip_to_mac: Dict[str, str] = {}
        self.lock = threading.Lock()
    
    def update(self, mac: str, ip: str):
        """Update MAC-IP mapping"""
        with self.lock:
            old_ip = self.mac_to_ip.get(mac)
            if old_ip and old_ip != ip:
                # IP changed for this MAC
                self.ip_to_mac.pop(old_ip, None)
            
            self.mac_to_ip[mac] = ip
            self.ip_to_mac[ip] = mac
    
    def get_mac_for_ip(self, ip: str) -> Optional[str]:
        """Get MAC address for IP"""
        with self.lock:
            return self.ip_to_mac.get(ip)
    
    def get_ip_for_mac(self, mac: str) -> Optional[str]:
        """Get current IP for MAC"""
        with self.lock:
            return self.mac_to_ip.get(mac)
    
    def scan_network(self, network: str) -> Dict[str, str]:
        """Scan network for MAC addresses using ARP"""
        try:
            # Check if we have permission for raw sockets
            import os
            if os.geteuid() != 0:
                logger.debug("ARP scan requires root privileges, skipping")
                return {}
            
            # Create ARP request
            arp_request = ARP(pdst=network)
            broadcast = Ether(dst="ff:ff:ff:ff:ff:ff")
            arp_request_broadcast = broadcast / arp_request
            
            # Send request and receive response
            answered_list = srp(arp_request_broadcast, timeout=2, verbose=False)[0]
            
            # Parse results
            results = {}
            for element in answered_list:
                ip = element[1].psrc
                mac = element[1].hwsrc.upper()
                results[ip] = mac
                self.update(mac, ip)
            
            return results
            
        except Exception as e:
            logger.debug(f"ARP scan not available: {e}")
            return {}

# ─────────────────────────────────────────────────────────────
# Camera Discovery and Management
# ─────────────────────────────────────────────────────────────
class CameraDetector:
    def __init__(self):
        self.config = Config()
        self.cameras: Dict[str, Camera] = {}  # MAC -> Camera
        self.mac_tracker = MACTracker()
        self.lock = threading.RLock()
        
        # Parse credentials
        self.credentials = self._parse_credentials()
        
        # Smart discovery state
        self.discovery_count = 0
        self.last_camera_count = 0
        self.stable_count = 0
        self.is_steady_state = False
        self.last_full_discovery = 0
        self.known_camera_ips: Set[str] = set()
        
        # MQTT client
        self.mqtt_client = None
        self.mqtt_connected = False
        self._setup_mqtt()
        
        # Start background tasks
        self._start_background_tasks()
        
        logger.info(f"Camera Detector initialized: {self.config.SERVICE_ID}")
    
    def _parse_credentials(self) -> List[Tuple[str, str]]:
        """Parse camera credentials from config"""
        creds = []
        
        # If specific username/password are provided via environment, use ONLY those
        if self.config.DEFAULT_USERNAME and self.config.DEFAULT_PASSWORD:
            logger.info(f"Using provided credentials for user: {self.config.DEFAULT_USERNAME}")
            return [(self.config.DEFAULT_USERNAME, self.config.DEFAULT_PASSWORD)]
        
        try:
            for pair in self.config.CAMERA_CREDENTIALS.split(','):
                pair = pair.strip()
                if ':' in pair:
                    user, passwd = pair.split(':', 1)
                    user = user.strip()
                    passwd = passwd.strip()
                    # Validate credentials
                    if len(user) > 0:  # Username is required
                        creds.append((user, passwd))
                        logger.debug(f"Added credential for user: {user}")
        except Exception as e:
            logger.error(f"Error parsing credentials: {e}")
            # Fall back to default credentials
            creds = [("admin", ""), ("admin", "admin")]
        
        # Ensure we have at least one credential
        if not creds:
            creds = [("admin", ""), ("admin", "admin")]
            
        return creds
    
    def _setup_mqtt(self):
        """Setup MQTT client with resilient connection"""
        self.mqtt_client = mqtt.Client(
            client_id=self.config.SERVICE_ID,
            clean_session=False
        )
        
        # Set callbacks
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
        
        # Configure TLS if enabled
        if self.config.MQTT_TLS:
            import ssl
            self.mqtt_client.tls_set(
                ca_certs=self.config.TLS_CA_PATH,
                cert_reqs=ssl.CERT_REQUIRED,
                tls_version=ssl.PROTOCOL_TLS
            )
        
        # Set LWT
        lwt_topic = f"{self.config.TOPIC_HEALTH}/{self.config.NODE_ID}/lwt"
        lwt_payload = json.dumps({
            'node_id': self.config.NODE_ID,
            'service': 'camera_detector',
            'status': 'offline',
            'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        })
        self.mqtt_client.will_set(lwt_topic, lwt_payload, qos=1, retain=True)
        
        # Connect
        self._mqtt_connect_with_retry()
    
    def _mqtt_connect_with_retry(self):
        """Connect to MQTT with exponential backoff retry logic"""
        attempt = 0
        max_attempts = 10
        
        while attempt < max_attempts:
            try:
                port = 8883 if self.config.MQTT_TLS else 1883
                self.mqtt_client.connect(
                    self.config.MQTT_BROKER,
                    port,
                    keepalive=60
                )
                self.mqtt_client.loop_start()
                logger.info(f"MQTT connection initiated to {self.config.MQTT_BROKER}:{port}")
                break
            except Exception as e:
                attempt += 1
                delay = min(5 * (2 ** attempt), 300)  # Exponential backoff, max 5 minutes
                logger.error(f"MQTT connection failed (attempt {attempt}/{max_attempts}): {e}")
                if attempt >= max_attempts:
                    logger.error("Max MQTT connection attempts reached, service will run degraded")
                    break
                logger.info(f"Retrying in {delay}s...")
                time.sleep(delay)
    
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            self.mqtt_connected = True
            logger.info("MQTT connected successfully")
            self._publish_health()
        else:
            self.mqtt_connected = False
            logger.error(f"MQTT connection failed with code {rc}")
    
    def _on_mqtt_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback"""
        self.mqtt_connected = False
        logger.warning(f"MQTT disconnected with code {rc}")
    
    def _start_background_tasks(self):
        """Start background tasks"""
        # Discovery task
        threading.Thread(target=self._discovery_loop, daemon=True).start()
        
        # Health check task
        threading.Thread(target=self._health_check_loop, daemon=True).start()
        
        # MAC tracking task
        if self.config.MAC_TRACKING_ENABLED:
            threading.Thread(target=self._mac_tracking_loop, daemon=True).start()
        
        # Periodic health reporting
        threading.Timer(60, self._periodic_health_report).start()
        
        # Network change detection
        if self.config.SMART_DISCOVERY_ENABLED:
            threading.Thread(target=self._network_change_monitor, daemon=True).start()
    
    def _get_local_networks(self) -> List[str]:
        """Get local network subnets"""
        networks = []
        seen_networks = set()
        
        for iface in netifaces.interfaces():
            try:
                addrs = netifaces.ifaddresses(iface)
            except (ValueError, OSError) as e:
                logger.debug(f"Error getting addresses for interface {iface}: {e}")
                continue
                
            if netifaces.AF_INET in addrs:
                for addr in addrs[netifaces.AF_INET]:
                    ip = addr.get('addr')
                    netmask = addr.get('netmask')
                    
                    if ip and netmask and not ip.startswith('127.'):
                        try:
                            # Calculate network
                            network = ipaddress.IPv4Network(f"{ip}/{netmask}", strict=False)
                            network_str = str(network)
                            
                            # Avoid duplicates
                            if network_str not in seen_networks:
                                networks.append(network_str)
                                seen_networks.add(network_str)
                                logger.debug(f"Found network {network_str} on interface {iface}")
                        except Exception as e:
                            logger.debug(f"Error processing network on {iface}: {e}")
        
        # If we're on a /22 or larger network, also scan common camera subnets
        for network_str in list(networks):
            try:
                network = ipaddress.IPv4Network(network_str)
                if network.prefixlen <= 22:  # Large network
                    # Add adjacent /24 subnets that might have cameras
                    base_octets = str(network.network_address).split('.')[:2]
                    base = '.'.join(base_octets)
                    
                    # Common camera subnets
                    for third_octet in [1, 4, 5, 100, 101]:
                        subnet = f"{base}.{third_octet}.0/24"
                        if subnet not in seen_networks:
                            networks.append(subnet)
                            seen_networks.add(subnet)
                            logger.debug(f"Added adjacent subnet {subnet} for camera discovery")
            except Exception as e:
                logger.debug(f"Error processing adjacent subnet: {e}")
        
        return networks
    
    def _discovery_loop(self):
        """Main discovery loop with smart resource management"""
        while True:
            try:
                # Determine discovery mode
                if self.config.SMART_DISCOVERY_ENABLED:
                    # Check if we're in steady state
                    current_camera_count = len(self.cameras)
                    
                    if self.discovery_count < self.config.INITIAL_DISCOVERY_COUNT:
                        # Initial aggressive discovery
                        logger.info(f"Initial discovery scan {self.discovery_count + 1}/{self.config.INITIAL_DISCOVERY_COUNT}")
                        self._run_full_discovery()
                        interval = self.config.DISCOVERY_INTERVAL
                    
                    elif not self.is_steady_state:
                        # Check for stability
                        if current_camera_count == self.last_camera_count:
                            self.stable_count += 1
                            if self.stable_count >= 3:
                                self.is_steady_state = True
                                logger.info("Entering steady-state mode - reducing discovery frequency")
                        else:
                            self.stable_count = 0
                        
                        self._run_full_discovery()
                        interval = self.config.DISCOVERY_INTERVAL
                    
                    else:
                        # Steady state - reduced scanning
                        time_since_full = time.time() - self.last_full_discovery
                        
                        if time_since_full >= self.config.STEADY_STATE_INTERVAL:
                            # Periodic full scan
                            logger.info("Running periodic full discovery in steady state")
                            self._run_full_discovery()
                            interval = self.config.STEADY_STATE_INTERVAL
                        else:
                            # Quick health check only
                            logger.debug("Running quick health check")
                            self._run_quick_health_check()
                            interval = self.config.QUICK_CHECK_INTERVAL
                    
                    self.last_camera_count = current_camera_count
                    self.discovery_count += 1
                
                else:
                    # Traditional full discovery every interval
                    self._run_full_discovery()
                    interval = self.config.DISCOVERY_INTERVAL
                
            except Exception as e:
                logger.error(f"Discovery error: {e}", exc_info=True)
                interval = self.config.DISCOVERY_INTERVAL
            
            time.sleep(interval)
    
    def _run_full_discovery(self):
        """Run full discovery scan"""
        logger.info("Starting full camera discovery...")
        start_time = time.time()
        
        # Run discovery methods in parallel
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            # Update MAC mappings first if enabled
            if self.config.MAC_TRACKING_ENABLED:
                futures.append(
                    executor.submit(self._update_mac_mappings)
                )
            
            # Run discovery methods in parallel
            futures.extend([
                executor.submit(self._discover_onvif_cameras),
                executor.submit(self._discover_mdns_cameras),
                executor.submit(self._scan_rtsp_ports)
            ])
            
            # Wait for all to complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Discovery task error: {e}")
        
        # Update Frigate config after discovery
        if self.config.FRIGATE_UPDATE_ENABLED:
            self._update_frigate_config()
        
        # Update known IPs
        with self.lock:
            self.known_camera_ips = {cam.ip for cam in self.cameras.values()}
        
        self.last_full_discovery = time.time()
        elapsed = time.time() - start_time
        logger.info(f"Full discovery completed in {elapsed:.1f} seconds")
    
    def _run_quick_health_check(self):
        """Quick health check of known cameras without full network scan"""
        start_time = time.time()
        checked = 0
        online = 0
        
        with self.lock:
            cameras_to_check = list(self.cameras.values())
        
        # Check each known camera's availability
        import concurrent.futures
        
        def check_camera_health(camera):
            try:
                # Quick TCP check on RTSP port
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex((camera.ip, self.config.RTSP_PORT))
                sock.close()
                
                camera.online = (result == 0)
                camera.last_seen = time.time() if camera.online else camera.last_seen
                
                return camera.online
            except Exception as e:
                logger.debug(f"Health check error for {camera.ip}: {e}")
                return False
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(check_camera_health, cam): cam for cam in cameras_to_check}
            
            for future in concurrent.futures.as_completed(futures):
                checked += 1
                if future.result():
                    online += 1
        
        elapsed = time.time() - start_time
        logger.debug(f"Quick health check: {online}/{checked} cameras online in {elapsed:.1f}s")
        
        # Check if any cameras went offline
        newly_offline = []
        with self.lock:
            for camera in self.cameras.values():
                if not camera.online and camera.stream_active:
                    camera.stream_active = False
                    newly_offline.append(camera)
        
        # Publish status updates for offline cameras
        for camera in newly_offline:
            self._publish_camera_status(camera, "offline")
    
    def _mac_tracking_loop(self):
        """Periodic MAC address tracking"""
        while True:
            try:
                self._update_mac_mappings()
            except Exception as e:
                logger.error(f"MAC tracking error: {e}")
            
            time.sleep(60)  # Every minute
    
    def _network_change_monitor(self):
        """Monitor for network changes that require re-discovery"""
        last_networks = set(self._get_local_networks())
        
        while True:
            try:
                time.sleep(30)  # Check every 30 seconds
                
                current_networks = set(self._get_local_networks())
                
                if current_networks != last_networks:
                    logger.info("Network change detected - triggering discovery")
                    # Reset steady state
                    self.is_steady_state = False
                    self.stable_count = 0
                    # Trigger immediate discovery
                    threading.Thread(target=self._run_full_discovery, daemon=True).start()
                    
                last_networks = current_networks
                
            except Exception as e:
                logger.error(f"Network monitor error: {e}")
                time.sleep(60)
    
    def _update_mac_mappings(self):
        """Update MAC address mappings"""
        networks = self._get_local_networks()
        
        for network in networks:
            results = self.mac_tracker.scan_network(network)
            
            # Update camera IPs based on MAC
            with self.lock:
                for mac, camera in self.cameras.items():
                    new_ip = self.mac_tracker.get_ip_for_mac(mac)
                    if new_ip and new_ip != camera.ip:
                        camera.update_ip(new_ip)
                        self._publish_camera_status(camera, "ip_changed")
    
    def _discover_onvif_cameras(self):
        """Discover cameras using ONVIF WS-Discovery"""
        wsd = None
        try:
            wsd = WSDiscovery()
            wsd.start()
            
            services = wsd.searchServices(timeout=10)
            logger.info(f"WS-Discovery found {len(services)} services")
            
            for service in services:
                try:
                    # Extract camera info
                    xaddrs = service.getXAddrs()
                    types = service.getTypes()
                    scopes = service.getScopes()
                    
                    logger.debug(f"WS-Discovery service - XAddrs: {xaddrs}, Types: {types}, Scopes: {scopes}")
                    
                    if not xaddrs:
                        continue
                    
                    # Look for ONVIF device - be more inclusive
                    is_camera = False
                    type_str = ' '.join(str(t).lower() for t in types)
                    scope_str = ' '.join(str(s).lower() for s in scopes)
                    
                    # Check for camera-related keywords
                    camera_keywords = ['onvif', 'networkvideodevice', 'networkvideo', 'video', 'camera', 'imaging']
                    
                    for keyword in camera_keywords:
                        if keyword in type_str or keyword in scope_str:
                            is_camera = True
                            logger.debug(f"Found camera keyword '{keyword}' in service")
                            break
                    
                    if not is_camera:
                        logger.debug(f"Service at {xaddrs[0] if xaddrs else 'unknown'} not identified as camera")
                        continue
                    
                    onvif_url = xaddrs[0]
                    parsed = urlparse(onvif_url)
                    ip = parsed.hostname
                    
                    # Handle malformed URLs like http://[]/onvif/device_service
                    if not ip or ip == '[]':
                        logger.debug(f"Malformed ONVIF URL: {onvif_url}")
                        
                        # Try to extract IP from service source
                        # The WS-Discovery response comes from somewhere
                        try:
                            # Get the source address from the service
                            # This requires accessing the underlying socket info
                            # For now, skip this service
                            logger.debug(f"Skipping service with malformed URL: {onvif_url}")
                            continue
                        except Exception as e:
                            logger.debug(f"Error parsing ONVIF service URL: {e}")
                            continue
                    
                    if not ip:
                        continue
                    
                    # Get or determine MAC address
                    mac = self.mac_tracker.get_mac_for_ip(ip)
                    if not mac:
                        # Try to get MAC via ARP
                        mac = self._get_mac_address(ip)
                        if mac:
                            self.mac_tracker.update(mac, ip)
                        else:
                            mac = f"UNKNOWN-{ip.replace('.', '-')}"
                    
                    # Check if camera exists
                    with self.lock:
                        if mac in self.cameras:
                            camera = self.cameras[mac]
                            camera.last_seen = time.time()
                            if ip != camera.ip:
                                camera.update_ip(ip)
                        else:
                            # Create new camera
                            camera = Camera(
                                ip=ip,
                                mac=mac,
                                name=f"Camera-{mac[-8:]}",
                                onvif_url=onvif_url
                            )
                            self.cameras[mac] = camera
                    
                    # Get camera details via ONVIF
                    self._get_onvif_details(camera)
                    
                    # Publish discovery
                    self._publish_camera_discovery(camera)
                    
                except Exception as e:
                    logger.debug(f"Failed to process ONVIF service: {e}")
            
        except Exception as e:
            logger.error(f"ONVIF discovery error: {e}")
        finally:
            # Ensure WS-Discovery is properly stopped
            if wsd is not None:
                try:
                    wsd.stop()
                except:
                    pass
    
    def _discover_mdns_cameras(self):
        """Discover cameras via mDNS/Avahi"""
        try:
            # Use avahi-browse to find RTSP services
            result = subprocess.run(
                ['avahi-browse', '-ptr', '_rtsp._tcp'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if line.startswith('='):
                        parts = line.split(';')
                        if len(parts) >= 10:
                            ip = parts[7]
                            port = parts[8]
                            name = parts[3]
                            
                            if ip and port:
                                self._check_camera_at_ip(ip, f"mDNS: {name}")
            
        except Exception as e:
            logger.debug(f"mDNS discovery error: {e}")
    
    def _get_mac_address(self, ip: str) -> Optional[str]:
        """Get MAC address for IP using various methods"""
        # Try scapy ARP first
        mac = self.mac_tracker.get_mac_for_ip(ip)
        if mac:
            return mac
        
        # Try system ARP table
        try:
            result = subprocess.run(
                ['arp', '-n', ip],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if ip in line:
                        parts = line.split()
                        if len(parts) >= 3:
                            mac = parts[2]
                            if ':' in mac and mac != '<incomplete>':
                                return mac.upper()
        except:
            pass
        
        # Try to ping and check ARP (avoid infinite recursion)
        try:
            subprocess.run(['ping', '-c', '1', '-W', '1', ip], capture_output=True)
            time.sleep(0.1)
            # Only try system ARP again after ping, don't recurse
            result = subprocess.run(
                ['arp', '-n', ip],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if ip in line:
                        parts = line.split()
                        if len(parts) >= 3:
                            mac = parts[2]
                            if ':' in mac and mac != '<incomplete>':
                                return mac.upper()
        except:
            pass
        
        return None
    
    def _get_onvif_details(self, camera: Camera):
        """Get camera details via ONVIF"""
        for username, password in self.credentials:
            try:
                # Connect to camera
                mycam = ONVIFCamera(
                    camera.ip,
                    self.config.ONVIF_PORT,
                    username,
                    password,
                    wsdl_dir=os.path.join(os.path.dirname(__file__), 'wsdl')
                )
                
                # Get device info
                device_info = mycam.devicemgmt.GetDeviceInformation()
                camera.manufacturer = device_info.Manufacturer
                camera.model = device_info.Model
                camera.serial_number = getattr(device_info, 'SerialNumber', 'Unknown')
                camera.firmware_version = getattr(device_info, 'FirmwareVersion', 'Unknown')
                camera.name = f"{camera.manufacturer} {camera.model}"
                
                # Get capabilities
                capabilities = mycam.devicemgmt.GetCapabilities()
                camera.capabilities = {
                    'analytics': hasattr(capabilities, 'Analytics'),
                    'events': hasattr(capabilities, 'Events'),
                    'imaging': hasattr(capabilities, 'Imaging'),
                    'media': hasattr(capabilities, 'Media'),
                    'ptz': hasattr(capabilities, 'PTZ'),
                }
                
                # Get profiles
                media_service = mycam.create_media_service()
                profiles = media_service.GetProfiles()
                
                camera.profiles = []
                camera.rtsp_urls = {}
                
                for profile in profiles:
                    profile_obj = CameraProfile(
                        name=profile.Name,
                        token=profile.token
                    )
                    
                    # Get resolution
                    if hasattr(profile, 'VideoEncoderConfiguration'):
                        vec = profile.VideoEncoderConfiguration
                        if hasattr(vec, 'Resolution'):
                            profile_obj.resolution = (vec.Resolution.Width, vec.Resolution.Height)
                        if hasattr(vec, 'RateControl'):
                            profile_obj.framerate = vec.RateControl.FrameRateLimit
                        if hasattr(vec, 'Encoding'):
                            profile_obj.encoding = vec.Encoding
                    
                    camera.profiles.append(profile_obj)
                    
                    # Get RTSP URL
                    try:
                        uri = media_service.GetStreamUri({
                            'StreamSetup': {
                                'Stream': 'RTP-Unicast',
                                'Transport': {'Protocol': 'RTSP'}
                            },
                            'ProfileToken': profile.token
                        })
                        
                        if uri and uri.Uri:
                            # Build complete RTSP URL with credentials
                            rtsp_url = uri.Uri
                            parsed = urlparse(rtsp_url)
                            if username and password:
                                rtsp_url = f"{parsed.scheme}://{username}:{password}@{parsed.netloc}{parsed.path}"
                                if parsed.query:
                                    rtsp_url += f"?{parsed.query}"
                            
                            # Store URL by profile name
                            profile_key = profile.Name.lower().replace(' ', '_')
                            camera.rtsp_urls[profile_key] = rtsp_url
                            
                    except Exception as e:
                        logger.debug(f"Failed to get RTSP URL for profile {profile.Name}: {e}")
                
                # Store successful credentials
                camera.username = username
                camera.password = password
                camera.online = True
                
                # Get HTTP URL
                camera.http_url = f"http://{camera.ip}:{self.config.HTTP_PORT}"
                
                logger.info(f"ONVIF camera found: {camera.name} at {camera.ip} (MAC: {camera.mac})")
                return True
                
            except Exception as e:
                logger.debug(f"Failed ONVIF connection to {camera.ip} with {username}: {e}")
                continue
        
        return False
    
    def _check_camera_at_ip(self, ip: str, source: str = ""):
        """Check if there's a camera at the given IP"""
        try:
            logger.debug(f"Checking camera at {ip} from {source}")
            
            # Get or determine MAC
            mac = self.mac_tracker.get_mac_for_ip(ip)
            if not mac:
                mac = self._get_mac_address(ip)
                if mac:
                    self.mac_tracker.update(mac, ip)
                else:
                    mac = f"UNKNOWN-{ip.replace('.', '-')}"
            
            # Check if camera exists
            with self.lock:
                if mac in self.cameras:
                    camera = self.cameras[mac]
                    camera.last_seen = time.time()
                    if ip != camera.ip:
                        camera.update_ip(ip)
                else:
                    camera = Camera(
                        ip=ip,
                        mac=mac,
                        name=f"Camera-{mac[-8:]}"
                    )
                    self.cameras[mac] = camera
            
            logger.debug(f"Camera {mac} at {ip} - trying ONVIF...")
            
            # Try ONVIF first (with timeout protection)
            import threading
            onvif_done = threading.Event()
            onvif_result = [False]
            
            def try_onvif():
                try:
                    onvif_result[0] = self._get_onvif_details(camera)
                except Exception as e:
                    logger.debug(f"ONVIF error: {e}")
                finally:
                    onvif_done.set()
            
            onvif_thread = threading.Thread(target=try_onvif)
            onvif_thread.daemon = True
            
            try:
                onvif_thread.start()
                onvif_done_result = onvif_done.wait(timeout=self.config.ONVIF_TIMEOUT + 2)
            except RuntimeError:
                # Thread start was mocked in tests
                try_onvif()
                onvif_done_result = True
            
            if onvif_done_result:
                if onvif_result[0]:
                    self._publish_camera_discovery(camera)
                    logger.info(f"Camera discovered via ONVIF at {ip}")
                    return
            else:
                logger.debug(f"ONVIF timed out for {ip}")
            
            logger.debug(f"Camera {mac} at {ip} - trying RTSP...")
            
            # Try direct RTSP
            if self._check_rtsp_stream(camera):
                self._publish_camera_discovery(camera)
                logger.info(f"Camera discovered via RTSP at {ip}")
                return
            
            logger.debug(f"No valid camera found at {ip}")
                
        except Exception as e:
            logger.error(f"Failed to check camera at {ip}: {e}", exc_info=True)
    
    def _scan_rtsp_ports(self):
        """Scan network for RTSP ports"""
        try:
            networks = self._get_local_networks()
            logger.info(f"Starting RTSP port scan on {len(networks)} networks: {networks}")
            
            # Scan all networks in parallel for speed
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(networks))) as executor:
                futures = []
                for network in networks:
                    logger.debug(f"Submitting scan for network: {network}")
                    future = executor.submit(self._scan_single_network, network)
                    futures.append(future)
                
                # Wait for all scans to complete
                completed = 0
                for future in concurrent.futures.as_completed(futures):
                    completed += 1
                    try:
                        future.result()
                        logger.debug(f"Network scan completed ({completed}/{len(futures)})")
                    except Exception as e:
                        logger.error(f"Network scan error: {e}")
                    
        except Exception as e:
            logger.error(f"RTSP port scan error: {e}", exc_info=True)
    
    def _scan_single_network(self, network: str):
        """Scan a single network for RTSP ports"""
        # Try nmap first for faster scanning
        nmap_available = False
        try:
            result = subprocess.run(
                ['nmap', '-p', str(self.config.RTSP_PORT), '--open', '-sS', network],
                capture_output=True,
                text=True,
                timeout=30  # Reduced timeout per network
            )
            
            if result.returncode == 0:
                nmap_available = True
                current_ip = None
                for line in result.stdout.split('\n'):
                    if 'Nmap scan report for' in line:
                        # Extract IP
                        parts = line.split()
                        if len(parts) >= 5:
                            current_ip = parts[-1].strip('()')
                    elif 'open' in line and str(self.config.RTSP_PORT) in line and current_ip:
                        # Found open RTSP port
                        self._check_camera_at_ip(current_ip, "RTSP scan")
                        
        except FileNotFoundError:
            logger.debug("nmap not found, falling back to socket scanning")
        except Exception as e:
            logger.debug(f"nmap scan failed for {network}: {e}")
        
        # Fallback to socket scanning if nmap not available or failed
        if not nmap_available:
            logger.info(f"Using socket scanning for network {network}")
            self._socket_scan_network(network)
    
    def _socket_scan_network(self, network: str, target_range: Optional[Tuple[int, int]] = None):
        """Scan network using sockets when nmap is not available
        
        Args:
            network: Network to scan in CIDR notation
            target_range: Optional tuple of (start, end) for last octet to limit scan range
        """
        try:
            import concurrent.futures
            
            network_obj = ipaddress.IPv4Network(network)
            hosts = list(network_obj.hosts())
            
            # Apply target range filter if specified (for faster targeted scanning)
            if target_range and len(hosts) > 0:
                base_octets = str(hosts[0]).rsplit('.', 1)[0]
                filtered_hosts = []
                for host in hosts:
                    last_octet = int(str(host).split('.')[-1])
                    if target_range[0] <= last_octet <= target_range[1]:
                        filtered_hosts.append(host)
                if filtered_hosts:
                    logger.info(f"Targeted scan: {len(filtered_hosts)} IPs in range {target_range[0]}-{target_range[1]}")
                    hosts = filtered_hosts
            
            logger.info(f"Starting socket scan for network {network} with {len(hosts)} hosts")
            
            # Skip very large networks (likely Docker/virtualization)
            if len(hosts) > 5000:
                logger.info(f"Skipping very large network {network} (likely Docker/virtual)")
                return
                
            # In steady state, only scan for new IPs
            if self.is_steady_state and self.known_camera_ips:
                # Filter to only unknown IPs
                unknown_hosts = [h for h in hosts if str(h) not in self.known_camera_ips]
                if len(unknown_hosts) < len(hosts) / 2:
                    logger.info(f"Steady state: scanning only {len(unknown_hosts)} new IPs (skipping {len(hosts) - len(unknown_hosts)} known)")
                    hosts = unknown_hosts
                    if not hosts:
                        return
                
            # For large networks, prioritize common camera IP ranges
            if len(hosts) > 1000:
                logger.info(f"Large network {network}, focusing on common camera IP ranges...")
                # Common IP endings for cameras
                priority_endings = list(range(100, 255))
                priority_hosts = [h for h in hosts if int(str(h).split('.')[-1]) in priority_endings]
                other_hosts = [h for h in hosts if h not in priority_hosts]
                hosts = priority_hosts + other_hosts[:100]  # Limit scan
            
            logger.info(f"Socket scanning {len(hosts)} IPs on port {self.config.RTSP_PORT}...")
            
            found_count = 0
            
            def scan_ip(ip_str):
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(0.2)  # Reduced timeout for faster scanning
                    result = sock.connect_ex((ip_str, self.config.RTSP_PORT))
                    sock.close()
                    
                    if result == 0:
                        return ip_str
                except:
                    pass
                return None
            
            # Scan in parallel for speed - use more workers for faster scanning
            # Determine optimal worker count based on network size and available CPU cores
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            # Use more workers on multi-core systems (up to 500 for high-core systems)
            base_workers = min(500, cpu_count * 50)
            worker_count = min(base_workers, max(100, len(hosts) // 2))
            logger.info(f"Using {worker_count} workers for scanning ({cpu_count} CPU cores detected)")
            
            found_ips = []
            
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                    # Submit all tasks
                    future_to_ip = {}
                    for ip in hosts:
                        future = executor.submit(scan_ip, str(ip))
                        future_to_ip[future] = str(ip)
                    
                    # Wait for completion with timeout
                    done, not_done = concurrent.futures.wait(
                        future_to_ip.keys(), 
                        timeout=len(hosts) * 0.3,  # 0.3s per host max
                        return_when=concurrent.futures.ALL_COMPLETED
                    )
                    
                    # Process completed futures
                    for future in done:
                        try:
                            result = future.result()
                            if result:
                                found_count += 1
                                found_ips.append(result)
                                logger.info(f"Found open RTSP port at {result}")
                        except Exception as e:
                            logger.error(f"Error processing scan result: {e}")
                    
                    # Log any incomplete futures
                    if not_done:
                        logger.warning(f"{len(not_done)} scan tasks did not complete in time")
                        
            except Exception as e:
                logger.error(f"Error during concurrent scan: {e}")
            
            # Check cameras after scanning completes
            logger.info(f"Port scan complete. Checking {len(found_ips)} cameras...")
            for ip in found_ips:
                try:
                    self._check_camera_at_ip(ip, "Socket scan")
                except Exception as e:
                    logger.error(f"Error checking camera at {ip}: {e}")
            
            if found_count > 0:
                logger.info(f"Socket scan complete: found {found_count} cameras on {network}")
                        
        except Exception as e:
            logger.error(f"Socket scan error: {e}")
    
    def _check_rtsp_stream(self, camera: Camera) -> bool:
        """Check if camera has accessible RTSP stream"""
        # Common RTSP paths - prioritize Amcrest based on discovered cameras
        rtsp_paths = [
            '/cam/realmonitor?channel=1&subtype=0',  # Amcrest main stream
            '/cam/realmonitor?channel=1&subtype=1',  # Amcrest sub stream
            '/stream1',
            '/h264/ch1/main/av_stream',
            '/live/ch00_0',
            '/MediaInput/h264/stream_1',
            '/11',
            '/video1',
            '/streaming/channels/101',
            '/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp',
            '/live/0/MAIN',
            '/live/0/HIGH',
            '/axis-media/media.amp',
            '/stream',
            '/live',
            '/video'
        ]
        
        # If we know the manufacturer, prioritize their paths
        if camera.manufacturer and 'amcrest' in camera.manufacturer.lower():
            # Move Amcrest paths to front
            rtsp_paths = [
                '/cam/realmonitor?channel=1&subtype=0',
                '/cam/realmonitor?channel=1&subtype=1'
            ] + [p for p in rtsp_paths if '/cam/realmonitor' not in p]
        
        # Test credentials and paths in parallel for faster discovery
        import concurrent.futures
        
        def test_rtsp_url(cred_path):
            username, password, path = cred_path
            rtsp_url = f"rtsp://{camera.ip}:{self.config.RTSP_PORT}{path}"
            if username and password:
                rtsp_url = f"rtsp://{username}:{password}@{camera.ip}:{self.config.RTSP_PORT}{path}"
            
            if self._validate_rtsp_stream(rtsp_url):
                return (username, password, path, rtsp_url)
            return None
        
        # Create all combinations to test
        test_combinations = [
            (username, password, path)
            for username, password in self.credentials
            for path in rtsp_paths[:6]  # Test first 6 paths in parallel
        ]
        
        # Test in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(test_rtsp_url, combo) for combo in test_combinations]
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    username, password, path, rtsp_url = result
                    camera.rtsp_urls['main'] = rtsp_url
                    camera.username = username
                    camera.password = password
                    camera.online = True
                    camera.stream_active = True
                    camera.last_validated = time.time()
                    
                    logger.info(f"RTSP stream found at {camera.ip} (MAC: {camera.mac}) - {path}")
                    
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    
                    return True
        
        # If first batch didn't work, try remaining paths sequentially
        for username, password in self.credentials:
            for path in rtsp_paths[6:]:
                rtsp_url = f"rtsp://{camera.ip}:{self.config.RTSP_PORT}{path}"
                if username and password:
                    rtsp_url = f"rtsp://{username}:{password}@{camera.ip}:{self.config.RTSP_PORT}{path}"
                
                if self._validate_rtsp_stream(rtsp_url):
                    camera.rtsp_urls['main'] = rtsp_url
                    camera.username = username
                    camera.password = password
                    camera.online = True
                    camera.stream_active = True
                    camera.last_validated = time.time()
                    
                    logger.info(f"RTSP stream found at {camera.ip} (MAC: {camera.mac}) - {path}")
                    return True
        
        return False
    
    def _validate_rtsp_stream(self, rtsp_url: str) -> bool:
        """Validate RTSP stream is accessible"""
        import threading
        result = [False]
        
        def validate_with_timeout():
            cap = None
            try:
                # Validate timeout value
                timeout_ms = max(self.config.RTSP_TIMEOUT * 1000, 1000)  # Minimum 1 second
                
                # Set OpenCV capture properties for faster timeout
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)
                
                # Check if capture opened successfully
                if not cap.isOpened():
                    logger.debug(f"RTSP stream failed to open: {rtsp_url}")
                    return
                
                # Try to read a frame
                ret, frame = cap.read()
                
                result[0] = ret and frame is not None
                
            except Exception as e:
                logger.debug(f"RTSP validation error for {rtsp_url}: {e}")
            finally:
                # Ensure capture is always released
                if cap is not None:
                    try:
                        cap.release()
                    except:
                        pass
        
        # Run validation in a thread with timeout
        thread = threading.Thread(target=validate_with_timeout)
        thread.daemon = True
        
        try:
            thread.start()
            thread.join(timeout=self.config.RTSP_TIMEOUT + 1)  # Give 1 extra second
            
            if thread.is_alive():
                logger.warning(f"RTSP validation timed out for {rtsp_url}")
                return False
                
        except RuntimeError:
            # Thread start was mocked in tests
            validate_with_timeout()
            
        return result[0]
    
    def _health_check_loop(self):
        """Check health of known cameras"""
        while True:
            try:
                current_time = time.time()
                cameras_to_rediscover = []
                
                with self.lock:
                    for mac, camera in list(self.cameras.items()):
                        # Check if camera is offline
                        if current_time - camera.last_seen > self.config.OFFLINE_THRESHOLD:
                            if camera.online:
                                camera.online = False
                                camera.stream_active = False
                                self._publish_camera_status(camera, "offline")
                                logger.warning(f"Camera {camera.name} ({camera.ip}) is offline")
                                # Mark for immediate rediscovery (likely DHCP address change)
                                cameras_to_rediscover.append(camera)
                        
                        # Validate RTSP stream periodically
                        elif camera.online and camera.primary_rtsp_url:
                            if current_time - camera.last_validated > self.config.HEALTH_CHECK_INTERVAL:
                                if self._validate_rtsp_stream(camera.primary_rtsp_url):
                                    camera.stream_active = True
                                    camera.last_validated = current_time
                                else:
                                    camera.stream_active = False
                                    self._publish_camera_status(camera, "stream_error")
                                    logger.warning(f"Camera {camera.name} stream error")
                
                # Trigger immediate rediscovery for offline cameras
                if cameras_to_rediscover:
                    logger.info(f"Triggering immediate rediscovery for {len(cameras_to_rediscover)} offline cameras")
                    threading.Thread(target=self._rediscover_offline_cameras, 
                                   args=(cameras_to_rediscover,), daemon=True).start()
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
            
            time.sleep(30)
    
    def _rediscover_offline_cameras(self, offline_cameras: List[Camera]):
        """Quickly rediscover cameras that went offline (likely DHCP change)"""
        try:
            # Update MAC mappings first
            if self.config.MAC_TRACKING_ENABLED:
                self._update_mac_mappings()
            
            # Check if MAC mapping found new IPs
            for camera in offline_cameras:
                new_ip = self.mac_tracker.get_ip_for_mac(camera.mac)
                if new_ip and new_ip != camera.ip:
                    logger.info(f"Found camera {camera.name} at new IP {new_ip} (was {camera.ip})")
                    camera.update_ip(new_ip)
                    # Validate the camera at new IP
                    if self._validate_camera_at_new_ip(camera):
                        self._publish_camera_status(camera, "ip_changed")
                        continue
            
            # For cameras still not found, do a targeted scan
            networks = self._get_local_networks()
            for network in networks[:3]:  # Limit to first 3 networks for speed
                logger.info(f"Quick scan for offline cameras on {network}")
                # Use high parallelization for fast scan
                self._quick_scan_for_cameras(network, offline_cameras)
                
        except Exception as e:
            logger.error(f"Error in offline camera rediscovery: {e}")
    
    def _validate_camera_at_new_ip(self, camera: Camera) -> bool:
        """Validate camera at its current IP address"""
        try:
            # Quick RTSP check with known credentials
            if camera.username and camera.password:
                # Try the known working RTSP path first
                if camera.rtsp_urls:
                    for name, url in camera.rtsp_urls.items():
                        # Update URL with new IP
                        parsed = urlparse(url)
                        new_url = f"{parsed.scheme}://{camera.username}:{camera.password}@{camera.ip}:{parsed.port or 554}{parsed.path}"
                        if parsed.query:
                            new_url += f"?{parsed.query}"
                        
                        if self._validate_rtsp_stream(new_url):
                            camera.rtsp_urls[name] = new_url
                            camera.online = True
                            camera.stream_active = True
                            camera.last_seen = time.time()
                            camera.last_validated = time.time()
                            return True
            
            # Fall back to full check
            return self._check_rtsp_stream(camera)
            
        except Exception as e:
            logger.error(f"Error validating camera at new IP: {e}")
            return False
    
    def _quick_scan_for_cameras(self, network: str, target_cameras: List[Camera]):
        """Quick targeted scan for specific cameras"""
        try:
            import concurrent.futures
            import multiprocessing
            
            network_obj = ipaddress.IPv4Network(network)
            hosts = list(network_obj.hosts())
            
            # Use maximum parallelization for quick scan
            cpu_count = multiprocessing.cpu_count()
            worker_count = min(500, cpu_count * 100, len(hosts))
            
            found_cameras = []
            
            def check_ip_for_camera(ip_str):
                # Quick check if any target camera responds at this IP
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.2)  # Very short timeout for quick scan
                result = sock.connect_ex((ip_str, self.config.RTSP_PORT))
                sock.close()
                
                if result == 0:
                    # RTSP port is open, check if it's one of our cameras
                    mac = self._get_mac_address(ip_str)
                    if mac:
                        for camera in target_cameras:
                            if camera.mac == mac:
                                logger.info(f"Found offline camera {camera.name} at {ip_str}")
                                camera.update_ip(ip_str)
                                if self._validate_camera_at_new_ip(camera):
                                    return camera
                return None
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {executor.submit(check_ip_for_camera, str(ip)): str(ip) for ip in hosts}
                
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result:
                        found_cameras.append(result)
                        # Remove from target list
                        target_cameras.remove(result)
                        if not target_cameras:
                            # All cameras found
                            break
                            
        except Exception as e:
            logger.error(f"Quick scan error: {e}")
    
    def _publish_camera_discovery(self, camera: Camera):
        """Publish camera discovery event"""
        try:
            payload = {
                'event': 'discovered',
                'camera': camera.to_dict(),
                'node_id': self.config.NODE_ID,
                'timestamp': datetime.now(timezone.utc).isoformat() + 'Z'
            }
            
            topic = f"{self.config.TOPIC_DISCOVERY}/{camera.id}"
            self.mqtt_client.publish(topic, json.dumps(payload), qos=1, retain=True)
            logger.debug(f"Published discovery for camera {camera.id}")
            
        except Exception as e:
            logger.error(f"Failed to publish discovery: {e}")
    
    def _publish_camera_status(self, camera: Camera, status: str):
        """Publish camera status change"""
        try:
            payload = {
                'camera_id': camera.id,
                'status': status,
                'camera': camera.to_dict(),
                'node_id': self.config.NODE_ID,
                'timestamp': datetime.now(timezone.utc).isoformat() + 'Z'
            }
            
            topic = f"{self.config.TOPIC_STATUS}/{camera.id}"
            self.mqtt_client.publish(topic, json.dumps(payload), qos=1)
            logger.info(f"Camera {camera.name} status: {status}")
            
        except Exception as e:
            logger.error(f"Failed to publish status: {e}")
    
    def _update_frigate_config(self):
        """Update Frigate camera configuration"""
        try:
            # Base Frigate config
            frigate_config = {
                'mqtt': {
                    'host': self.config.MQTT_BROKER,
                    'port': self.config.MQTT_PORT,
                    'topic_prefix': 'frigate',
                    'client_id': 'frigate',
                    'stats_interval': 60
                },
                'detectors': {
                    'default': {
                        'type': 'cpu',
                        'num_threads': 3
                    }
                },
                'cameras': {}
            }
            
            # Add cameras
            with self.lock:
                for camera in self.cameras.values():
                    if camera.online and camera.primary_rtsp_url:
                        camera_config = camera.to_frigate_config()
                        if camera_config:
                            frigate_config['cameras'].update(camera_config)
            
            # Write to file
            os.makedirs(os.path.dirname(self.config.FRIGATE_CONFIG_PATH), exist_ok=True)
            with open(self.config.FRIGATE_CONFIG_PATH, 'w') as f:
                yaml.dump(frigate_config, f, default_flow_style=False, sort_keys=False)
            
            # Publish to MQTT
            self.mqtt_client.publish(
                self.config.TOPIC_FRIGATE_CONFIG,
                json.dumps(frigate_config),
                qos=1,
                retain=True
            )
            
            # Trigger Frigate reload
            self.mqtt_client.publish(self.config.FRIGATE_RELOAD_TOPIC, "", qos=1)
            
            logger.info(f"Updated Frigate config with {len(frigate_config['cameras'])} cameras")
            
        except Exception as e:
            logger.error(f"Failed to update Frigate config: {e}")
    
    def _periodic_health_report(self):
        """Publish periodic health report"""
        try:
            self._publish_health()
        except Exception as e:
            logger.error(f"Health report error: {e}")
        
        # Reschedule
        threading.Timer(60, self._periodic_health_report).start()
    
    def _publish_health(self):
        """Publish health status"""
        with self.lock:
            online_cameras = [c for c in self.cameras.values() if c.online]
            streaming_cameras = [c for c in self.cameras.values() if c.stream_active]
            
            payload = {
                'node_id': self.config.NODE_ID,
                'service': 'camera_detector',
                'timestamp': datetime.now(timezone.utc).isoformat() + 'Z',
                'status': 'online' if self.mqtt_connected else 'degraded',
                'stats': {
                    'total_cameras': len(self.cameras),
                    'online_cameras': len(online_cameras),
                    'streaming_cameras': len(streaming_cameras),
                    'discovery_interval': self.config.DISCOVERY_INTERVAL,
                    'mac_tracking_enabled': self.config.MAC_TRACKING_ENABLED,
                },
                'cameras': {
                    cam.id: {
                        'name': cam.name,
                        'ip': cam.ip,
                        'mac': cam.mac,
                        'online': cam.online,
                        'streaming': cam.stream_active,
                        'ip_history': cam.ip_history
                    }
                    for cam in self.cameras.values()
                }
            }
        
        try:
            self.mqtt_client.publish(
                self.config.TOPIC_HEALTH,
                json.dumps(payload),
                qos=1,
                retain=True
            )
        except Exception as e:
            logger.error(f"Failed to publish health: {e}")
    
    def get_health(self) -> dict:
        """Get health status for health check"""
        with self.lock:
            return {
                'healthy': self.mqtt_connected and len(self.cameras) > 0,
                'cameras': len(self.cameras),
                'online': len([c for c in self.cameras.values() if c.online])
            }
    
    def run(self):
        """Main run loop"""
        logger.info("Camera Detector Service started")
        
        try:
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean shutdown"""
        logger.info("Cleaning up Camera Detector Service")
        
        # Mark all cameras offline
        with self.lock:
            for camera in self.cameras.values():
                if camera.online:
                    camera.online = False
                    self._publish_camera_status(camera, "offline")
        
        # Publish offline status
        try:
            lwt_payload = json.dumps({
                'node_id': self.config.NODE_ID,
                'service': 'camera_detector',
                'status': 'offline',
                'timestamp': datetime.now(timezone.utc).isoformat() + 'Z'
            })
            self.mqtt_client.publish(
                f"{self.config.TOPIC_HEALTH}/{self.config.NODE_ID}/lwt",
                lwt_payload,
                qos=1,
                retain=True
            )
        except:
            pass
        
        # Disconnect MQTT
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()

# ─────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────
def main():
    detector = CameraDetector()
    detector.run()

if __name__ == "__main__":
    main()
