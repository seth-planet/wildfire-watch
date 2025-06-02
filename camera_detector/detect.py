#!/usr/bin/env python3
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
from datetime import datetime, timedelta
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
    MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
    MQTT_TLS = os.getenv("MQTT_TLS", "false").lower() == "true"
    TLS_CA_PATH = os.getenv("TLS_CA_PATH", "/mnt/data/certs/ca.crt")
    
    # Discovery Settings
    DISCOVERY_INTERVAL = int(os.getenv("DISCOVERY_INTERVAL", "300"))  # 5 minutes
    RTSP_TIMEOUT = int(os.getenv("RTSP_TIMEOUT", "10"))
    ONVIF_TIMEOUT = int(os.getenv("ONVIF_TIMEOUT", "5"))
    MAC_TRACKING_ENABLED = os.getenv("MAC_TRACKING_ENABLED", "true").lower() == "true"
    
    # Camera Settings
    DEFAULT_USERNAME = os.getenv("CAMERA_USERNAME", "admin")
    DEFAULT_PASSWORD = os.getenv("CAMERA_PASSWORD", "")
    RTSP_PORT = int(os.getenv("RTSP_PORT", "554"))
    ONVIF_PORT = int(os.getenv("ONVIF_PORT", "80"))
    HTTP_PORT = int(os.getenv("HTTP_PORT", "80"))
    
    # Camera credentials (comma-separated user:pass pairs)
    CAMERA_CREDENTIALS = os.getenv("CAMERA_CREDENTIALS", "admin:,admin:admin,admin:12345")
    
    # Health Monitoring
    HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", "60"))
    OFFLINE_THRESHOLD = int(os.getenv("OFFLINE_THRESHOLD", "180"))  # 3 minutes
    
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
            logger.error(f"ARP scan failed: {e}")
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
        for pair in self.config.CAMERA_CREDENTIALS.split(','):
            if ':' in pair:
                user, passwd = pair.split(':', 1)
                creds.append((user.strip(), passwd.strip()))
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
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        })
        self.mqtt_client.will_set(lwt_topic, lwt_payload, qos=1, retain=True)
        
        # Connect
        self._mqtt_connect_with_retry()
    
    def _mqtt_connect_with_retry(self):
        """Connect to MQTT with retry logic"""
        while True:
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
                logger.error(f"MQTT connection failed: {e}")
                time.sleep(5)
    
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
    
    def _get_local_networks(self) -> List[str]:
        """Get local network subnets"""
        networks = []
        
        for iface in netifaces.interfaces():
            addrs = netifaces.ifaddresses(iface)
            if netifaces.AF_INET in addrs:
                for addr in addrs[netifaces.AF_INET]:
                    ip = addr.get('addr')
                    netmask = addr.get('netmask')
                    
                    if ip and netmask and not ip.startswith('127.'):
                        try:
                            # Calculate network
                            network = ipaddress.IPv4Network(f"{ip}/{netmask}", strict=False)
                            networks.append(str(network))
                        except:
                            pass
        
        return networks
    
    def _discovery_loop(self):
        """Main discovery loop"""
        while True:
            try:
                logger.info("Starting camera discovery...")
                
                # Update MAC mappings first
                if self.config.MAC_TRACKING_ENABLED:
                    self._update_mac_mappings()
                
                # ONVIF/WS-Discovery
                self._discover_onvif_cameras()
                
                # mDNS discovery
                self._discover_mdns_cameras()
                
                # Scan for RTSP ports
                self._scan_rtsp_ports()
                
                # Update Frigate config
                if self.config.FRIGATE_UPDATE_ENABLED:
                    self._update_frigate_config()
                
            except Exception as e:
                logger.error(f"Discovery error: {e}", exc_info=True)
            
            time.sleep(self.config.DISCOVERY_INTERVAL)
    
    def _mac_tracking_loop(self):
        """Periodic MAC address tracking"""
        while True:
            try:
                self._update_mac_mappings()
            except Exception as e:
                logger.error(f"MAC tracking error: {e}")
            
            time.sleep(60)  # Every minute
    
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
        try:
            wsd = WSDiscovery()
            wsd.start()
            
            services = wsd.searchServices(timeout=10)
            
            for service in services:
                try:
                    # Extract camera info
                    xaddrs = service.getXAddrs()
                    if not xaddrs:
                        continue
                    
                    # Check if this is a camera service
                    types = service.getTypes()
                    scopes = service.getScopes()
                    
                    # Look for ONVIF device
                    is_camera = False
                    for t in types:
                        if 'onvif' in str(t).lower() or 'networkvideodevice' in str(t).lower():
                            is_camera = True
                            break
                    
                    if not is_camera:
                        # Check scopes
                        for scope in scopes:
                            if 'onvif' in str(scope).lower() or 'camera' in str(scope).lower():
                                is_camera = True
                                break
                    
                    if not is_camera:
                        continue
                    
                    onvif_url = xaddrs[0]
                    parsed = urlparse(onvif_url)
                    ip = parsed.hostname
                    
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
            
            wsd.stop()
            
        except Exception as e:
            logger.error(f"ONVIF discovery error: {e}")
    
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
        
        # Try to ping and check ARP
        try:
            subprocess.run(['ping', '-c', '1', '-W', '1', ip], capture_output=True)
            time.sleep(0.1)
            return self._get_mac_address(ip)  # Recursive call after ping
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
            
            # Try ONVIF first
            if self._get_onvif_details(camera):
                self._publish_camera_discovery(camera)
                return
            
            # Try direct RTSP
            if self._check_rtsp_stream(camera):
                self._publish_camera_discovery(camera)
                return
                
        except Exception as e:
            logger.debug(f"Failed to check camera at {ip}: {e}")
    
    def _scan_rtsp_ports(self):
        """Scan network for RTSP ports"""
        try:
            networks = self._get_local_networks()
            
            for network in networks:
                # Quick scan for RTSP port
                try:
                    result = subprocess.run(
                        ['nmap', '-p', str(self.config.RTSP_PORT), '--open', '-sS', network],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    
                    if result.returncode == 0:
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
                                
                except Exception as e:
                    logger.debug(f"nmap scan failed: {e}")
                    
        except Exception as e:
            logger.debug(f"RTSP port scan error: {e}")
    
    def _check_rtsp_stream(self, camera: Camera) -> bool:
        """Check if camera has accessible RTSP stream"""
        # Common RTSP paths
        rtsp_paths = [
            '/stream1',
            '/cam/realmonitor?channel=1&subtype=0',
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
        
        # Try each credential combination
        for username, password in self.credentials:
            for path in rtsp_paths:
                rtsp_url = f"rtsp://{camera.ip}:{self.config.RTSP_PORT}{path}"
                if username and password:
                    rtsp_url = f"rtsp://{username}:{password}@{camera.ip}:{self.config.RTSP_PORT}{path}"
                
                if self._validate_rtsp_stream(rtsp_url):
                    # Found working stream
                    camera.rtsp_urls['main'] = rtsp_url
                    camera.username = username
                    camera.password = password
                    camera.online = True
                    camera.stream_active = True
                    camera.last_validated = time.time()
                    
                    logger.info(f"RTSP stream found at {camera.ip} (MAC: {camera.mac})")
                    return True
        
        return False
    
    def _validate_rtsp_stream(self, rtsp_url: str) -> bool:
        """Validate RTSP stream is accessible"""
        try:
            # Set OpenCV capture properties for faster timeout
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.config.RTSP_TIMEOUT * 1000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.config.RTSP_TIMEOUT * 1000)
            
            # Try to read a frame
            ret, frame = cap.read()
            cap.release()
            
            return ret and frame is not None
            
        except Exception as e:
            logger.debug(f"RTSP validation error for {rtsp_url}: {e}")
            return False
    
    def _health_check_loop(self):
        """Check health of known cameras"""
        while True:
            try:
                current_time = time.time()
                
                with self.lock:
                    for mac, camera in list(self.cameras.items()):
                        # Check if camera is offline
                        if current_time - camera.last_seen > self.config.OFFLINE_THRESHOLD:
                            if camera.online:
                                camera.online = False
                                camera.stream_active = False
                                self._publish_camera_status(camera, "offline")
                                logger.warning(f"Camera {camera.name} ({camera.ip}) is offline")
                        
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
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
            
            time.sleep(30)
    
    def _publish_camera_discovery(self, camera: Camera):
        """Publish camera discovery event"""
        try:
            payload = {
                'event': 'discovered',
                'camera': camera.to_dict(),
                'node_id': self.config.NODE_ID,
                'timestamp': datetime.utcnow().isoformat() + 'Z'
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
                'timestamp': datetime.utcnow().isoformat() + 'Z'
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
                'timestamp': datetime.utcnow().isoformat() + 'Z',
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
                'timestamp': datetime.utcnow().isoformat() + 'Z'
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
