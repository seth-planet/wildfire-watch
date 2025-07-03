#!/usr/bin/env python3.12
"""Camera Detector with Resilient Network Operations

This is an updated version of detect.py that uses the ResilientMQTTClient
and NetworkTimeoutUtils to prevent hanging and ensure graceful degradation.
"""

import os
import sys
import queue
import logging
import json
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mqtt_resilient import ResilientMQTTClient, ConnectionState
from network_timeout_utils import NetworkUtils, ONVIFTimeout, RTSPTimeout, NetworkTimeoutError

# Import original components we're not changing
from detect import (
    Config, Camera, MACTracker, FrigateConfigGenerator,
    validate_rtsp_url_patterns, _rtsp_validation_worker
)

logger = logging.getLogger(__name__)

class ResilientCameraDetector:
    """Camera Detector with timeout-aware network operations"""
    
    def __init__(self):
        self.config = Config()
        self.cameras: Dict[str, Camera] = {}
        self.lock = threading.Lock()
        self.mac_tracker = MACTracker()
        self.frigate_generator = FrigateConfigGenerator(self.config)
        
        # Parse credentials
        self.credentials = self._parse_credentials()
        
        # Service state
        self._running = True
        self._discovery_count = 0
        
        # MQTT queues
        self.mqtt_outgoing = queue.Queue(maxsize=1000)
        self.mqtt_incoming = queue.Queue(maxsize=100)
        
        # MQTT client
        self.mqtt_client = None
        self.mqtt_healthy = False
        
        # Setup
        self._setup_mqtt()
        
        logger.info(f"Resilient Camera Detector initialized: {self.config.SERVICE_ID}")
    
    def _parse_credentials(self) -> List[Tuple[str, str]]:
        """Parse camera credentials from config"""
        credentials = []
        
        # Always include anonymous access
        credentials.append(("", ""))
        
        # Parse configured credentials
        if self.config.CAMERA_CREDENTIALS:
            for cred in self.config.CAMERA_CREDENTIALS.split(','):
                cred = cred.strip()
                if ':' in cred:
                    username, password = cred.split(':', 1)
                    credentials.append((username.strip(), password.strip()))
                else:
                    logger.warning(f"Invalid credential format: {cred}")
        
        logger.info(f"Loaded {len(credentials)} credential sets")
        return credentials
    
    def _setup_mqtt(self):
        """Setup resilient MQTT client"""
        # LWT configuration
        lwt_topic = f"{self.config.TOPIC_HEALTH}/{self.config.NODE_ID}/lwt"
        lwt_payload = json.dumps({
            'node_id': self.config.NODE_ID,
            'service': 'camera_detector',
            'status': 'offline',
            'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        })
        
        lwt_config = {
            'topic': lwt_topic,
            'payload': lwt_payload,
            'qos': 1,
            'retain': True
        }
        
        # TLS configuration if enabled
        tls_config = None
        if self.config.MQTT_TLS:
            tls_config = {'ca_certs': self.config.TLS_CA_PATH}
        
        # Create resilient MQTT client
        self.mqtt_client = ResilientMQTTClient(
            host=self.config.MQTT_BROKER,
            port=8883 if self.config.MQTT_TLS else 1883,
            client_id=self.config.SERVICE_ID,
            outgoing_queue=self.mqtt_outgoing,
            incoming_queue=self.mqtt_incoming,
            keepalive=60,
            socket_timeout=10,
            max_retries=5,
            initial_retry_delay=5,
            max_retry_delay=120,
            tls_config=tls_config,
            lwt_config=lwt_config,
            health_callback=self._mqtt_health_callback
        )
        
        # Start MQTT client
        self.mqtt_client.start()
        
        logger.info("Resilient MQTT client started")
    
    def _mqtt_health_callback(self, state: ConnectionState):
        """Handle MQTT state changes"""
        self.mqtt_healthy = state == ConnectionState.CONNECTED
        
        if state == ConnectionState.CONNECTED:
            logger.info("MQTT connection healthy, publishing initial health report")
            self._publish_health()
        elif state == ConnectionState.FAILED:
            logger.error("MQTT connection FAILED - service running in degraded mode")
            # Could trigger alerts here
    
    def _publish_mqtt(self, topic: str, payload: str, retain: bool = False):
        """Publish MQTT message through resilient client"""
        try:
            # Add retain flag to payload for the worker
            msg = {
                'topic': topic,
                'payload': payload,
                'retain': retain
            }
            self.mqtt_outgoing.put_nowait((topic, json.dumps(msg)))
        except queue.Full:
            logger.error(f"MQTT outgoing queue full, dropping message for {topic}")
    
    def _discover_cameras_onvif(self) -> List[Camera]:
        """Discover cameras using ONVIF with timeout protection"""
        discovered = []
        
        try:
            # Use timeout-aware ONVIF discovery
            devices = ONVIFTimeout.discover_with_timeout(
                timeout=self.config.ONVIF_TIMEOUT,
                max_devices=100
            )
            
            for device in devices:
                ip = device['ip']
                port = device.get('port', 80)
                
                # Create camera object
                camera = Camera(
                    ip=ip,
                    port=port,
                    manufacturer="ONVIF",
                    model="Unknown",
                    name=f"Camera-{ip}"
                )
                
                # Try to get more info with credentials
                for username, password in self.credentials[:3]:  # Try first 3 credentials
                    info = ONVIFTimeout.get_device_info_with_timeout(
                        ip, port, username, password,
                        timeout=self.config.ONVIF_TIMEOUT
                    )
                    if info:
                        camera.manufacturer = info.get('manufacturer', 'ONVIF')
                        camera.model = info.get('model', 'Unknown')
                        camera.username = username
                        camera.password = password
                        break
                
                discovered.append(camera)
                logger.info(f"ONVIF discovered camera at {ip}:{port}")
                
        except Exception as e:
            logger.error(f"ONVIF discovery error: {e}")
            
        return discovered
    
    def _validate_camera_rtsp(self, camera: Camera) -> bool:
        """Validate camera RTSP streams with timeout protection"""
        # Try common RTSP paths
        rtsp_paths = [
            "/stream1", "/live", "/h264", "/rtsp", "/video1",
            "/cam/realmonitor?channel=1&subtype=0",
            "/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp",
            "/live/ch00_0", "/nphMpeg4/g726-640x480", "/MediaInput/h264"
        ]
        
        for username, password in self.credentials:
            for path in rtsp_paths[:6]:  # Test first 6 paths
                rtsp_url = f"rtsp://{camera.ip}:{self.config.RTSP_PORT}{path}"
                if username and password:
                    rtsp_url = f"rtsp://{username}:{password}@{camera.ip}:{self.config.RTSP_PORT}{path}"
                
                # Use timeout-aware validation
                try:
                    if RTSPTimeout.validate_stream_with_timeout(rtsp_url, timeout=self.config.RTSP_TIMEOUT):
                        camera.rtsp_urls['main'] = rtsp_url
                        camera.username = username
                        camera.password = password
                        camera.online = True
                        camera.stream_active = True
                        camera.last_validated = time.time()
                        
                        logger.info(f"RTSP stream validated at {camera.ip} - {path}")
                        return True
                except NetworkTimeoutError:
                    logger.warning(f"RTSP validation timed out for {camera.ip}")
                    continue
        
        return False
    
    def _network_scan(self, network: str) -> List[Camera]:
        """Scan network for cameras with timeout protection"""
        discovered = []
        
        # Common camera ports
        camera_ports = [554, 80, 8080, 8000, 8888]
        
        # Quick TCP scan with timeout
        import ipaddress
        try:
            net = ipaddress.ip_network(network, strict=False)
            for ip in net.hosts():
                ip_str = str(ip)
                
                for port in camera_ports:
                    if NetworkUtils.tcp_port_check(ip_str, port, timeout=1.0):
                        camera = Camera(
                            ip=ip_str,
                            port=port,
                            manufacturer="Unknown",
                            model="Unknown",
                            name=f"Camera-{ip_str}"
                        )
                        discovered.append(camera)
                        logger.debug(f"Found device at {ip_str}:{port}")
                        break
                        
        except Exception as e:
            logger.error(f"Network scan error: {e}")
            
        return discovered
    
    def _update_mac_addresses(self):
        """Update MAC addresses with timeout protection"""
        try:
            # Get local networks
            networks = self._get_local_networks()
            
            for iface, network in networks:
                mac_map = NetworkUtils.arp_scan_with_timeout(
                    iface, network,
                    timeout=30.0
                )
                
                # Update tracker
                for ip, mac in mac_map.items():
                    self.mac_tracker.update(mac, ip)
                    
                    # Update camera MAC if found
                    with self.lock:
                        for camera in self.cameras.values():
                            if camera.ip == ip and not camera.mac:
                                camera.mac = mac
                                camera.id = camera.generate_id()
                                logger.info(f"Updated MAC for {camera.ip}: {mac}")
                                
        except Exception as e:
            logger.error(f"MAC update error: {e}")
    
    def _get_local_networks(self) -> List[Tuple[str, str]]:
        """Get local network interfaces and subnets"""
        networks = []
        
        try:
            import netifaces
            
            for iface in netifaces.interfaces():
                addrs = netifaces.ifaddresses(iface)
                
                # Skip if no IPv4
                if netifaces.AF_INET not in addrs:
                    continue
                
                for addr in addrs[netifaces.AF_INET]:
                    ip = addr.get('addr')
                    netmask = addr.get('netmask')
                    
                    if ip and netmask and not ip.startswith('127.'):
                        # Calculate network address
                        import ipaddress
                        try:
                            interface = ipaddress.ip_interface(f"{ip}/{netmask}")
                            network = str(interface.network)
                            networks.append((iface, network))
                        except Exception:
                            pass
                            
        except Exception as e:
            logger.error(f"Failed to get local networks: {e}")
            
        return networks
    
    def _publish_health(self):
        """Publish health report"""
        try:
            health = {
                'node_id': self.config.NODE_ID,
                'service': 'camera_detector',
                'status': 'online',
                'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                'cameras': {
                    'total': len(self.cameras),
                    'online': sum(1 for c in self.cameras.values() if c.online),
                    'streams_active': sum(1 for c in self.cameras.values() if c.stream_active)
                },
                'mqtt': {
                    'connected': self.mqtt_healthy,
                    'stats': self.mqtt_client.get_stats() if self.mqtt_client else {}
                },
                'discovery_count': self._discovery_count
            }
            
            self._publish_mqtt(
                self.config.TOPIC_HEALTH,
                json.dumps(health),
                retain=True
            )
        except Exception as e:
            logger.error(f"Failed to publish health: {e}")
    
    def run(self):
        """Main service loop"""
        logger.info("Starting Camera Detector main loop")
        
        # Start background tasks
        threading.Thread(target=self._discovery_loop, daemon=True).start()
        threading.Thread(target=self._health_check_loop, daemon=True).start()
        
        if self.config.MAC_TRACKING_ENABLED:
            threading.Thread(target=self._mac_tracking_loop, daemon=True).start()
        
        # Health reporting timer
        def periodic_health():
            while self._running:
                self._publish_health()
                time.sleep(60)
        
        threading.Thread(target=periodic_health, daemon=True).start()
        
        # Main loop - just keep service alive
        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down Camera Detector")
            self.shutdown()
    
    def _discovery_loop(self):
        """Camera discovery loop"""
        while self._running:
            try:
                self._discovery_count += 1
                logger.info(f"Starting discovery cycle {self._discovery_count}")
                
                discovered_cameras = []
                
                # ONVIF discovery
                discovered_cameras.extend(self._discover_cameras_onvif())
                
                # Network scan (if not too many cameras already)
                if len(discovered_cameras) < 50:
                    for _, network in self._get_local_networks()[:2]:  # Limit networks
                        discovered_cameras.extend(self._network_scan(network))
                
                # Process discovered cameras
                for camera in discovered_cameras:
                    with self.lock:
                        # Check if already known
                        existing = None
                        for known in self.cameras.values():
                            if known.ip == camera.ip:
                                existing = known
                                break
                        
                        if not existing:
                            # Validate RTSP
                            if self._validate_camera_rtsp(camera):
                                # Get MAC address
                                mac = self.mac_tracker.get_mac_for_ip(camera.ip)
                                if mac:
                                    camera.mac = mac
                                    camera.id = camera.generate_id()
                                
                                # Add to known cameras
                                self.cameras[camera.id] = camera
                                logger.info(f"Added new camera: {camera.id} at {camera.ip}")
                                
                                # Publish discovery
                                self._publish_camera_discovery(camera)
                
                # Update Frigate config
                self._update_frigate_config()
                
                # Sleep until next cycle
                interval = self.config.DISCOVERY_INTERVAL
                if self.config.SMART_DISCOVERY_ENABLED and self._discovery_count > 3:
                    # Reduce frequency after initial discovery
                    interval *= 2
                
                logger.info(f"Discovery cycle complete, sleeping {interval}s")
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Discovery loop error: {e}", exc_info=True)
                time.sleep(30)
    
    def _health_check_loop(self):
        """Monitor camera health"""
        while self._running:
            try:
                with self.lock:
                    for camera in list(self.cameras.values()):
                        # Check if offline
                        if time.time() - camera.last_seen > self.config.OFFLINE_THRESHOLD:
                            if camera.online:
                                camera.online = False
                                camera.stream_active = False
                                self._publish_camera_status(camera, "offline")
                                logger.warning(f"Camera {camera.id} is offline")
                
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                time.sleep(30)
    
    def _mac_tracking_loop(self):
        """Update MAC addresses periodically"""
        while self._running:
            try:
                self._update_mac_addresses()
                time.sleep(300)  # Every 5 minutes
            except Exception as e:
                logger.error(f"MAC tracking error: {e}")
                time.sleep(300)
    
    def _publish_camera_discovery(self, camera: Camera):
        """Publish camera discovery event"""
        topic = f"{self.config.TOPIC_DISCOVERY}/{camera.id}"
        self._publish_mqtt(topic, camera.to_json(), retain=True)
    
    def _publish_camera_status(self, camera: Camera, status: str):
        """Publish camera status update"""
        topic = f"{self.config.TOPIC_STATUS}/{camera.id}"
        payload = json.dumps({
            'camera_id': camera.id,
            'ip': camera.ip,
            'status': status,
            'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        })
        self._publish_mqtt(topic, payload, retain=False)
    
    def _update_frigate_config(self):
        """Generate and publish Frigate configuration"""
        with self.lock:
            config = self.frigate_generator.generate_config(list(self.cameras.values()))
            
        if config:
            # Publish config
            self._publish_mqtt(
                self.config.TOPIC_FRIGATE_CONFIG,
                json.dumps(config),
                retain=True
            )
            
            # Trigger reload
            self._publish_mqtt(
                self.config.TOPIC_FRIGATE_RELOAD,
                json.dumps({'reload': True}),
                retain=False
            )
            
            logger.info("Published updated Frigate configuration")
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down Camera Detector")
        self._running = False
        
        # Stop MQTT client
        if self.mqtt_client:
            self.mqtt_client.stop(timeout=5.0)
        
        logger.info("Camera Detector shutdown complete")

def main():
    """Main entry point"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    detector = ResilientCameraDetector()
    detector.run()

if __name__ == "__main__":
    main()