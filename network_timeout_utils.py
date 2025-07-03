#!/usr/bin/env python3.12
"""Network Timeout Utilities for Safety-Critical Systems

This module provides timeout-aware network operations to prevent hanging
in the Wildfire Watch system. All operations have explicit timeouts and
proper error handling.

Key Features:
    - Socket operations with timeouts
    - ONVIF discovery with timeout protection  
    - Subprocess execution with timeout
    - Network scanning with timeout limits
"""

import socket
import subprocess
import time
import logging
from typing import Optional, List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

class NetworkTimeoutError(Exception):
    """Raised when a network operation times out"""
    pass

class NetworkUtils:
    """Utilities for timeout-aware network operations"""
    
    @staticmethod
    def tcp_port_check(host: str, port: int, timeout: float = 5.0) -> bool:
        """Check if a TCP port is open with timeout.
        
        Args:
            host: Target hostname or IP
            port: Port number to check
            timeout: Connection timeout in seconds
            
        Returns:
            bool: True if port is open, False otherwise
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        
        try:
            result = sock.connect_ex((host, port))
            return result == 0
        except socket.timeout:
            logger.debug(f"TCP check timed out for {host}:{port}")
            return False
        except Exception as e:
            logger.debug(f"TCP check failed for {host}:{port}: {e}")
            return False
        finally:
            sock.close()
    
    @staticmethod
    def udp_discover(broadcast_addr: str, port: int, probe_data: bytes,
                    timeout: float = 5.0, max_responses: int = 100) -> List[Tuple[str, bytes]]:
        """Send UDP broadcast and collect responses with timeout.
        
        Args:
            broadcast_addr: Broadcast address (e.g., "192.168.1.255")
            port: UDP port
            probe_data: Data to send as probe
            timeout: Total timeout for discovery
            max_responses: Maximum responses to collect
            
        Returns:
            List of (ip_address, response_data) tuples
        """
        responses = []
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(0.1)  # Short timeout for non-blocking receives
        
        try:
            # Send broadcast
            sock.sendto(probe_data, (broadcast_addr, port))
            logger.debug(f"Sent UDP broadcast to {broadcast_addr}:{port}")
            
            # Collect responses
            end_time = time.time() + timeout
            while time.time() < end_time and len(responses) < max_responses:
                try:
                    data, addr = sock.recvfrom(4096)
                    responses.append((addr[0], data))
                    logger.debug(f"UDP response from {addr[0]}")
                except socket.timeout:
                    # No response in this interval, continue
                    continue
                except Exception as e:
                    logger.debug(f"UDP receive error: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"UDP discovery error: {e}")
        finally:
            sock.close()
            
        return responses
    
    @staticmethod
    def execute_with_timeout(cmd: List[str], timeout: float = 30.0,
                           capture_output: bool = True) -> subprocess.CompletedProcess:
        """Execute subprocess with timeout protection.
        
        Args:
            cmd: Command and arguments as list
            timeout: Execution timeout in seconds
            capture_output: Whether to capture stdout/stderr
            
        Returns:
            CompletedProcess object
            
        Raises:
            NetworkTimeoutError: If command times out
            subprocess.CalledProcessError: If command fails
        """
        try:
            result = subprocess.run(
                cmd,
                timeout=timeout,
                capture_output=capture_output,
                text=True,
                check=True
            )
            return result
        except subprocess.TimeoutExpired as e:
            logger.error(f"Command timed out after {timeout}s: {' '.join(cmd)}")
            raise NetworkTimeoutError(f"Command timed out: {cmd[0]}") from e
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {' '.join(cmd)} - {e.stderr}")
            raise
    
    @staticmethod
    def arp_scan_with_timeout(interface: str, network: str, 
                            timeout: float = 30.0) -> Dict[str, str]:
        """Perform ARP scan with timeout protection.
        
        Args:
            interface: Network interface to scan on
            network: Network in CIDR format (e.g., "192.168.1.0/24")
            timeout: Scan timeout in seconds
            
        Returns:
            Dict mapping IP addresses to MAC addresses
        """
        mac_map = {}
        
        try:
            # Use arp-scan if available (requires NET_ADMIN capability)
            cmd = ["arp-scan", "-I", interface, "-l", "-t", str(int(timeout * 1000))]
            result = NetworkUtils.execute_with_timeout(cmd, timeout=timeout)
            
            # Parse output
            for line in result.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 2 and ':' in parts[1]:
                    ip, mac = parts[0], parts[1]
                    mac_map[ip] = mac.upper()
                    
        except (NetworkTimeoutError, subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to reading ARP cache
            logger.debug("arp-scan failed, reading ARP cache")
            try:
                with open('/proc/net/arp', 'r') as f:
                    lines = f.readlines()[1:]  # Skip header
                    for line in lines:
                        parts = line.split()
                        if len(parts) >= 4 and parts[2] != "0x0":
                            ip, mac = parts[0], parts[3]
                            if mac != "00:00:00:00:00:00":
                                mac_map[ip] = mac.upper()
            except Exception as e:
                logger.debug(f"Failed to read ARP cache: {e}")
                
        return mac_map

class ONVIFTimeout:
    """ONVIF operations with timeout protection"""
    
    @staticmethod
    def discover_with_timeout(timeout: float = 10.0, 
                            max_devices: int = 100) -> List[Dict[str, Any]]:
        """Discover ONVIF devices with timeout protection.
        
        Uses WS-Discovery protocol with proper timeout handling.
        
        Args:
            timeout: Discovery timeout in seconds
            max_devices: Maximum devices to discover
            
        Returns:
            List of device info dicts with 'ip', 'port', 'services' keys
        """
        devices = []
        
        # WS-Discovery probe message
        probe_msg = '''<?xml version="1.0" encoding="UTF-8"?>
        <s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" 
                    xmlns:a="http://schemas.xmlsoap.org/ws/2004/08/addressing">
            <s:Header>
                <a:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</a:Action>
                <a:MessageID>uuid:12345678-1234-1234-1234-123456789012</a:MessageID>
                <a:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</a:To>
            </s:Header>
            <s:Body>
                <d:Probe xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery">
                    <d:Types>dn:NetworkVideoTransmitter</d:Types>
                </d:Probe>
            </s:Body>
        </s:Envelope>'''
        
        # Send to WS-Discovery multicast address
        responses = NetworkUtils.udp_discover(
            "239.255.255.250",
            3702,
            probe_msg.encode('utf-8'),
            timeout=timeout,
            max_responses=max_devices
        )
        
        # Parse responses
        for ip, data in responses:
            try:
                # Parse XML response
                root = ET.fromstring(data.decode('utf-8', errors='ignore'))
                
                # Extract XAddrs (service endpoints)
                for elem in root.iter():
                    if 'XAddrs' in elem.tag:
                        addrs = elem.text
                        if addrs:
                            # Parse first address
                            addr = addrs.split()[0]
                            if addr.startswith('http'):
                                # Extract port from URL
                                import urllib.parse
                                parsed = urllib.parse.urlparse(addr)
                                devices.append({
                                    'ip': ip,
                                    'port': parsed.port or 80,
                                    'services': addrs.split()
                                })
                                break
                                
            except Exception as e:
                logger.debug(f"Failed to parse ONVIF response from {ip}: {e}")
                
        return devices
    
    @staticmethod
    def get_device_info_with_timeout(host: str, port: int, 
                                   username: str, password: str,
                                   timeout: float = 10.0) -> Optional[Dict[str, str]]:
        """Get ONVIF device info with timeout protection.
        
        Args:
            host: Device IP address
            port: ONVIF port
            username: Device username
            password: Device password  
            timeout: Operation timeout
            
        Returns:
            Device info dict or None if failed
        """
        # This would normally use python-onvif-zeep, but we'll use
        # a simpler HTTP request approach for timeout control
        
        import requests
        from requests.auth import HTTPDigestAuth
        
        device_info_body = '''<?xml version="1.0" encoding="UTF-8"?>
        <s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">
            <s:Body>
                <tds:GetDeviceInformation xmlns:tds="http://www.onvif.org/ver10/device/wsdl"/>
            </s:Body>
        </s:Envelope>'''
        
        url = f"http://{host}:{port}/onvif/device_service"
        
        try:
            response = requests.post(
                url,
                data=device_info_body,
                headers={'Content-Type': 'application/soap+xml'},
                auth=HTTPDigestAuth(username, password),
                timeout=timeout
            )
            
            if response.status_code == 200:
                # Parse response (simplified)
                return {
                    'status': 'success',
                    'manufacturer': 'Unknown',
                    'model': 'Unknown'
                }
            else:
                logger.debug(f"ONVIF request failed with status {response.status_code}")
                return None
                
        except requests.Timeout:
            logger.debug(f"ONVIF request timed out for {host}:{port}")
            return None
        except Exception as e:
            logger.debug(f"ONVIF request failed for {host}:{port}: {e}")
            return None

class RTSPTimeout:
    """RTSP operations with timeout protection"""
    
    @staticmethod 
    def validate_stream_with_timeout(rtsp_url: str, timeout: float = 10.0) -> bool:
        """Validate RTSP stream with timeout protection.
        
        Uses a subprocess to avoid OpenCV hanging issues.
        
        Args:
            rtsp_url: RTSP URL to validate
            timeout: Validation timeout in seconds
            
        Returns:
            bool: True if stream is valid
        """
        # Use ffprobe for reliable timeout handling
        cmd = [
            "ffprobe",
            "-v", "error",
            "-rtsp_transport", "tcp",
            "-timeout", str(int(timeout * 1000000)),  # microseconds
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_type",
            "-of", "default=noprint_wrappers=1:nokey=1",
            rtsp_url
        ]
        
        try:
            result = NetworkUtils.execute_with_timeout(cmd, timeout=timeout)
            # Check if video stream was detected
            return "video" in result.stdout.lower()
        except (NetworkTimeoutError, subprocess.CalledProcessError):
            return False
        except FileNotFoundError:
            # ffprobe not available, fall back to False
            logger.warning("ffprobe not found, cannot validate RTSP streams")
            return False