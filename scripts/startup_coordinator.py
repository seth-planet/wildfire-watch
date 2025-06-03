#!/usr/bin/env python3
"""
Startup coordinator for Wildfire Watch services
Ensures proper timing and dependency management
"""
import os
import sys
import time
import socket
import subprocess
import paho.mqtt.client as mqtt
from typing import Dict, List, Optional

class StartupCoordinator:
    def __init__(self):
        self.services = {
            'mqtt_broker': {
                'check': self._check_mqtt,
                'wait': 5,
                'timeout': 30
            },
            'camera_detector': {
                'check': self._check_http_health,
                'port': 8080,
                'wait': 10,
                'timeout': 60
            },
            'security_nvr': {
                'check': self._check_http_health,
                'port': 5000,
                'wait': 15,
                'timeout': 120
            },
            'fire_consensus': {
                'check': self._check_mqtt_topic,
                'topic': 'system/consensus_telemetry',
                'wait': 5,
                'timeout': 60
            }
        }
        
    def _check_mqtt(self, service: str) -> bool:
        """Check if MQTT broker is accessible"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(('mqtt_broker', 1883))
            sock.close()
            return result == 0
        except:
            return False
    
    def _check_http_health(self, service: str) -> bool:
        """Check HTTP health endpoint"""
        port = self.services[service].get('port', 80)
        try:
            import requests
            response = requests.get(f'http://{service}:{port}/health', timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _check_mqtt_topic(self, service: str) -> bool:
        """Check if service is publishing to MQTT"""
        topic = self.services[service].get('topic')
        if not topic:
            return False
            
        try:
            received = False
            
            def on_message(client, userdata, msg):
                nonlocal received
                received = True
            
            client = mqtt.Client()
            client.on_message = on_message
            client.connect("mqtt_broker", 1883, 60)
            client.subscribe(topic)
            client.loop_start()
            
            # Wait up to 5 seconds for a message
            for _ in range(50):
                if received:
                    break
                time.sleep(0.1)
            
            client.loop_stop()
            client.disconnect()
            return received
        except:
            return False
    
    def wait_for_service(self, service: str) -> bool:
        """Wait for a service to be ready"""
        config = self.services.get(service, {})
        check_func = config.get('check')
        timeout = config.get('timeout', 60)
        wait_after = config.get('wait', 0)
        
        if not check_func:
            return True
        
        print(f"Waiting for {service}...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if check_func(service):
                print(f"✓ {service} is ready")
                if wait_after > 0:
                    print(f"  Waiting {wait_after}s for stabilization...")
                    time.sleep(wait_after)
                return True
            time.sleep(1)
        
        print(f"✗ {service} failed to start within {timeout}s")
        return False
    
    def coordinate_startup(self) -> bool:
        """Coordinate service startup"""
        print("Wildfire Watch Startup Coordinator")
        print("=" * 40)
        
        # Start services in order
        startup_order = [
            'mqtt_broker',
            'camera_detector',
            'security_nvr',
            'fire_consensus'
        ]
        
        for service in startup_order:
            if not self.wait_for_service(service):
                return False
        
        print("\n✓ All services started successfully!")
        return True

def main():
    # Wait for startup delay if specified
    startup_delay = int(os.environ.get('STARTUP_DELAY', '0'))
    if startup_delay > 0:
        print(f"Waiting {startup_delay}s before startup...")
        time.sleep(startup_delay)
    
    coordinator = StartupCoordinator()
    success = coordinator.coordinate_startup()
    
    if not success:
        print("\nStartup failed! Check service logs.")
        sys.exit(1)
    
    # Keep running if specified
    if os.environ.get('KEEP_RUNNING', 'false').lower() == 'true':
        print("\nStartup coordinator running...")
        while True:
            time.sleep(60)

if __name__ == '__main__':
    main()
