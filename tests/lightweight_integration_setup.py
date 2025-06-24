#!/usr/bin/env python3.12
"""
Lightweight integration test setup using docker-compose
More reliable than manual container management
"""
import os
import sys
import time
import subprocess
import tempfile
import yaml
from pathlib import Path
import docker
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class LightweightIntegrationSetup:
    """Manages integration test environment using docker-compose"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.docker_client = docker.from_env()
        self.compose_file = None
        self.test_dir = None
        
    def create_test_compose_file(self):
        """Create a minimal docker-compose file for testing"""
        self.test_dir = tempfile.mkdtemp(prefix="wildfire_test_")
        
        compose_content = {
            'version': '3.8',
            'services': {
                'mqtt-broker': {
                    'image': 'eclipse-mosquitto:2.0',
                    'container_name': 'mqtt-broker-test',
                    'ports': ['18833:1883'],
                    'command': 'mosquitto -c /mosquitto-no-auth.conf',
                    'healthcheck': {
                        'test': ['CMD', 'mosquitto_sub', '-t', '$SYS/#', '-C', '1'],
                        'interval': '5s',
                        'timeout': '3s',
                        'retries': 5,
                        'start_period': '10s'
                    },
                    'networks': ['test_net']
                },
                'camera-detector': {
                    'build': {
                        'context': str(self.project_root / 'camera_detector'),
                        'dockerfile': 'Dockerfile'
                    },
                    'container_name': 'camera-detector-test',
                    'depends_on': {
                        'mqtt-broker': {'condition': 'service_healthy'}
                    },
                    'environment': {
                        'MQTT_BROKER': 'mqtt-broker',
                        'MQTT_PORT': '1883',
                        'LOG_LEVEL': 'DEBUG',
                        'CAMERA_CREDENTIALS': 'admin:admin,admin:',
                        'DISCOVERY_INTERVAL': '30'
                    },
                    'networks': ['test_net']
                },
                'fire-consensus': {
                    'build': {
                        'context': str(self.project_root / 'fire_consensus'),
                        'dockerfile': 'Dockerfile.test'
                    },
                    'container_name': 'fire-consensus-test',
                    'depends_on': {
                        'mqtt-broker': {'condition': 'service_healthy'}
                    },
                    'environment': {
                        'MQTT_BROKER': 'mqtt-broker',
                        'MQTT_PORT': '1883',
                        'LOG_LEVEL': 'DEBUG',
                        'CONSENSUS_THRESHOLD': '2',
                        'MIN_CONFIDENCE': '0.7',
                        'NODE_ID': 'test-node'
                    },
                    'healthcheck': {
                        'test': ['CMD', 'python', '-c', 
                                "import socket; s=socket.socket(); s.settimeout(5); s.connect(('mqtt-broker', 1883)); s.close()"],
                        'interval': '10s',
                        'timeout': '5s',
                        'retries': 3,
                        'start_period': '20s'
                    },
                    'networks': ['test_net']
                },
                'gpio-trigger': {
                    'build': {
                        'context': str(self.project_root / 'gpio_trigger'),
                        'dockerfile': 'Dockerfile',
                        'args': {
                            'BUILD_ENV': 'test'
                        }
                    },
                    'container_name': 'gpio-trigger-test',
                    'depends_on': {
                        'mqtt-broker': {'condition': 'service_healthy'}
                    },
                    'environment': {
                        'MQTT_BROKER': 'mqtt-broker',
                        'MQTT_PORT': '1883',
                        'LOG_LEVEL': 'DEBUG',
                        'GPIO_SIMULATION': 'true',
                        'SIMULATION_MODE_WARNINGS': 'false',
                        'NODE_ID': 'test-node'
                    },
                    'networks': ['test_net']
                }
            },
            'networks': {
                'test_net': {
                    'driver': 'bridge',
                    'ipam': {
                        'config': [{'subnet': '192.168.200.0/24'}]
                    }
                }
            }
        }
        
        self.compose_file = Path(self.test_dir) / 'docker-compose.test.yml'
        with open(self.compose_file, 'w') as f:
            yaml.dump(compose_content, f, default_flow_style=False)
            
        logger.info(f"Created test compose file: {self.compose_file}")
        return self.compose_file
        
    def setup_all_services(self):
        """Start all services using docker-compose"""
        if not self.compose_file:
            self.create_test_compose_file()
            
        try:
            # First, clean up any existing containers
            logger.info("Cleaning up any existing test containers...")
            subprocess.run(
                ['docker-compose', '-f', str(self.compose_file), 'down', '-v', '--remove-orphans'],
                capture_output=True,
                text=True,
                check=False  # Don't fail if nothing to clean
            )
            
            # Build images
            logger.info("Building Docker images...")
            result = subprocess.run(
                ['docker-compose', '-f', str(self.compose_file), 'build'],
                capture_output=True,
                text=True,
                check=True
            )
            logger.debug(f"Build output: {result.stdout}")
            
            # Start services
            logger.info("Starting services...")
            result = subprocess.run(
                ['docker-compose', '-f', str(self.compose_file), 'up', '-d'],
                capture_output=True,
                text=True,
                check=True
            )
            logger.debug(f"Up output: {result.stdout}")
            
            # Wait for services to be healthy
            logger.info("Waiting for services to be healthy...")
            max_wait = 60
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                # Check container states
                containers_ready = True
                
                for container_name in ['mqtt-broker-test', 'camera-detector-test', 
                                      'fire-consensus-test', 'gpio-trigger-test']:
                    try:
                        container = self.docker_client.containers.get(container_name)
                        if container.status != 'running':
                            containers_ready = False
                            logger.debug(f"{container_name} status: {container.status}")
                            
                            # If exited, show logs
                            if container.status == 'exited':
                                logs = container.logs(tail=20).decode('utf-8')
                                logger.error(f"{container_name} exited. Last logs:\n{logs}")
                                
                    except docker.errors.NotFound:
                        containers_ready = False
                        logger.debug(f"{container_name} not found yet")
                        
                if containers_ready:
                    logger.info("All containers are running!")
                    # Give them a bit more time to fully initialize
                    time.sleep(5)
                    return True
                    
                time.sleep(2)
                
            raise Exception("Timeout waiting for services to be ready")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Docker-compose command failed: {e.stderr}")
            raise
        except Exception as e:
            logger.error(f"Failed to setup services: {e}")
            self.cleanup()
            raise
            
    def cleanup(self):
        """Clean up test environment"""
        if self.compose_file and self.compose_file.exists():
            logger.info("Cleaning up test environment...")
            
            try:
                # Stop and remove containers
                result = subprocess.run(
                    ['docker-compose', '-f', str(self.compose_file), 'down', '-v', '--remove-orphans'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode != 0:
                    logger.warning(f"Cleanup had issues: {result.stderr}")
                else:
                    logger.info("Docker-compose cleanup completed")
                    
            except subprocess.TimeoutExpired:
                logger.warning("Cleanup timed out, forcing removal...")
                # Force remove containers
                for container_name in ['mqtt-broker-test', 'camera-detector-test', 
                                     'fire-consensus-test', 'gpio-trigger-test']:
                    try:
                        container = self.docker_client.containers.get(container_name)
                        container.remove(force=True)
                    except:
                        pass
                        
            # Clean up temp directory
            if self.test_dir:
                import shutil
                try:
                    shutil.rmtree(self.test_dir)
                except:
                    pass
                    
    def is_mqtt_ready(self, host="localhost", port=18833):
        """Check if MQTT broker is ready"""
        try:
            import paho.mqtt.client as mqtt
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "test_ready_check")
            client.connect(host, port, 5)
            client.disconnect()
            return True
        except Exception:
            return False


if __name__ == "__main__":
    setup = LightweightIntegrationSetup()
    
    if len(sys.argv) > 1 and sys.argv[1] == "cleanup":
        setup.cleanup()
    else:
        try:
            setup.setup_all_services()
            print("\n✓ All services ready for integration testing")
            print("\nPress Ctrl+C to stop...")
            
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nShutting down...")
        except Exception as e:
            print(f"Setup failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            setup.cleanup()