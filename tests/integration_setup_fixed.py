#!/usr/bin/env python3.12
"""
Integration test setup utilities with improved container management
Creates required Docker containers and services for testing
"""
import os
import sys
import time
import docker
import tempfile
import subprocess
import logging
from pathlib import Path
from typing import Dict, Optional

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class IntegrationTestSetup:
    """Manages Docker containers and services for integration testing with improved error handling"""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.temp_dir = None
        self.containers = {}
        self.networks = {}
        self.cleanup_on_exit = True
        
    def setup_network(self):
        """Create test network with proper cleanup"""
        network_name = "wildfire_test_net"
        
        # Try to get existing network
        try:
            network = self.docker_client.networks.get(network_name)
            logger.info(f"Using existing network: {network_name}")
        except docker.errors.NotFound:
            # Create new network
            network = self.docker_client.networks.create(
                network_name,
                driver="bridge",
                ipam=docker.types.IPAMConfig(
                    pool_configs=[
                        docker.types.IPAMPool(subnet="192.168.200.0/24")
                    ]
                )
            )
            logger.info(f"Created new network: {network_name}")
            
        self.networks["test_net"] = network
        return network
        
    def wait_for_container_healthy(self, container_name: str, timeout: int = 60) -> bool:
        """Wait for container to be healthy"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                container = self.docker_client.containers.get(container_name)
                if container.status == "running":
                    # Check if container has health check
                    health = container.attrs.get("State", {}).get("Health", {})
                    if health:
                        health_status = health.get("Status", "none")
                        if health_status == "healthy":
                            logger.info(f"Container {container_name} is healthy")
                            return True
                        elif health_status == "unhealthy":
                            logger.error(f"Container {container_name} is unhealthy")
                            # Get last health check log
                            logs = health.get("Log", [])
                            if logs:
                                logger.error(f"Last health check: {logs[-1]}")
                            return False
                    else:
                        # No health check, just check if running
                        logger.info(f"Container {container_name} is running (no health check)")
                        return True
                elif container.status == "exited":
                    logger.error(f"Container {container_name} exited")
                    # Get container logs
                    logs = container.logs(tail=50).decode('utf-8')
                    logger.error(f"Container logs:\\n{logs}")
                    return False
            except docker.errors.NotFound:
                logger.debug(f"Container {container_name} not found yet")
            
            time.sleep(1)
            
        logger.error(f"Timeout waiting for container {container_name}")
        return False
        
    def cleanup_existing_container(self, container_name: str):
        """Clean up existing container safely"""
        try:
            container = self.docker_client.containers.get(container_name)
            
            # Check container state
            if container.status == "running":
                logger.info(f"Stopping running container: {container_name}")
                container.stop(timeout=5)
                
            # Wait for container to stop
            time.sleep(1)
            
            # Remove container
            try:
                container.remove(force=True)
                logger.info(f"Removed container: {container_name}")
            except docker.errors.APIError as e:
                if "removal" in str(e) and "already in progress" in str(e):
                    logger.warning(f"Container {container_name} already being removed")
                    # Wait for removal to complete
                    time.sleep(2)
                else:
                    raise
                    
        except docker.errors.NotFound:
            logger.debug(f"Container {container_name} not found, nothing to cleanup")
            
    def setup_mqtt_broker(self):
        """Create and start MQTT broker container"""
        network = self.setup_network()
        container_name = "mqtt-broker-test"
        
        # Clean up existing container
        self.cleanup_existing_container(container_name)
        
        try:
            # Create simple mosquitto config
            mosquitto_config = """
listener 1883
allow_anonymous true
log_type all
"""
            
            # Create MQTT broker container
            container = self.docker_client.containers.run(
                "eclipse-mosquitto:2.0",
                name=container_name,
                ports={'1883/tcp': 18833},
                network="wildfire_test_net",
                detach=True,
                remove=False,  # Don't auto-remove
                command="sh -c 'echo \"$MOSQUITTO_CONFIG\" > /mosquitto/config/mosquitto.conf && mosquitto -c /mosquitto/config/mosquitto.conf'",
                environment={
                    "MOSQUITTO_CONFIG": mosquitto_config
                },
                healthcheck={
                    "test": ["CMD", "mosquitto_sub", "-t", "$SYS/#", "-C", "1"],
                    "interval": 5000000000,  # 5s in nanoseconds
                    "timeout": 3000000000,   # 3s in nanoseconds
                    "retries": 5,
                    "start_period": 10000000000  # 10s in nanoseconds
                }
            )
            
            self.containers["mqtt-broker"] = container
            logger.info(f"Created MQTT broker: {container.name}")
            
            # Wait for broker to be ready
            if self.wait_for_container_healthy(container_name, timeout=30):
                logger.info("MQTT broker is ready")
                return container
            else:
                raise Exception("MQTT broker failed to become healthy")
                
        except Exception as e:
            logger.error(f"Failed to setup MQTT broker: {e}")
            raise
            
    def setup_camera_detector(self):
        """Create camera detector container"""
        mqtt_container = self.setup_mqtt_broker()
        container_name = "camera-detector-test"
        
        # Clean up existing container
        self.cleanup_existing_container(container_name)
        
        # Build camera detector if needed
        try:
            image = self.docker_client.images.get("wildfire-watch/camera_detector:test")
        except docker.errors.ImageNotFound:
            logger.info("Building camera detector image...")
            project_root = Path(__file__).parent.parent
            image, logs = self.docker_client.images.build(
                path=str(project_root / "camera_detector"),
                tag="wildfire-watch/camera_detector:test",
                rm=True,
                buildargs={
                    "PLATFORM": "linux/amd64"
                }
            )
            
        try:
            container = self.docker_client.containers.run(
                "wildfire-watch/camera_detector:test",
                name=container_name,
                network="wildfire_test_net",
                environment={
                    "MQTT_BROKER": "mqtt-broker-test",
                    "MQTT_PORT": "1883",
                    "LOG_LEVEL": "DEBUG",
                    "CAMERA_CREDENTIALS": "username:password,admin:",
                    "DISCOVERY_INTERVAL": "30"
                },
                detach=True,
                remove=False  # Don't auto-remove
            )
            
            self.containers["camera-detector"] = container
            logger.info(f"Created camera detector: {container.name}")
            
            # Give it time to start
            time.sleep(3)
            return container
            
        except Exception as e:
            logger.error(f"Failed to setup camera detector: {e}")
            raise
            
    def setup_fire_consensus(self):
        """Create fire consensus container with fixed startup"""
        mqtt_container = self.setup_mqtt_broker()
        container_name = "fire-consensus-test"
        
        # Clean up existing container
        self.cleanup_existing_container(container_name)
        
        try:
            image = self.docker_client.images.get("wildfire-watch/fire_consensus:test")
        except docker.errors.ImageNotFound:
            logger.info("Building fire consensus image...")
            project_root = Path(__file__).parent.parent
            image, logs = self.docker_client.images.build(
                path=str(project_root / "fire_consensus"),
                tag="wildfire-watch/fire_consensus:test",
                rm=True,
                buildargs={
                    "PLATFORM": "linux/amd64"
                }
            )
            
        try:
            # Create a simple entrypoint that doesn't require D-Bus/Avahi for testing
            test_entrypoint = """#!/bin/sh
echo "Starting Fire Consensus Service (Test Mode)"
echo "MQTT Broker: ${MQTT_BROKER}"
echo "Node ID: ${NODE_ID}"
exec python -u consensus.py
"""
            
            container = self.docker_client.containers.run(
                "wildfire-watch/fire_consensus:test",
                name=container_name,
                network="wildfire_test_net",
                environment={
                    "MQTT_BROKER": "mqtt-broker-test",
                    "MQTT_PORT": "1883",
                    "LOG_LEVEL": "DEBUG",
                    "CONSENSUS_THRESHOLD": "2",
                    "MIN_CONFIDENCE": "0.7",
                    "NODE_ID": "test-node"
                },
                detach=True,
                remove=False  # Don't auto-remove
            )
            
            self.containers["fire-consensus"] = container
            logger.info(f"Created fire consensus: {container.name}")
            
            # Wait for it to be healthy
            if self.wait_for_container_healthy(container_name, timeout=30):
                logger.info("Fire consensus is ready")
                return container
            else:
                # Get logs for debugging
                logs = container.logs(tail=100).decode('utf-8')
                logger.error(f"Fire consensus logs:\\n{logs}")
                raise Exception("Fire consensus failed to become healthy")
                
        except Exception as e:
            logger.error(f"Failed to setup fire consensus: {e}")
            raise
            
    def setup_gpio_trigger(self):
        """Create GPIO trigger container"""
        mqtt_container = self.setup_mqtt_broker()
        container_name = "gpio-trigger-test"
        
        # Clean up existing container
        self.cleanup_existing_container(container_name)
        
        try:
            image = self.docker_client.images.get("wildfire-watch/gpio_trigger:test")
        except docker.errors.ImageNotFound:
            logger.info("Building GPIO trigger image...")
            project_root = Path(__file__).parent.parent
            image, logs = self.docker_client.images.build(
                path=str(project_root / "gpio_trigger"),
                tag="wildfire-watch/gpio_trigger:test",
                rm=True,
                buildargs={
                    "PLATFORM": "linux/amd64",
                    "BUILD_ENV": "test"
                }
            )
            
        try:
            container = self.docker_client.containers.run(
                "wildfire-watch/gpio_trigger:test",
                name=container_name,
                network="wildfire_test_net",
                environment={
                    "MQTT_BROKER": "mqtt-broker-test",
                    "MQTT_PORT": "1883",
                    "LOG_LEVEL": "DEBUG",
                    "GPIO_SIMULATION": "true",
                    "SIMULATION_MODE_WARNINGS": "false",
                    "NODE_ID": "test-node"
                },
                detach=True,
                remove=False,  # Don't auto-remove
                healthcheck={
                    "test": ["CMD", "python3.12", "-c", "import socket; s=socket.socket(); s.settimeout(5); s.connect(('mqtt-broker-test', 1883)); s.close()"],
                    "interval": 5000000000,  # 5s in nanoseconds
                    "timeout": 3000000000,   # 3s in nanoseconds
                    "retries": 5,
                    "start_period": 10000000000  # 10s in nanoseconds
                }
            )
            
            self.containers["gpio-trigger"] = container
            logger.info(f"Created GPIO trigger: {container.name}")
            
            # Give it time to start
            time.sleep(3)
            return container
            
        except Exception as e:
            logger.error(f"Failed to setup GPIO trigger: {e}")
            raise
            
    def setup_all_services(self):
        """Setup all required services for integration testing"""
        logger.info("Setting up integration test environment...")
        
        try:
            mqtt = self.setup_mqtt_broker()
            logger.info(f"✓ MQTT broker: {mqtt.name}")
            
            camera = self.setup_camera_detector()
            logger.info(f"✓ Camera detector: {camera.name}")
            
            consensus = self.setup_fire_consensus()
            logger.info(f"✓ Fire consensus: {consensus.name}")
            
            trigger = self.setup_gpio_trigger()
            logger.info(f"✓ GPIO trigger: {trigger.name}")
            
            # Final wait for all services to stabilize
            logger.info("Waiting for services to stabilize...")
            time.sleep(5)
            
            # Verify all containers are still running
            for name, container in self.containers.items():
                container.reload()
                if container.status != "running":
                    raise Exception(f"Container {name} is not running: {container.status}")
                    
            logger.info("All services are running and ready")
            
            return {
                "mqtt-broker": mqtt,
                "camera-detector": camera, 
                "fire-consensus": consensus,
                "gpio-trigger": trigger
            }
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            self.cleanup()
            raise
            
    def cleanup(self):
        """Clean up test containers and networks with improved error handling"""
        logger.info("Cleaning up test environment...")
        
        # Stop and remove containers in reverse order
        container_order = ["gpio-trigger", "fire-consensus", "camera-detector", "mqtt-broker"]
        
        for name in container_order:
            if name in self.containers:
                container = self.containers[name]
                container_name = container.name
                
                try:
                    # Reload container state
                    container.reload()
                    
                    if container.status == "running":
                        logger.info(f"Stopping {container_name}...")
                        container.stop(timeout=5)
                        
                    # Wait a bit before removal
                    time.sleep(0.5)
                    
                    try:
                        container.remove(force=True)
                        logger.info(f"✓ Removed {container_name}")
                    except docker.errors.APIError as e:
                        if "removal" in str(e) and "already in progress" in str(e):
                            logger.warning(f"Container {container_name} already being removed")
                        elif "No such container" in str(e):
                            logger.warning(f"Container {container_name} already removed")
                        else:
                            logger.error(f"Failed to remove {container_name}: {e}")
                            
                except docker.errors.NotFound:
                    logger.info(f"Container {container_name} already removed")
                except Exception as e:
                    logger.error(f"Error cleaning up {container_name}: {e}")
                    
        # Remove networks
        for name, network in self.networks.items():
            try:
                network.reload()
                network.remove()
                logger.info(f"✓ Removed network {name}")
            except docker.errors.NotFound:
                logger.info(f"Network {name} already removed")
            except Exception as e:
                logger.warning(f"Could not remove network {name}: {e}")
                
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
    # Command line interface for manual testing
    setup = IntegrationTestSetup()
    
    if len(sys.argv) > 1 and sys.argv[1] == "cleanup":
        setup.cleanup()
    else:
        try:
            containers = setup.setup_all_services()
            print("\\n✓ All services ready for integration testing")
            print("Run 'python integration_setup_fixed.py cleanup' to clean up")
            
            # Keep running for manual testing
            print("\\nPress Ctrl+C to stop and cleanup...")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\\nShutting down...")
                
        except Exception as e:
            print(f"✗ Setup failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            setup.cleanup()
            sys.exit(0)