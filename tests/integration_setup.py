#!/usr/bin/env python3.12
"""
Integration test setup utilities
Creates required Docker containers and services for testing
"""
import os
import sys
import time
import docker
import tempfile
import subprocess
from pathlib import Path

class IntegrationTestSetup:
    """Manages Docker containers and services for integration testing"""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.temp_dir = None
        self.containers = {}
        self.networks = {}
        
    def setup_network(self):
        """Create test network"""
        try:
            network = self.docker_client.networks.get("wildfire_test_net")
        except docker.errors.NotFound:
            network = self.docker_client.networks.create(
                "wildfire_test_net",
                driver="bridge",
                ipam=docker.types.IPAMConfig(
                    pool_configs=[
                        docker.types.IPAMPool(subnet="192.168.200.0/24")
                    ]
                )
            )
        self.networks["test_net"] = network
        return network
        
    def setup_mqtt_broker(self):
        """Create and start MQTT broker container"""
        network = self.setup_network()
        
        try:
            container = self.docker_client.containers.get("mqtt-broker-test")
            if container.status != "running":
                container.start()
        except docker.errors.NotFound:
            # Create MQTT broker container on different port to avoid conflicts
            container = self.docker_client.containers.run(
                "eclipse-mosquitto:2.0",
                name="mqtt-broker-test",
                ports={'1883/tcp': 18833},  # Use different external port
                network="wildfire_test_net",
                detach=True,
                remove=True,
                command="mosquitto -c /mosquitto-no-auth.conf"
            )
            
        self.containers["mqtt-broker"] = container
        
        # Wait for MQTT broker to be ready
        time.sleep(3)
        return container
        
    def setup_camera_detector(self):
        """Create camera detector container"""
        mqtt_container = self.setup_mqtt_broker()
        
        # Build camera detector if needed
        try:
            image = self.docker_client.images.get("wildfire-watch/camera_detector:test")
        except docker.errors.ImageNotFound:
            # Build the image
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
            container = self.docker_client.containers.get("camera-detector-test")
            if container.status != "running":
                container.start()
        except docker.errors.NotFound:
            container = self.docker_client.containers.run(
                "wildfire-watch/camera_detector:test",
                name="camera-detector-test",
                network="wildfire_test_net",
                environment={
                    "MQTT_BROKER": "mqtt-broker-test",
                    "MQTT_PORT": "1883",  # Internal port within container network
                    "LOG_LEVEL": "DEBUG",
                    "CAMERA_CREDENTIALS": "username:password,admin:",
                    "DISCOVERY_INTERVAL": "30"
                },
                detach=True,
                remove=True
            )
            
        self.containers["camera-detector"] = container
        time.sleep(2)
        return container
        
    def setup_fire_consensus(self):
        """Create fire consensus container"""
        mqtt_container = self.setup_mqtt_broker()
        
        try:
            image = self.docker_client.images.get("wildfire-watch/fire_consensus:test")
        except docker.errors.ImageNotFound:
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
            container = self.docker_client.containers.get("fire-consensus-test")
            if container.status != "running":
                container.start()
        except docker.errors.NotFound:
            container = self.docker_client.containers.run(
                "wildfire-watch/fire_consensus:test",
                name="fire-consensus-test",
                network="wildfire_test_net",
                environment={
                    "MQTT_BROKER": "mqtt-broker-test",
                    "MQTT_PORT": "1883",  # Internal port within container network
                    "LOG_LEVEL": "DEBUG",
                    "CONSENSUS_THRESHOLD": "2",
                    "MIN_CONFIDENCE": "0.7"
                },
                detach=True,
                remove=True
            )
            
        self.containers["fire-consensus"] = container
        time.sleep(2)
        return container
        
    def setup_gpio_trigger(self):
        """Create GPIO trigger container"""
        mqtt_container = self.setup_mqtt_broker()
        
        try:
            image = self.docker_client.images.get("wildfire-watch/gpio_trigger:test")
        except docker.errors.ImageNotFound:
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
            container = self.docker_client.containers.get("gpio-trigger-test")
            if container.status != "running":
                container.start()
        except docker.errors.NotFound:
            container = self.docker_client.containers.run(
                "wildfire-watch/gpio_trigger:test",
                name="gpio-trigger-test",
                network="wildfire_test_net",
                environment={
                    "MQTT_BROKER": "mqtt-broker-test",
                    "MQTT_PORT": "1883",  # Internal port within container network
                    "LOG_LEVEL": "DEBUG",
                    "GPIO_SIMULATION": "true",
                    "SIMULATION_MODE_WARNINGS": "false"
                },
                detach=True,
                remove=True
            )
            
        self.containers["gpio-trigger"] = container
        time.sleep(2)
        return container
        
    def setup_all_services(self):
        """Setup all required services for integration testing"""
        print("Setting up integration test environment...")
        
        mqtt = self.setup_mqtt_broker()
        print(f"✓ MQTT broker: {mqtt.name}")
        
        camera = self.setup_camera_detector()
        print(f"✓ Camera detector: {camera.name}")
        
        consensus = self.setup_fire_consensus()
        print(f"✓ Fire consensus: {consensus.name}")
        
        trigger = self.setup_gpio_trigger()
        print(f"✓ GPIO trigger: {trigger.name}")
        
        # Wait for all services to be ready
        print("Waiting for services to be ready...")
        time.sleep(10)
        
        return {
            "mqtt-broker": mqtt,
            "camera-detector": camera, 
            "fire-consensus": consensus,
            "gpio-trigger": trigger
        }
        
    def cleanup(self):
        """Clean up test containers and networks"""
        print("Cleaning up test environment...")
        
        # Stop and remove containers
        for name, container in self.containers.items():
            try:
                container.stop(timeout=5)
                container.remove()
                print(f"✓ Removed {name}")
            except Exception as e:
                print(f"⚠ Could not remove {name}: {e}")
                
        # Remove networks
        for name, network in self.networks.items():
            try:
                network.remove()
                print(f"✓ Removed network {name}")
            except Exception as e:
                print(f"⚠ Could not remove network {name}: {e}")
                
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
            print("\n✓ All services ready for integration testing")
            print("Run 'python integration_setup.py cleanup' to clean up")
        except Exception as e:
            print(f"✗ Setup failed: {e}")
            setup.cleanup()
            sys.exit(1)