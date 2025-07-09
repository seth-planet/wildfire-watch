#!/usr/bin/env python3.12
"""
Docker SDK-based integration tests for Wildfire Watch
Tests services running in Docker containers using Docker SDK
"""
import os
import sys
import time
import json
import pytest
import docker
import paho.mqtt.client as mqtt
from typing import Dict, List, Optional
import logging
from pathlib import Path

# Add parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests.helpers import DockerHealthChecker, ensure_docker_available, requires_docker

logger = logging.getLogger(__name__)


class DockerSDKIntegrationTest:
    """Test integration using Docker SDK directly"""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.containers = {}
        self.test_passed = False
        self.network = None
        
    def setup_network(self):
        """Create Docker network for test containers"""
        network_name = "wildfire-test-net"
        
        # Remove existing network if any
        try:
            old_net = self.docker_client.networks.get(network_name)
            old_net.remove()
        except docker.errors.NotFound:
            pass
            
        # Create new network
        self.network = self.docker_client.networks.create(
            name=network_name,
            driver="bridge"
        )
        logger.info(f"Created network: {network_name}")
        return self.network
        
    def start_mqtt_container(self):
        """Start MQTT broker container"""
        logger.info("Starting MQTT broker container...")
        
        # Pull mosquitto image if needed
        try:
            self.docker_client.images.get("eclipse-mosquitto:latest")
        except docker.errors.ImageNotFound:
            logger.info("Pulling mosquitto image...")
            self.docker_client.images.pull("eclipse-mosquitto:latest")
        
        # Remove existing test container if any
        container_name = "mqtt-test-sdk"
        try:
            old_container = self.docker_client.containers.get(container_name)
            old_container.stop(timeout=5)
            old_container.remove()
        except docker.errors.NotFound:
            pass
            
        # Create mosquitto config with verbose logging
        config_content = """
listener 1883
allow_anonymous true
log_type all
log_dest stdout
"""
        
        # Create and start container
        container = self.docker_client.containers.run(
            "eclipse-mosquitto:latest",
            name=container_name,
            ports={'1883/tcp': 11884},  # Different port to avoid conflicts
            detach=True,
            remove=False,
            network=self.network.name,
            command=["sh", "-c", f"echo '{config_content}' > /mosquitto/config/mosquitto.conf && /usr/sbin/mosquitto -c /mosquitto/config/mosquitto.conf"]
        )
        
        self.containers['mqtt'] = container
        
        # Wait for MQTT to be ready
        for i in range(10):
            try:
                client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "test")
                client.connect("localhost", 11884, 60)
                client.disconnect()
                logger.info("✓ MQTT broker ready")
                return True
            except Exception as e:
                if i == 9:
                    logger.error(f"MQTT connection failed after 10 attempts: {e}")
                    return False
                time.sleep(1)
                
    def build_and_start_fire_consensus(self):
        """Build and start fire consensus container"""
        logger.info("Building and starting fire consensus container...")
        
        # Build the image
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        try:
            # Build image from project root with fire_consensus Dockerfile
            image, build_logs = self.docker_client.images.build(
                path=project_root,
                dockerfile="fire_consensus/Dockerfile",
                tag="wildfire-test/fire_consensus:test",
                rm=True,
                nocache=True
            )
            
            # Log build output
            for log in build_logs:
                if 'stream' in log:
                    logger.debug(log['stream'].strip())
                    
            logger.info("✓ Built fire_consensus image")
            
        except Exception as e:
            logger.error(f"Failed to build fire_consensus: {e}")
            return False
            
        # Remove existing container if any
        container_name = "fire-consensus-test-sdk"
        try:
            old_container = self.docker_client.containers.get(container_name)
            old_container.stop(timeout=5)
            old_container.remove()
        except docker.errors.NotFound:
            pass
            
        # Start container
        try:
            container = self.docker_client.containers.run(
                "wildfire-test/fire_consensus:test",
                name=container_name,
                detach=True,
                remove=False,
                network=self.network.name,
                environment={
                    'MQTT_BROKER': 'mqtt-test-sdk',
                    'MQTT_PORT': '1883',
                    'CONSENSUS_THRESHOLD': '2',
                    'MIN_CONFIDENCE': '0.7',
                    'SINGLE_CAMERA_TRIGGER': 'false',
                    'LOG_LEVEL': 'DEBUG',
                    'CAMERA_WINDOW': '15',  # Longer window for testing (correct env var name)
                    'INCREASE_COUNT': '3',      # Require 3 growing detections
                    'AREA_INCREASE_RATIO': '1.1',  # 10% growth threshold
                    'MOVING_AVERAGE_WINDOW': '2',  # Reduce for faster testing
                    'COOLDOWN_PERIOD': '0',  # No cooldown for testing
                    'CAMERA_TIMEOUT': '300',  # 5 minutes timeout for testing
                    'PYTHONUNBUFFERED': '1'  # Force unbuffered output for debugging
                },
                command=["python3.12", "-u", "consensus.py"]  # Skip entrypoint script, use Python 3.12 matching Dockerfile
            )
            
            self.containers['consensus'] = container
            logger.info("✓ Fire consensus container started")
            
            # Wait for initialization and MQTT connection
            time.sleep(8)  # Increased wait time
            
            # Check if container is still running
            container.reload()
            if container.status != 'running':
                logs = container.logs(tail=50).decode('utf-8')
                logger.error(f"Container exited. Logs:\n{logs}")
                return False
                
            # Check initial logs and wait for subscription
            logs = container.logs(tail=50).decode('utf-8')  # Get more logs
            logger.info(f"Initial consensus logs:\n{logs}")
            
            # Wait for MQTT to be fully connected
            if "Fire Consensus Service started" not in logs:
                logger.warning("Consensus service may not be fully started yet")
                time.sleep(3)  # Extra wait
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to start fire_consensus: {e}")
            return False
            
    def test_fire_detection_flow(self):
        """Test fire detection consensus flow"""
        logger.info("\n=== DOCKER SDK INTEGRATION TEST ===")
        
        try:
            # Setup network
            self.setup_network()
            
            # Start MQTT broker
            if not self.start_mqtt_container():
                return False
                
            # Build and start fire consensus
            if not self.build_and_start_fire_consensus():
                return False
                
            # Connect MQTT client for testing
            fire_triggered = False
            
            def on_message(client, userdata, msg):
                nonlocal fire_triggered
                logger.info(f"Received: {msg.topic} - {msg.payload}")
                if msg.topic == "fire/trigger":
                    fire_triggered = True
                    
            def on_connect(client, userdata, flags, rc, properties):
                logger.info(f"Test client connected with result code {rc}")
                client.subscribe("#")  # Subscribe to all topics for debugging
                    
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "test-publisher")
            client.on_message = on_message
            client.on_connect = on_connect
            client.connect("localhost", 11884, 60)
            client.loop_start()
            
            # Wait for connection
            time.sleep(2)
            
            # Send camera telemetry first so consensus knows about cameras
            logger.info("Sending camera telemetry...")
            for camera_id in ['camera_1', 'camera_2', 'camera_3']:
                telemetry = {
                    'camera_id': camera_id,
                    'status': 'online',
                    'timestamp': time.time()
                }
                client.publish('system/camera_telemetry', json.dumps(telemetry), retain=False)
                logger.info(f"  Sent telemetry for {camera_id}")
                time.sleep(0.1)  # Small delay between messages
            
            # Wait for telemetry to be processed
            time.sleep(3)
            
            # Send fire detections with growth pattern
            logger.info("Injecting fire detections with growth pattern...")
            
            # Send detections from multiple cameras with increasing size
            base_size = 100
            for i in range(10):  # More detections for growth analysis
                # Simulate fire growth
                size = base_size + (i * 20)  # Growing fire
                
                for camera_id in ['camera_1', 'camera_2', 'camera_3']:
                    detection = {
                        'camera_id': camera_id,
                        'confidence': 0.85,
                        'object_type': 'fire',
                        'object_id': f'{camera_id}_fire_1',  # Consistent object ID
                        'bounding_box': [0.1, 0.1, 0.1 + (size/1000), 0.1 + (size/1000)],  # Normalized bbox
                        'timestamp': time.time()
                    }
                    
                    topic = f"fire/detection/{camera_id}"
                    client.publish(topic, json.dumps(detection), retain=False)
                    logger.info(f"  Sent detection {i+1} from {camera_id} (size: {size})")
                    
                time.sleep(1)  # 1 second between detection sets
                    
            
            # Wait for consensus
            logger.info("Waiting for consensus...")
            time.sleep(10)  # Give more time for consensus processing
            
            # Check logs
            if 'consensus' in self.containers:
                logs = self.containers['consensus'].logs().decode('utf-8')  # Get all logs
                logger.info(f"\nConsensus container logs:\n{logs}")
                
            # Also check MQTT broker logs
            if 'mqtt' in self.containers:
                mqtt_logs = self.containers['mqtt'].logs(tail=50).decode('utf-8')
                logger.info(f"\nMQTT broker logs:\n{mqtt_logs}")
                
            client.loop_stop()
            client.disconnect()
            
            if fire_triggered:
                logger.info("✓ Fire trigger received - test passed!")
                self.test_passed = True
            else:
                logger.error("✗ No fire trigger received - test failed")
                self.test_passed = False
                
            return self.test_passed
            
        except Exception as e:
            logger.error(f"Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Clean up containers and network"""
        logger.info("\nCleaning up containers...")
        
        # Stop and remove containers
        for name, container in self.containers.items():
            try:
                container.stop(timeout=5)
                container.remove()
                logger.info(f"✓ Removed {name} container")
            except Exception as e:
                logger.error(f"Error removing {name}: {e}")
                
        # Remove network
        if self.network:
            try:
                self.network.remove()
                logger.info("✓ Removed test network")
            except Exception as e:
                logger.error(f"Error removing network: {e}")


@pytest.mark.docker
@pytest.mark.timeout(300)
def test_docker_sdk_integration(parallel_test_context, docker_container_manager):
    """Test Docker container integration using SDK"""
    test = DockerSDKIntegrationTest()
    test.parallel_context = parallel_test_context
    test.container_manager = docker_container_manager
    
    success = test.test_fire_detection_flow()
    assert success, "Docker SDK integration test should trigger fire consensus"


if __name__ == "__main__":
    # Enable debug logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run test
    test = DockerSDKIntegrationTest()
    success = test.test_fire_detection_flow()
    
    if success:
        print("\n✅ Test PASSED")
        sys.exit(0)
    else:
        print("\n❌ Test FAILED")
        sys.exit(1)