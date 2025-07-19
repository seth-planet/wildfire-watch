#!/usr/bin/env python3.12
"""
Docker SDK-based integration tests for Wildfire Watch
Tests services running in Docker containers using Docker SDK with proper test isolation
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
from threading import Event

# Import test utilities for proper isolation
from test_utils.helpers import ParallelTestContext, DockerContainerManager
from test_utils.topic_namespace import create_namespaced_client

logger = logging.getLogger(__name__)


class DockerSDKIntegrationTest:
    """Test integration using Docker SDK with proper isolation"""
    
    def __init__(self, parallel_context: ParallelTestContext, docker_manager: DockerContainerManager, mqtt_broker):
        self.docker_client = docker.from_env()
        self.parallel_context = parallel_context
        self.docker_manager = docker_manager
        self.mqtt_broker = mqtt_broker
        self.containers = {}
        self.test_passed = False
        self.network = None  # Will use host network mode
        
    def get_service_env(self, service_name: str) -> Dict[str, str]:
        """Get environment variables for a service with proper test isolation"""
        env = self.parallel_context.get_service_env(service_name)
        # Override MQTT settings to use test broker
        conn_params = self.mqtt_broker.get_connection_params()
        env['MQTT_BROKER'] = conn_params['host']
        env['MQTT_PORT'] = str(conn_params['port'])
        env['MQTT_TLS'] = 'false'
        return env
        
    def start_service_container(self, service_name: str, image: str, depends_on: List[str] = None) -> docker.models.containers.Container:
        """Start a service container with proper isolation"""
        logger.info(f"Starting {service_name} container...")
        
        container_name = self.docker_manager.get_container_name(f"sdk-{service_name}")
        env_vars = self.get_service_env(service_name)
        
        # Wait for dependencies
        if depends_on:
            for dep in depends_on:
                logger.info(f"Waiting for {dep} to be ready...")
                time.sleep(2)
        
        # Start container
        container = self.docker_manager.start_container(
            image=image,
            name=container_name,
            config={
                'environment': env_vars,
                'network_mode': 'host',
                'detach': True
            },
            wait_timeout=10
        )
        
        self.containers[service_name] = container
        logger.info(f"Started {service_name} container: {container_name}")
        return container
        
        # Not starting a separate MQTT container - using test broker
        
        # Wait for MQTT to be ready
        for i in range(10):
            try:
                client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "test")
                conn_params = self.mqtt_broker.get_connection_params()
                client.connect(conn_params['host'], conn_params['port'], 60)
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
        
        # Get connection params
        conn_params = self.mqtt_broker.get_connection_params()
        
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
        container_name = self.docker_manager.get_container_name("fire-consensus-sdk")
        try:
            old_container = self.docker_client.containers.get(container_name)
            old_container.stop(timeout=5)
            old_container.remove()
        except docker.errors.NotFound:
            pass
            
        # Get namespace prefix from parallel context
        namespace = self.parallel_context.namespace
        topic_prefix = namespace.namespace if hasattr(namespace, 'namespace') else ''
        
        # Start container
        try:
            container = self.docker_client.containers.run(
                "wildfire-test/fire_consensus:test",
                name=container_name,
                detach=True,
                remove=False,
                network_mode='host',
                environment={
                    'MQTT_BROKER': conn_params['host'],
                    'MQTT_PORT': str(conn_params['port']),
                    'TOPIC_PREFIX': topic_prefix,  # Add topic prefix for test isolation
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
            # Using test MQTT broker from fixture
            conn_params = self.mqtt_broker.get_connection_params()
            logger.info(f"Using test MQTT broker on {conn_params['host']}:{conn_params['port']}")
                
            # Build and start fire consensus
            if not self.build_and_start_fire_consensus():
                return False
                
            # Get namespace prefix
            namespace = self.parallel_context.namespace
            topic_prefix = namespace.namespace if hasattr(namespace, 'namespace') else ''
            
            # Connect MQTT client for testing
            fire_triggered = False
            
            def on_message(client, userdata, msg):
                nonlocal fire_triggered
                logger.info(f"Received: {msg.topic} - {msg.payload}")
                # Check for fire trigger with namespace prefix
                expected_topic = f"{topic_prefix}/fire/trigger" if topic_prefix else "fire/trigger"
                if msg.topic == expected_topic:
                    fire_triggered = True
                    
            def on_connect(client, userdata, flags, rc, properties):
                logger.info(f"Test client connected with result code {rc}")
                client.subscribe("#")  # Subscribe to all topics for debugging
                    
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "test-publisher")
            client.on_message = on_message
            client.on_connect = on_connect
            client.connect(conn_params['host'], conn_params['port'], 60)
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
                telemetry_topic = f"{topic_prefix}/system/camera_telemetry" if topic_prefix else "system/camera_telemetry"
                client.publish(telemetry_topic, json.dumps(telemetry), retain=False)
                logger.info(f"  Sent telemetry for {camera_id} to {telemetry_topic}")
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
                    
                    base_topic = f"fire/detection/{camera_id}"
                    topic = f"{topic_prefix}/{base_topic}" if topic_prefix else base_topic
                    client.publish(topic, json.dumps(detection), retain=False)
                    logger.info(f"  Sent detection {i+1} from {camera_id} (size: {size}) to {topic}")
                    
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
        """Clean up containers using DockerContainerManager"""
        logger.info("\nCleaning up containers...")
        
        # The DockerContainerManager will handle cleanup of containers it started
        # We just need to clear our tracking list
        self.containers.clear()
        logger.info("✓ Container cleanup handled by DockerContainerManager")


@pytest.mark.docker
@pytest.mark.integration
@pytest.mark.timeout(300)
def test_docker_sdk_integration(parallel_test_context, docker_container_manager, test_mqtt_broker):
    """Test Docker container integration using SDK with proper isolation"""
    test = DockerSDKIntegrationTest(parallel_test_context, docker_container_manager, test_mqtt_broker)
    
    try:
        success = test.test_fire_detection_flow()
        assert success, "Docker SDK integration test should trigger fire consensus"
    finally:
        test.cleanup()


if __name__ == "__main__":
    # Enable debug logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Cannot run without pytest fixtures
    print("This test requires pytest fixtures. Run with: pytest test_integration_docker_sdk.py")
    sys.exit(1)
    
    if success:
        print("\n✅ Test PASSED")
        sys.exit(0)
    else:
        print("\n❌ Test FAILED")
        sys.exit(1)