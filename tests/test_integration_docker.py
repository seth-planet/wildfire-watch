#!/usr/bin/env python3.12
"""
Docker-based integration tests for Wildfire Watch
Tests services running in Docker containers
"""
import os
import sys
import time
import json
import pytest
import subprocess
import docker
import threading
import paho.mqtt.client as mqtt
from typing import Dict, List, Optional

# Import parallel test utilities
from test_utils.helpers import ParallelTestContext, DockerContainerManager
from test_utils.topic_namespace import create_namespaced_client

class DockerIntegrationTest:
    """Test integration with Docker containers"""
    
    def __init__(self, parallel_context: ParallelTestContext, docker_manager: DockerContainerManager, mqtt_broker=None):
        self.docker_client = docker.from_env()
        self.parallel_context = parallel_context
        self.docker_manager = docker_manager
        self.mqtt_broker = mqtt_broker
        self.containers = {}
        self.test_passed = False
        self.network = None
        
    def build_docker_images(self):
        """Build required Docker images - creates them if missing or outdated"""
        print("Building/updating Docker images...")
        
        services = ['mqtt_broker', 'fire_consensus', 'gpio_trigger']
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Check if docker-compose.yml exists
        compose_file = os.path.join(project_root, 'docker-compose.yml')
        if not os.path.exists(compose_file):
            print(f"ERROR: docker-compose.yml not found at {compose_file}")
            print("This file is required for building service containers.")
            return False
        
        images_built = []
        for service in services:
            dockerfile_path = os.path.join(project_root, service)
            if os.path.exists(os.path.join(dockerfile_path, 'Dockerfile')):
                print(f"Building {service}...")
                try:
                    # Try docker compose v2 first, then v1
                    result = subprocess.run([
                        'docker', 'compose', 'build', service
                    ], cwd=project_root, capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        # Try docker-compose v1
                        result = subprocess.run([
                            'docker-compose', 'build', service
                        ], cwd=project_root, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        print(f"✓ Built {service}")
                        images_built.append(service)
                    else:
                        print(f"Failed to build {service}: {result.stderr}")
                        # Try direct docker build as fallback with correct context
                        dockerfile_relative = os.path.join(service, 'Dockerfile')
                        self.docker_client.images.build(
                            path=project_root,  # Use project root as context
                            dockerfile=dockerfile_relative,
                            tag=f"wildfire-watch/{service}:test",
                            rm=True
                        )
                        print(f"✓ Built {service} with docker build")
                        images_built.append(service)
                except Exception as e:
                    print(f"Error building {service}: {e}")
                    return False
            else:
                print(f"WARNING: Dockerfile not found for {service}")
                return False
        
        return len(images_built) == len(services)
                    
    def create_test_network(self):
        """Network is provided by fixture, just return it"""
        if self.network:
            print(f"Using test network from fixture: {self.network.name}")
            return self.network
        else:
            raise RuntimeError("No test network provided by fixture")
        
    def start_mqtt_container(self):
        """Use test MQTT broker from fixture"""
        print("Using test MQTT broker...")
        
        # Get connection parameters from test broker
        if self.mqtt_broker:
            conn_params = self.mqtt_broker.get_connection_params()
            self.mqtt_port = conn_params['port']
            self.mqtt_host = conn_params['host']
            
            # Test connection
            try:
                client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "test")
                client.connect(self.mqtt_host, self.mqtt_port, 60)
                client.disconnect()
                print(f"✓ MQTT broker ready on {self.mqtt_host}:{self.mqtt_port}")
                return True
            except Exception as e:
                print(f"MQTT connection failed: {e}")
                return False
        else:
            print("ERROR: No test MQTT broker provided")
            return False
            
    def start_consensus_container(self):
        """Start fire consensus container"""
        print("Starting fire consensus container...")
        
        # Get MQTT connection parameters
        mqtt_host = self.mqtt_host if hasattr(self, 'mqtt_host') else 'localhost'
        mqtt_port = self.mqtt_port if hasattr(self, 'mqtt_port') else 1883
        
        # Build consensus image using standard Dockerfile (like SDK test)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        try:
            # Build image from project root with standard fire_consensus Dockerfile
            print("Building fire_consensus image...")
            image, build_logs = self.docker_client.images.build(
                path=project_root,
                dockerfile="fire_consensus/Dockerfile",  # Use standard Dockerfile
                tag="wildfire-watch/fire_consensus:test",
                rm=True,
                nocache=False  # Allow cache for faster builds
            )
            
            # Log build output for debugging
            for log in build_logs:
                if 'stream' in log:
                    debug_msg = log['stream'].strip()
                    if debug_msg and not debug_msg.startswith('Step'):
                        print(f"  Build: {debug_msg[:100]}")
            
            print("✓ Built fire_consensus image")
            
            # Remove old container
            consensus_container_name = f"consensus-test-{self.parallel_context.worker_id}"
            try:
                old = self.docker_client.containers.get(consensus_container_name)
                old.stop()
                old.remove()
            except docker.errors.NotFound:
                pass
            
            # Start container with proper topic namespace and environment
            container = self.docker_client.containers.run(
                "wildfire-watch/fire_consensus:test",
                name=consensus_container_name,
                network_mode='host',  # Use host network to access test broker
                environment={
                    'MQTT_BROKER': mqtt_host,  # Use test broker host
                    'MQTT_PORT': str(mqtt_port),  # Use test broker port
                    'TOPIC_PREFIX': self.parallel_context.namespace.namespace,  # Add topic namespace
                    'CONSENSUS_THRESHOLD': '2',
                    'SINGLE_CAMERA_TRIGGER': 'true',  # Allow single camera to trigger for testing
                    'MIN_CONFIDENCE': '0.7',  # Lower confidence threshold
                    'MIN_AREA_RATIO': '0.001',  # Very small minimum area
                    'AREA_INCREASE_RATIO': '1.2',  # Lower growth requirement
                    'MIN_CONFIDENCE': '0.7',
                    'LOG_LEVEL': 'DEBUG',
                    'CAMERA_WINDOW': '15',  # Correct env var name (not DETECTION_WINDOW)
                    'INCREASE_COUNT': '3',
                    'AREA_INCREASE_RATIO': '1.1',  # 10% growth threshold like SDK test
                    'MOVING_AVERAGE_WINDOW': '2',  # Reduced for faster testing
                    'COOLDOWN_PERIOD': '0',
                    'CAMERA_TIMEOUT': '300',  # 5 minutes timeout for testing
                    'PYTHONUNBUFFERED': '1'  # Force unbuffered output for debugging
                },
                command=["python3.12", "-u", "consensus.py"],  # Skip entrypoint, use Python directly
                detach=True,
                remove=False
            )
            
            self.containers['consensus'] = container
            print("✓ Fire consensus container started")
            
            # Wait for initialization and MQTT connection
            print("Waiting for consensus service to initialize...")
            time.sleep(8)  # Give service time to start properly
            
            # Check if container is still running
            container.reload()
            if container.status != 'running':
                logs = container.logs(tail=50).decode('utf-8')
                print(f"ERROR: Container exited early. Logs:\n{logs}")
                return False
            
            # Check initial logs
            logs = container.logs(tail=20).decode('utf-8')
            print(f"Consensus initial logs:\n{logs}")
            
            return True
            
        except Exception as e:
            print(f"Failed to start consensus container: {e}")
            return False
            
    def test_docker_fire_detection_flow(self):
        """Test fire detection flow with Docker containers"""
        print("\n=== DOCKER INTEGRATION TEST ===")
        
        # 1. Build images - always try to build/update
        if not self.build_docker_images():
            # Check if Docker is available
            try:
                self.docker_client.ping()
                # Docker is available but build failed - this is a real error
                raise RuntimeError(
                    "Docker is available but images could not be built. "
                    "Please check docker-compose.yml and Dockerfiles exist and are valid."
                )
            except docker.errors.DockerException as e:
                # Docker is not available - fail with clear message
                raise RuntimeError(
                    f"Docker is not available or not running: {e}\n"
                    "Please install Docker and ensure the Docker daemon is running."
                ) from e
        
        # 2. Start MQTT
        if not self.start_mqtt_container():
            return False
            
        # 3. Start consensus
        if not self.start_consensus_container():
            return False
            
        # 4. Wait for services to connect
        print("Waiting for services to initialize...")
        
        # Get topic namespace from parallel context
        topic_namespace = self.parallel_context.namespace.namespace
        fire_trigger_topic = f"{topic_namespace}/fire/trigger"
        
        # 5. Monitor for fire trigger with proper topic namespace
        triggered = False
        messages = []
        
        def on_message(client, userdata, msg):
            nonlocal triggered
            messages.append((msg.topic, msg.payload))
            if msg.topic == fire_trigger_topic:
                print(f"✓ Fire trigger received on {msg.topic}!")
                triggered = True
            elif "consensus" in msg.topic or "telemetry" in msg.topic or "fire" in msg.topic:
                print(f"Debug: {msg.topic} - {msg.payload[:100]}")
                
        monitor = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, f"monitor_{self.parallel_context.worker_id}")
        monitor.on_message = on_message
        monitor.connect(self.mqtt_host, self.mqtt_port)
        monitor.subscribe(f"{topic_namespace}/#", qos=0)  # Subscribe to namespaced topics
        monitor.loop_start()
        
        # Create publisher client and start loop for async operations
        publisher = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, f"publisher_{self.parallel_context.worker_id}")
        publisher.connect(self.mqtt_host, self.mqtt_port)
        publisher.loop_start()  # Start async loop for publisher
        
        # Give consensus service time to connect to MQTT broker
        print("Waiting for consensus service to connect to MQTT...")
        time.sleep(5)  # Increased wait for service initialization
        
        # Verify consensus service is responding
        health_topic = f"{topic_namespace}/system/fire_consensus/health"
        health_received = False
        
        def on_health(client, userdata, msg):
            nonlocal health_received
            if msg.topic == health_topic:
                health_received = True
                print(f"✓ Consensus service health check received")
        
        monitor.message_callback_add(health_topic, on_health)
        
        # Wait for health message
        for i in range(10):
            if health_received:
                break
            time.sleep(1)
        
        if not health_received:
            print("WARNING: No health messages from consensus service")
        
        # 6. Send camera telemetry first
        print("Sending camera telemetry...")
        
        # Debug: Check connection
        print(f"Publisher connected to {self.mqtt_host}:{self.mqtt_port}")
        
        # Send telemetry for two cameras (without waiting for publish)
        for camera_id in ['docker_test_cam1', 'docker_test_cam2']:
            telemetry = {
                'camera_id': camera_id,
                'status': 'online',
                'timestamp': time.time()
            }
            topic = f'{topic_namespace}/system/camera_telemetry'
            print(f"  Publishing telemetry for {camera_id} to topic: {topic}")
            # Don't wait for publish - just send and continue (like SDK test)
            publisher.publish(topic, json.dumps(telemetry), qos=1, retain=False)
            print(f"  ✓ Sent telemetry for {camera_id}")
            time.sleep(0.1)  # Small delay between messages
        
        print("Waiting for telemetry to be processed...")
        time.sleep(3)  # Give consensus time to process telemetry
        
        # 7. Inject fire detections with growth pattern
        print("Injecting fire detections with growth pattern...")
        
        # Send more detections to ensure consensus triggers
        num_detections = 20  # Increased to 20 for better chance of triggering
        current_time = time.time()
        
        for i in range(num_detections):
            # Use very aggressive growth to ensure trigger
            # Start at 150 pixels, grow by 60% each time (very aggressive)
            size = 150 * (1.6 ** i)
            
            for camera_id in ['docker_test_cam1', 'docker_test_cam2']:
                detection = {
                    'camera_id': camera_id,
                    'confidence': 0.85 + (i * 0.01),  # Increasing confidence
                    'bbox': [100, 100, 100 + size, 100 + size],  # Use pixel coordinates
                    'timestamp': current_time + i * 0.5,  # Space detections 0.5s apart
                    'object_id': 'fire_1',  # Same object ID for growth tracking
                    'object_type': 'fire'  # Add explicit object type
                }
                # Publish to the correct topic with namespace
                topic = f'{topic_namespace}/fire/detection'
                # Don't wait for publish - just send (like SDK test)
                publisher.publish(
                    topic,
                    json.dumps(detection), 
                    qos=1,
                    retain=False
                )
                
            print(f"  Sent detection {i+1}/{num_detections} (size: {size:.0f} pixels, conf: {0.85 + (i * 0.01):.2f})")
            
            # Add delay between detection batches for processing
            if i < num_detections - 1:  # Don't delay after last detection
                time.sleep(0.3)  # Give consensus time to process each batch
            
        # 8. Wait and check
        print("Waiting for consensus...")
        # Wait for trigger event instead of fixed sleep
        trigger_event = threading.Event()
        
        def on_trigger_message(client, userdata, msg):
            nonlocal triggered
            if msg.topic == fire_trigger_topic:
                print(f"✓ Fire trigger received on {msg.topic}!")
                triggered = True
                trigger_event.set()
        
        # Update the callback for fire trigger
        monitor.message_callback_add(fire_trigger_topic, on_trigger_message)
        
        # Wait for consensus with longer timeout
        if not trigger_event.wait(timeout=30):  # Increased from 15 to 30
            print("WARNING: No fire trigger received within 30 seconds")
        
        # Check consensus container logs
        if 'consensus' in self.containers:
            try:
                logs = self.containers['consensus'].logs(tail=20).decode()
                print("\nConsensus container logs:")
                print(logs)
            except docker.errors.NotFound:
                print("\nConsensus container not found - may have exited early")
        
        # Cleanup
        monitor.loop_stop()
        monitor.disconnect()
        publisher.loop_stop()  # Stop async loop
        publisher.disconnect()
        
        self.test_passed = triggered
        return triggered
        
    def cleanup(self):
        """Clean up containers (network is managed by fixture)"""
        print("\nCleaning up containers...")
        for name, container in list(self.containers.items()):  # Use list() to avoid modification during iteration
            try:
                # First check if container still exists
                container.reload()  # This will raise NotFound if container is gone
                container.stop(timeout=5)
                container.wait(condition='not-running', timeout=10)
                container.remove()
                print(f"✓ Removed {name} container")
            except docker.errors.NotFound:
                print(f"Container {name} already removed")
            except Exception as e:
                print(f"Error removing {name}: {e}")
        
        # Clear the containers dict after cleanup
        self.containers.clear()
        
        # Network cleanup is handled by the fixture, not here


@pytest.mark.slow
@pytest.mark.docker
@pytest.mark.integration
@pytest.mark.timeout(1800)  # Use 1800 second timeout as requested
def test_docker_integration(parallel_test_context, docker_container_manager, test_mqtt_broker):
    """Test Docker container integration"""
    import docker
    
    # Create a test network for this test
    docker_client = docker.from_env()
    network_name = f"test-network-{parallel_test_context.worker_id}"
    
    # Clean up any existing network
    try:
        old_net = docker_client.networks.get(network_name)
        # Disconnect all containers from the network first
        for container in old_net.containers:
            try:
                old_net.disconnect(container, force=True)
                print(f"Disconnected {container.name} from network {network_name}")
            except Exception as e:
                print(f"Warning: Could not disconnect {container.name}: {e}")
        # Now remove the network
        old_net.remove()
        print(f"Removed existing network: {network_name}")
    except docker.errors.NotFound:
        pass
    except Exception as e:
        print(f"Warning: Error cleaning up existing network: {e}")
        # Continue with test even if cleanup fails
    
    # Create new network
    network = docker_client.networks.create(
        name=network_name,
        driver="bridge"
    )
    
    test = DockerIntegrationTest(parallel_test_context, docker_container_manager, test_mqtt_broker)
    test.network = network
    
    try:
        success = test.test_docker_fire_detection_flow()
        assert success, "Docker integration test should trigger fire consensus"
        print("\n✅ DOCKER INTEGRATION TEST PASSED")
    finally:
        test.cleanup()
        # Clean up network - disconnect containers first
        try:
            # Refresh network state to get current containers
            network.reload()
            # Disconnect all remaining containers
            for container in list(network.containers):  # Use list() to avoid modification during iteration
                try:
                    # Check if container still exists before disconnecting
                    container.reload()
                    network.disconnect(container, force=True)
                    print(f"Final cleanup: Disconnected {container.name} from network")
                except docker.errors.NotFound:
                    print(f"Container already removed during cleanup")
                except Exception as e:
                    print(f"Warning: Could not disconnect container during cleanup: {e}")
            # Now remove the network
            network.remove()
            print(f"Cleaned up network: {network.name}")
        except docker.errors.NotFound:
            print(f"Network already removed")
        except Exception as e:
            print(f"Warning: Error during final network cleanup: {e}")


if __name__ == "__main__":
    # Standalone execution
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from test_utils.helpers import ParallelTestContext, DockerContainerManager
    
    context = ParallelTestContext("standalone")
    manager = DockerContainerManager("standalone")
    test_docker_integration(context, manager)