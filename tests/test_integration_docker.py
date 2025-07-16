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
import paho.mqtt.client as mqtt
from typing import Dict, List, Optional

# Import parallel test utilities
from test_utils.helpers import ParallelTestContext, DockerContainerManager
from test_utils.topic_namespace import create_namespaced_client

class DockerIntegrationTest:
    """Test integration with Docker containers"""
    
    def __init__(self, parallel_context: ParallelTestContext, docker_manager: DockerContainerManager):
        self.docker_client = docker.from_env()
        self.parallel_context = parallel_context
        self.docker_manager = docker_manager
        self.containers = {}
        self.test_passed = False
        self.network = None
        
    def build_docker_images(self):
        """Build required Docker images"""
        print("Building Docker images...")
        
        services = ['mqtt_broker', 'fire_consensus', 'gpio_trigger']
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
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
                except Exception as e:
                    print(f"Error building {service}: {e}")
                    
    def create_test_network(self):
        """Network is provided by fixture, just return it"""
        if self.network:
            print(f"Using test network from fixture: {self.network.name}")
            return self.network
        else:
            raise RuntimeError("No test network provided by fixture")
        
    def start_mqtt_container(self):
        """Start MQTT broker container"""
        print("Starting MQTT broker container...")
        
        # Create network if not exists
        if not self.network:
            self.create_test_network()
        
        # Check if mosquitto image exists
        try:
            self.docker_client.images.get("eclipse-mosquitto:latest")
        except docker.errors.ImageNotFound:
            print("Pulling mosquitto image...")
            self.docker_client.images.pull("eclipse-mosquitto:latest")
        
        # Remove existing test container if any
        container_name = f"mqtt-test-{self.parallel_context.worker_id}"
        try:
            old = self.docker_client.containers.get(container_name)
            old.stop()
            old.remove()
        except docker.errors.NotFound:
            pass
            
        # Create mosquitto config that listens on all interfaces
        mosquitto_config = """listener 1883 0.0.0.0
allow_anonymous true
log_type all
log_dest stdout
"""
        
        # Create and start container with custom config using dynamic port
        container = self.docker_client.containers.run(
            "eclipse-mosquitto:latest",
            name=container_name,
            network=self.network.name,
            ports={'1883/tcp': None},  # Use dynamic port allocation
            detach=True,
            remove=False,
            command=[
                "sh", "-c",
                f'echo "{mosquitto_config}" > /mosquitto/config/mosquitto.conf && '
                'exec mosquitto -c /mosquitto/config/mosquitto.conf'
            ]
        )
        
        self.containers['mqtt'] = container
        
        # Get the dynamically allocated port
        container.reload()  # Refresh container info to get port mapping
        port_info = container.attrs['NetworkSettings']['Ports']['1883/tcp']
        if port_info and len(port_info) > 0:
            self.mqtt_port = int(port_info[0]['HostPort'])
        else:
            print("ERROR: Could not get MQTT container port")
            return False
        
        print(f"MQTT broker started on port {self.mqtt_port}")
        
        # Wait for MQTT to be ready
        time.sleep(3)
        
        # Test connection
        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "test")
            client.connect("localhost", self.mqtt_port, 60)
            client.disconnect()
            print("✓ MQTT broker ready")
            return True
        except Exception as e:
            print(f"MQTT connection failed: {e}")
            return False
            
    def start_consensus_container(self):
        """Start fire consensus container"""
        print("Starting fire consensus container...")
        
        # Get the MQTT container name that was used
        mqtt_container_name = f"mqtt-test-{self.parallel_context.worker_id}"
        
        # Build custom consensus image for testing
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        consensus_path = os.path.join(project_root, 'fire_consensus')
        
        try:
            # Create test entrypoint script first
            test_entrypoint = '''#!/bin/sh
echo "Starting test environment..."
# Try to start D-Bus and Avahi, but continue if they fail
dbus-daemon --system --fork || echo "D-Bus failed (normal in test)"
if command -v avahi-daemon >/dev/null 2>&1; then
    mkdir -p /var/run/avahi-daemon || true
    avahi-daemon --no-drop-root --daemonize --no-chroot || echo "Avahi failed (normal in test)"
fi
echo "=================================================="
echo "Fire Consensus Service (TEST MODE)"
echo "MQTT Broker: ${MQTT_BROKER}"
echo "=================================================="
exec "$@"
'''
            
            test_entrypoint_path = os.path.join(consensus_path, 'test_entrypoint.sh')
            with open(test_entrypoint_path, 'w') as f:
                f.write(test_entrypoint)
            os.chmod(test_entrypoint_path, 0o755)
            
            # Create a test-specific Dockerfile that runs as root
            test_dockerfile = '''FROM python:3.12-slim

# Install system dependencies for mDNS and resilience
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        avahi-daemon \
        libnss-mdns \
        dbus \
        iputils-ping \
        && rm -rf /var/lib/apt/lists/*

# Configure mDNS resolution
COPY fire_consensus/nsswitch.conf /etc/nsswitch.conf

# For testing, run as root to allow D-Bus and Avahi
WORKDIR /app

# Copy utils module from root context
COPY utils /utils
# Add root to Python path so utils can be imported
ENV PYTHONPATH=/

# Install Python dependencies
COPY fire_consensus/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY fire_consensus/consensus.py .
COPY fire_consensus/test_entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3.12", "-u", "consensus.py"]
'''
            
            # Write test Dockerfile
            test_dockerfile_path = os.path.join(consensus_path, 'Dockerfile.test')
            with open(test_dockerfile_path, 'w') as f:
                f.write(test_dockerfile)
            
            # Build image with test Dockerfile using project root context
            dockerfile_relative = os.path.join('fire_consensus', 'Dockerfile.test')
            image, _ = self.docker_client.images.build(
                path=project_root,  # Use project root as build context
                dockerfile=dockerfile_relative,
                tag="wildfire-watch/fire_consensus:test",
                rm=True
            )
            
            # Clean up test files
            os.remove(test_dockerfile_path)
            os.remove(test_entrypoint_path)
            
            # Remove old container
            consensus_container_name = f"consensus-test-{self.parallel_context.worker_id}"
            try:
                old = self.docker_client.containers.get(consensus_container_name)
                old.stop()
                old.remove()
            except docker.errors.NotFound:
                pass
            
            # Start container
            container = self.docker_client.containers.run(
                "wildfire-watch/fire_consensus:test",
                name=consensus_container_name,
                network=self.network.name,
                environment={
                    'MQTT_BROKER': mqtt_container_name,  # Use the actual MQTT container name
                    'MQTT_PORT': '1883',  # Internal port, not mapped port
                    'CONSENSUS_THRESHOLD': '2',
                    'SINGLE_CAMERA_TRIGGER': 'false',
                    'MIN_CONFIDENCE': '0.7',
                    'LOG_LEVEL': 'DEBUG',
                    'DETECTION_WINDOW': '15',
                    'INCREASE_COUNT': '3',
                    'AREA_INCREASE_RATIO': '1.1',
                    'MOVING_AVERAGE_WINDOW': '2',
                    'COOLDOWN_PERIOD': '0'
                },
                detach=True,
                remove=False
                # Use default entrypoint and command from Dockerfile
            )
            
            self.containers['consensus'] = container
            print("✓ Fire consensus container started")
            return True
            
        except Exception as e:
            print(f"Failed to start consensus container: {e}")
            return False
            
    def test_docker_fire_detection_flow(self):
        """Test fire detection flow with Docker containers"""
        print("\n=== DOCKER INTEGRATION TEST ===")
        
        # 1. Build images
        self.build_docker_images()
        
        # 2. Start MQTT
        if not self.start_mqtt_container():
            return False
            
        # 3. Start consensus
        if not self.start_consensus_container():
            return False
            
        # 4. Wait for services to connect
        print("Waiting for services to initialize...")
        time.sleep(5)
        
        # 5. Monitor for fire trigger
        triggered = False
        messages = []
        
        def on_message(client, userdata, msg):
            nonlocal triggered
            messages.append((msg.topic, msg.payload))
            if msg.topic == "fire/trigger":
                print(f"✓ Fire trigger received!")
                triggered = True
            elif "consensus" in msg.topic or "telemetry" in msg.topic:
                print(f"Debug: {msg.topic} - {msg.payload[:100]}")
                
        monitor = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "monitor")
        monitor.on_message = on_message
        monitor.connect("localhost", self.mqtt_port)
        monitor.subscribe("#", qos=0)  # Subscribe to all topics for debugging
        monitor.loop_start()
        
        # 6. Send camera telemetry first
        print("Sending camera telemetry...")
        publisher = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "publisher")
        publisher.connect("localhost", self.mqtt_port)
        
        # Send telemetry for two cameras
        for camera_id in ['docker_test_cam1', 'docker_test_cam2']:
            telemetry = {
                'camera_id': camera_id,
                'status': 'online',
                'timestamp': time.time()
            }
            publisher.publish('system/camera_telemetry', json.dumps(telemetry), qos=1)
            print(f"  Sent telemetry for {camera_id}")
        
        time.sleep(2)  # Wait for telemetry to be processed
        
        # 7. Inject fire detections with growth pattern
        print("Injecting fire detections...")
        
        # Send initial small detections
        for i in range(4):
            for camera_id in ['docker_test_cam1', 'docker_test_cam2']:
                detection = {
                    'camera_id': camera_id,
                    'object_type': 'fire',  # Fixed key name
                    'object_id': f'{camera_id}_fire_1',
                    'confidence': 0.85,
                    'bbox': [0.1, 0.1, 0.2, 0.2],  # Small initial fire
                    'timestamp': time.time()
                }
                publisher.publish(f'fire/detection/{camera_id}', json.dumps(detection), qos=1)
            print(f"  Sent initial detection {i+1}")
            time.sleep(0.5)
        
        # Send growing detections
        for i in range(6):
            size = 0.2 + (i * 0.05)  # Growing fire
            for camera_id in ['docker_test_cam1', 'docker_test_cam2']:
                detection = {
                    'camera_id': camera_id,
                    'object_type': 'fire',
                    'object_id': f'{camera_id}_fire_1',
                    'confidence': 0.85,
                    'bbox': [0.1, 0.1, 0.1 + size, 0.1 + size],  # Growing bbox
                    'timestamp': time.time()
                }
                publisher.publish(f'fire/detection/{camera_id}', json.dumps(detection), qos=1)
            print(f"  Sent growing detection {i+5} (size: {size:.2f})")
            time.sleep(0.5)
            
        # 8. Wait and check
        print("Waiting for consensus...")
        time.sleep(10)
        
        # Check consensus container logs
        if 'consensus' in self.containers:
            logs = self.containers['consensus'].logs(tail=20).decode()
            print("\nConsensus container logs:")
            print(logs)
        
        # Cleanup
        monitor.loop_stop()
        monitor.disconnect()
        publisher.disconnect()
        
        self.test_passed = triggered
        return triggered
        
    def cleanup(self):
        """Clean up containers (network is managed by fixture)"""
        print("\nCleaning up containers...")
        for name, container in self.containers.items():
            try:
                container.stop()
                container.remove()
                print(f"✓ Removed {name} container")
            except Exception as e:
                print(f"Error removing {name}: {e}")
        
        # Network cleanup is handled by the fixture, not here


@pytest.mark.slow
@pytest.mark.docker
@pytest.mark.integration
def test_docker_integration(parallel_test_context, docker_container_manager):
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
    
    test = DockerIntegrationTest(parallel_test_context, docker_container_manager)
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
            for container in network.containers:
                try:
                    network.disconnect(container, force=True)
                    print(f"Final cleanup: Disconnected {container.name} from network")
                except Exception as e:
                    print(f"Warning: Could not disconnect {container.name} during cleanup: {e}")
            # Now remove the network
            network.remove()
            print(f"Cleaned up network: {network.name}")
        except Exception as e:
            print(f"Warning: Error during final network cleanup: {e}")


if __name__ == "__main__":
    # Standalone execution
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tests.helpers import ParallelTestContext, DockerContainerManager
    
    context = ParallelTestContext("standalone")
    manager = DockerContainerManager("standalone")
    test_docker_integration(context, manager)