#!/usr/bin/env python3.12
"""
Test Helper Utilities for Wildfire Watch

This module provides utilities for testing with real MQTT brokers,
hardware validation, and multi-client simulation.
"""

import queue
import json
import threading
import contextlib
import docker
import pytest
import time
import yaml
import shutil
from functools import wraps
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
import paho.mqtt.client as mqtt

# Import topic namespace utilities
try:
    from topic_namespace import TopicNamespace, NamespacedMQTTClient, create_namespaced_client
except ImportError:
    from tests.topic_namespace import TopicNamespace, NamespacedMQTTClient, create_namespaced_client


class MqttMessageListener:
    """A thread-safe listener to capture MQTT messages for testing."""

    def __init__(self, mqtt_client: mqtt.Client, topic: str, timeout: float = 5.0):
        """
        Initialize the message listener.
        
        Args:
            mqtt_client: Connected MQTT client instance
            topic: Topic pattern to subscribe to (supports wildcards)
            timeout: Default timeout for waiting for messages
        """
        self.client = mqtt_client
        self.topic = topic
        self.messages = queue.Queue()
        self.timeout = timeout
        self._original_on_message = None

    def _on_message(self, client, userdata, message: mqtt.MQTTMessage):
        """Callback to add received messages to the queue."""
        self.messages.put(message)

    def __enter__(self):
        """Subscribe to topic and set up message handler."""
        # Store original callback to restore later
        self._original_on_message = self.client.on_message
        self.client.on_message = self._on_message
        self.client.subscribe(self.topic)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Unsubscribe and restore original callback."""
        self.client.unsubscribe(self.topic)
        self.client.on_message = self._original_on_message

    def wait_for_message(self, timeout: Optional[float] = None) -> mqtt.MQTTMessage:
        """
        Wait for a message and return it. Raises TimeoutError if none received.
        
        Args:
            timeout: Override default timeout (seconds)
            
        Returns:
            The received MQTT message
            
        Raises:
            TimeoutError: If no message received within timeout
            AssertionError: If message has empty payload
        """
        timeout = timeout or self.timeout
        try:
            msg = self.messages.get(timeout=timeout)
            if not msg.payload:
                raise AssertionError(f"Received message on topic '{self.topic}' with empty payload.")
            return msg
        except queue.Empty:
            raise TimeoutError(f"No message received on topic '{self.topic}' within {timeout}s.")

    def wait_for_json_message(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Wait for a message and decode its payload from JSON.
        
        Args:
            timeout: Override default timeout (seconds)
            
        Returns:
            Decoded JSON payload as dictionary
            
        Raises:
            TimeoutError: If no message received within timeout
            ValueError: If payload cannot be decoded as JSON
        """
        msg = self.wait_for_message(timeout)
        try:
            return json.loads(msg.payload.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to decode JSON from payload on topic '{self.topic}': {e}") from e

    def get_all_messages(self, max_wait: float = 0.5) -> List[mqtt.MQTTMessage]:
        """
        Get all messages received so far, waiting briefly for any stragglers.
        
        Args:
            max_wait: Maximum time to wait for additional messages
            
        Returns:
            List of all received messages
        """
        messages = []
        deadline = threading.Event()
        
        def timeout_handler():
            deadline.set()
            
        timer = threading.Timer(max_wait, timeout_handler)
        timer.start()
        
        try:
            while not deadline.is_set():
                try:
                    msg = self.messages.get(timeout=0.1)
                    messages.append(msg)
                except queue.Empty:
                    if messages:  # If we got messages, don't wait full duration
                        break
        finally:
            timer.cancel()
            
        return messages


# Hardware validation functions (import from conftest if available)
try:
    from conftest import has_coral_tpu, has_tensorrt, has_hailo
except ImportError:
    # Define them here if conftest is not accessible
    import subprocess
    import os
    
    def has_coral_tpu():
        """Check if Coral TPU is available"""
        try:
            result = subprocess.run(
                ['python3.8', '-c', 'from pycoral.utils.edgetpu import list_edge_tpus; print(len(list_edge_tpus()))'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                num_tpus = int(result.stdout.strip())
                return num_tpus > 0
        except:
            pass
        return False

    def has_tensorrt():
        """Check if TensorRT is available"""
        try:
            import tensorrt
            return True
        except ImportError:
            return False

    def has_hailo():
        """Check if Hailo device is available"""
        try:
            # Check if Hailo device exists
            if os.path.exists('/dev/hailo0'):
                return True
            # Alternative: try to use hailortcli
            result = subprocess.run(
                ['hailortcli', 'scan'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                return True
        except:
            pass
        return False


def check_container_device(container_name: str, device_path: str) -> bool:
    """
    Check if a device path exists inside a running container.
    
    Args:
        container_name: Name of the container to check
        device_path: Device path to verify (e.g., /dev/apex_0)
        
    Returns:
        True if device exists in container, False otherwise
    """
    try:
        client = docker.from_env()
        container = client.containers.get(container_name)
        if container.status != 'running':
            return False
        exit_code, _ = container.exec_run(f"test -e {device_path}")
        return exit_code == 0
    except (docker.errors.NotFound, docker.errors.APIError):
        return False


def requires_coral_tpu(func: Optional[Callable] = None, *, container_name: str = "security-nvr"):
    """
    Decorator to skip tests if Coral TPU is not available on host and in container.
    
    Args:
        func: Test function to decorate
        container_name: Name of container that should have the device
        
    Usage:
        @requires_coral_tpu
        def test_coral_inference():
            ...
            
        @requires_coral_tpu(container_name="custom-container")
        def test_custom_coral():
            ...
    """
    def decorator(test_func):
        @wraps(test_func)
        def wrapper(*args, **kwargs):
            if not has_coral_tpu():
                pytest.skip("Coral TPU not found on host")
            if not check_container_device(container_name, "/dev/apex_0"):
                # Also check USB Coral
                if not check_container_device(container_name, "/dev/bus/usb"):
                    pytest.skip(f"Coral TPU device not found in {container_name} container")
            return test_func(*args, **kwargs)
        return wrapper
    
    if func is None:
        # Decorator was called with arguments
        return decorator
    else:
        # Decorator was called without arguments
        return decorator(func)


def requires_tensorrt(func: Optional[Callable] = None, *, container_name: str = "security-nvr"):
    """
    Decorator to skip tests if NVIDIA GPU/TensorRT is not available.
    
    Args:
        func: Test function to decorate
        container_name: Name of container that should have GPU access
    """
    def decorator(test_func):
        @wraps(test_func)
        def wrapper(*args, **kwargs):
            if not has_tensorrt():
                pytest.skip("TensorRT/NVIDIA GPU not found on host")
            # Check for nvidia device
            if not check_container_device(container_name, "/dev/nvidia0"):
                # Also check for DRI devices
                if not check_container_device(container_name, "/dev/dri"):
                    pytest.skip(f"GPU device not found in {container_name} container")
            return test_func(*args, **kwargs)
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)


def requires_hailo(func: Optional[Callable] = None, *, container_name: str = "security-nvr"):
    """
    Decorator to skip tests if Hailo device is not available.
    
    Args:
        func: Test function to decorate
        container_name: Name of container that should have the device
    """
    def decorator(test_func):
        @wraps(test_func)
        def wrapper(*args, **kwargs):
            if not has_hailo():
                pytest.skip("Hailo device not found on host")
            if not check_container_device(container_name, "/dev/hailo0"):
                pytest.skip(f"Hailo device /dev/hailo0 not found in {container_name} container")
            return test_func(*args, **kwargs)
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)


@contextlib.contextmanager
def multi_camera_simulator(mqtt_client_factory: Callable, count: int, 
                          client_id_prefix: str = "sim-cam") -> List[mqtt.Client]:
    """
    Provide multiple connected MQTT clients for simulating cameras.
    
    Args:
        mqtt_client_factory: Factory function from conftest to create clients
        count: Number of camera clients to create
        client_id_prefix: Prefix for client IDs
        
    Yields:
        List of connected MQTT clients
        
    Usage:
        with multi_camera_simulator(mqtt_client_factory, count=3) as cameras:
            for i, camera_client in enumerate(cameras):
                camera_client.publish(topic, payload)
    """
    clients = []
    try:
        for i in range(count):
            client_id = f"{client_id_prefix}-{i}"
            client = mqtt_client_factory(client_id=client_id)
            clients.append(client)
        yield clients
    finally:
        # The mqtt_client_factory fixture handles cleanup automatically
        pass


def wait_for_consensus_trigger(mqtt_client: mqtt.Client, trigger_topic: str,
                              timeout: float = 10.0) -> Dict[str, Any]:
    """
    Helper to wait for fire consensus trigger message.
    
    Args:
        mqtt_client: Connected MQTT client
        trigger_topic: Topic to listen for trigger message
        timeout: Maximum time to wait
        
    Returns:
        Trigger message payload as dictionary
        
    Raises:
        TimeoutError: If no trigger received within timeout
    """
    with MqttMessageListener(mqtt_client, trigger_topic, timeout=timeout) as listener:
        payload = listener.wait_for_json_message()
        
        # Validate it's a proper trigger message
        if 'action' not in payload or payload['action'] != 'activate':
            raise ValueError(f"Invalid trigger message received: {payload}")
            
        return payload


class DockerHealthChecker:
    """Utilities for checking Docker container health before tests."""
    
    def __init__(self):
        self.client = docker.from_env()
    
    def is_container_healthy(self, container_name: str) -> bool:
        """
        Check if a container is running and healthy.
        
        Args:
            container_name: Name of the container
            
        Returns:
            True if container is healthy, False otherwise
        """
        try:
            container = self.client.containers.get(container_name)
            if container.status != 'running':
                return False
                
            # Check health status if available
            health = container.attrs.get('State', {}).get('Health', {})
            if health:
                return health.get('Status') == 'healthy'
                
            # If no health check, assume healthy if running
            return True
            
        except (docker.errors.NotFound, docker.errors.APIError):
            return False
    
    def wait_for_container(self, container_name: str, timeout: float = 60.0) -> bool:
        """
        Wait for a container to become healthy.
        
        Args:
            container_name: Name of the container
            timeout: Maximum time to wait
            
        Returns:
            True if container became healthy, False if timeout
        """
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.is_container_healthy(container_name):
                return True
            time.sleep(2)
            
        return False
    
    def ensure_containers_running(self, container_names: List[str]) -> None:
        """
        Ensure all required containers are running, skip test if not.
        
        Args:
            container_names: List of container names to check
            
        Raises:
            pytest.skip: If any container is not healthy
        """
        for name in container_names:
            if not self.is_container_healthy(name):
                pytest.skip(f"Container '{name}' is not running/healthy")


def requires_docker_containers(*container_names: str):
    """
    Decorator to ensure Docker containers are running before test.
    
    Args:
        container_names: Names of containers that must be running
        
    Usage:
        @requires_docker_containers("mqtt-broker", "security-nvr")
        def test_integration():
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            checker = DockerHealthChecker()
            checker.ensure_containers_running(list(container_names))
            return func(*args, **kwargs)
        return wrapper
    return decorator


def ensure_docker_available():
    """
    Ensure Docker daemon is available and running.
    
    Raises:
        pytest.skip: If Docker is not available
    """
    try:
        client = docker.from_env()
        # Try to ping Docker daemon
        client.ping()
    except (docker.errors.DockerException, Exception) as e:
        pytest.skip(f"Docker not available: {e}")


def requires_docker(func):
    """
    Decorator to ensure Docker is available before running test.
    
    Usage:
        @requires_docker
        def test_docker_integration():
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        ensure_docker_available()
        return func(*args, **kwargs)
    return wrapper


class DockerContainerManager:
    """Utilities for managing Docker containers in tests."""
    
    def __init__(self, client: docker.DockerClient = None, worker_id: str = 'master'):
        self.client = client or docker.from_env()
        self.worker_id = worker_id
        self.container_prefix = f"wf-{worker_id}"
        self.created_containers = []
        self.created_networks = []
        self.created_images = []
    
    def cleanup_old_container(self, container_name: str, timeout: int = 5, retry_count: int = 3) -> None:
        """
        Remove old container if it exists with improved error handling and retries.
        
        Args:
            container_name: Name of the container to remove
            timeout: Timeout for stopping the container
            retry_count: Number of retries for removal operations
        """
        for attempt in range(retry_count):
            try:
                old_container = self.client.containers.get(container_name)
                print(f"  Removing old container: {container_name} (attempt {attempt + 1})")
                
                # Stop container if running
                if old_container.status == 'running':
                    old_container.stop(timeout=timeout)
                    
                # Remove container
                old_container.remove(force=True)
                
                # Wait and verify removal
                time.sleep(1)
                
                # Verify container is actually removed
                try:
                    self.client.containers.get(container_name)
                    # Container still exists, continue trying
                    if attempt < retry_count - 1:
                        print(f"    Container still exists after removal, retrying...")
                        time.sleep(2)
                        continue
                except docker.errors.NotFound:
                    # Successfully removed
                    print(f"    Successfully removed container: {container_name}")
                    return
                    
            except docker.errors.NotFound:
                # Container doesn't exist, we're done
                return
            except docker.errors.APIError as e:
                # Handle specific API errors
                if "removal of container" in str(e) and "already in progress" in str(e):
                    print(f"    Container {container_name} already being removed by another process")
                    # Wait for the other process to finish
                    for wait_attempt in range(10):  # Wait up to 10 seconds
                        time.sleep(1)
                        try:
                            self.client.containers.get(container_name)
                        except docker.errors.NotFound:
                            print(f"    Container removal completed by other process")
                            return
                    # If we get here, removal is taking too long
                    if attempt < retry_count - 1:
                        print(f"    Removal taking too long, retrying...")
                        continue
                elif "No such container" in str(e):
                    # Container doesn't exist
                    return
                else:
                    print(f"    API error removing container: {e}")
                    if attempt < retry_count - 1:
                        time.sleep(2)
                        continue
            except Exception as e:
                print(f"    Error removing container: {e}")
                if attempt < retry_count - 1:
                    time.sleep(2)
                    continue
                    
        print(f"  Warning: Could not remove container {container_name} after {retry_count} attempts")
    
    def build_image_if_needed(self, image_tag: str, dockerfile_path: str, 
                             build_context: str = None) -> None:
        """
        Build Docker image if it doesn't exist.
        
        Args:
            image_tag: Tag for the image
            dockerfile_path: Path to Dockerfile
            build_context: Build context directory (defaults to current directory)
        """
        try:
            self.client.images.get(image_tag)
        except docker.errors.ImageNotFound:
            print(f"  Building image: {image_tag}")
            try:
                build_context = build_context or str(Path.cwd())
                image, build_logs = self.client.images.build(
                    path=build_context,
                    dockerfile=dockerfile_path,
                    tag=image_tag,
                    rm=True
                )
                self.created_images.append(image_tag)
                # Print build logs if verbose
                for log in build_logs:
                    if 'stream' in log:
                        print(f"    {log['stream'].strip()}")
            except Exception as e:
                print(f"  ✗ Failed to build image {image_tag}: {e}")
                raise
    
    def start_container(self, image: str, name: str, config: Dict[str, Any],
                       wait_timeout: int = 10, health_check_fn: Callable = None) -> docker.models.containers.Container:
        """
        Start a container with error handling and health checking.
        
        Args:
            image: Image to use
            name: Container name
            config: Container configuration dictionary
            wait_timeout: Time to wait for container to be ready
            health_check_fn: Optional function to check if container is healthy
            
        Returns:
            Started container instance
        """
        # Cleanup old container first
        self.cleanup_old_container(name)
        
        try:
            print(f"  Starting container: {name}")
            # Merge name into config
            config['name'] = name
            config['image'] = image
            
            # Add label for cleanup identification
            config.setdefault('labels', {})
            config['labels']['com.wildfire.test'] = 'true'
            
            # Add resource limits if not already specified
            if 'mem_limit' not in config:
                config['mem_limit'] = '512m'
            if 'cpu_quota' not in config:
                config['cpu_quota'] = 50000  # 50% of one CPU
            
            container = self.client.containers.run(**config)
            self.created_containers.append(container)
            
            # Wait for container to be ready
            print(f"  Waiting for {name} to initialize...")
            
            # Progressive health check with retries
            start_time = time.time()
            health_check_passed = False
            last_logs = ""
            
            while time.time() - start_time < wait_timeout:
                # Check container status first
                container.reload()
                if container.status != 'running':
                    logs = container.logs(tail=100).decode()
                    raise RuntimeError(f"Container {name} exited. Logs:\n{logs}")
                
                # Run custom health check if provided
                if health_check_fn:
                    try:
                        if health_check_fn(container):
                            health_check_passed = True
                            break
                        else:
                            # Store logs for potential error reporting
                            last_logs = container.logs(tail=100).decode()
                    except Exception as e:
                        print(f"    Health check error: {e}")
                        last_logs = container.logs(tail=100).decode()
                else:
                    # No health check function, just wait a bit
                    time.sleep(5)
                    health_check_passed = True
                    break
                
                # Wait before next check
                time.sleep(2)
            
            if not health_check_passed and health_check_fn:
                raise RuntimeError(f"Container {name} health check failed after {wait_timeout}s. Logs:\n{last_logs}")
            
            print(f"  ✓ {name} is running")
            return container
            
        except Exception as e:
            print(f"  ✗ Failed to start {name}: {e}")
            # Try to get logs for debugging
            try:
                if 'container' in locals():
                    logs = container.logs(tail=100).decode()
                    print(f"  Container logs:\n{logs}")
            except:
                pass
            raise
    
    def wait_for_container_log(self, container: docker.models.containers.Container,
                              log_pattern: str, timeout: int = 60) -> bool:
        """
        Wait for a specific log pattern to appear in container logs.
        
        Args:
            container: Container to monitor
            log_pattern: String pattern to look for in logs
            timeout: Maximum time to wait
            
        Returns:
            True if pattern found, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            logs = container.logs().decode()
            if log_pattern in logs:
                return True
            time.sleep(2)
            
        return False
    
    def wait_for_healthy(self, container_name: str, timeout: int = 60) -> bool:
        """
        Wait for a container to become healthy based on its health check.
        
        Args:
            container_name: Name of the container to check
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if container became healthy, False if timeout
            
        Raises:
            RuntimeError: If container exits or fails
        """
        start_time = time.time()
        last_health_status = None
        
        while time.time() - start_time < timeout:
            try:
                container = self.client.containers.get(container_name)
                container.reload()
                
                # Check if container is still running
                if container.status == 'exited':
                    logs = container.logs(tail=100).decode()
                    raise RuntimeError(f"Container {container_name} exited. Last logs:\n{logs}")
                
                # Check health status
                health = container.attrs.get('State', {}).get('Health', {})
                if health:
                    health_status = health.get('Status', 'none')
                    
                    # Log status changes
                    if health_status != last_health_status:
                        print(f"  Container {container_name} health: {health_status}")
                        last_health_status = health_status
                    
                    if health_status == 'healthy':
                        return True
                    elif health_status == 'unhealthy':
                        # Get last health check log
                        logs = health.get('Log', [])
                        if logs:
                            last_check = logs[-1]
                            output = last_check.get('Output', 'No output')
                            print(f"  Health check failed: {output}")
                else:
                    # No health check defined, check if running
                    if container.status == 'running':
                        # Give it a few seconds to stabilize
                        time.sleep(5)
                        return True
                
                time.sleep(2)
                
            except docker.errors.NotFound:
                raise RuntimeError(f"Container {container_name} not found")
            except Exception as e:
                print(f"  Error checking container health: {e}")
                time.sleep(2)
        
        # Timeout - get final logs for debugging
        try:
            container = self.client.containers.get(container_name)
            logs = container.logs(tail=50).decode()
            print(f"  Container {container_name} logs at timeout:\n{logs}")
        except:
            pass
            
        return False
    
    def get_container_name(self, service: str) -> str:
        """
        Get unique container name for a service.
        
        Args:
            service: Service name
            
        Returns:
            Worker-prefixed container name
        """
        return f"{self.container_prefix}-{service}"
    
    def get_network_name(self) -> str:
        """
        Get unique network name for this worker.
        
        Returns:
            Worker-prefixed network name
        """
        return f"{self.container_prefix}-network"
    
    def create_test_network(self, network_name: str = None) -> docker.models.networks.Network:
        """
        Create a test network for containers.
        
        Args:
            network_name: Name for the network (uses worker-based name if not provided)
            
        Returns:
            Created network instance
        """
        # Use worker-based name if not provided
        if network_name is None:
            network_name = self.get_network_name()
            
        # Remove old network if exists
        try:
            old_net = self.client.networks.get(network_name)
            old_net.remove()
        except docker.errors.NotFound:
            pass
        
        network = self.client.networks.create(
            name=network_name,
            driver="bridge"
        )
        self.created_networks.append(network)
        return network
    
    def cleanup(self, force: bool = False):
        """Clean up all created resources with improved error handling."""
        import logging
        logger = logging.getLogger(__name__)
        
        # Stop and remove containers with better error handling
        for container in self.created_containers[:]:  # Create copy to avoid modification during iteration
            container_name = "unknown"
            try:
                if isinstance(container, str):
                    container_name = container
                    try:
                        container = self.client.containers.get(container)
                    except docker.errors.NotFound:
                        # Container already removed, remove from list
                        try:
                            self.created_containers.remove(container_name)
                        except ValueError:
                            pass
                        continue
                else:
                    container_name = getattr(container, 'name', 'unknown')
                
                # Reload container status
                try:
                    container.reload()
                except docker.errors.NotFound:
                    # Container no longer exists
                    try:
                        self.created_containers.remove(container)
                    except ValueError:
                        pass
                    continue
                        
                # Force stop with shorter timeout
                if container.status == 'running':
                    try:
                        container.stop(timeout=3)
                        logger.debug(f"Stopped container {container_name}")
                    except Exception as e:
                        logger.debug(f"Error stopping container {container_name}: {e}")
                        if force:
                            try:
                                container.kill()
                                logger.debug(f"Force killed container {container_name}")
                            except Exception:
                                pass
                    
                # Remove container with retry logic
                removal_success = False
                for attempt in range(3):
                    try:
                        container.remove(force=True)
                        logger.debug(f"Removed container {container_name}")
                        removal_success = True
                        break
                    except docker.errors.APIError as e:
                        if "removal of container" in str(e) and "already in progress" in str(e):
                            logger.debug(f"Container {container_name} removal already in progress")
                            # Wait for completion
                            for wait_attempt in range(5):
                                time.sleep(1)
                                try:
                                    self.client.containers.get(container_name)
                                except docker.errors.NotFound:
                                    logger.debug(f"Container {container_name} removal completed")
                                    removal_success = True
                                    break
                            if removal_success:
                                break
                        elif "No such container" in str(e):
                            removal_success = True
                            break
                        else:
                            logger.debug(f"Attempt {attempt + 1}: Error removing container {container_name}: {e}")
                            if attempt < 2:
                                time.sleep(1)
                    except docker.errors.NotFound:
                        removal_success = True
                        break
                    except Exception as e:
                        logger.debug(f"Attempt {attempt + 1}: Error removing container {container_name}: {e}")
                        if attempt < 2:
                            time.sleep(1)
                
                if not removal_success:
                    logger.warning(f"Could not remove container {container_name} after 3 attempts")
                
                # Remove from tracking list
                try:
                    self.created_containers.remove(container)
                except ValueError:
                    pass
                
            except docker.errors.NotFound:
                # Container already removed, remove from list
                try:
                    self.created_containers.remove(container)
                except ValueError:
                    pass
            except Exception as e:
                logger.warning(f"Error cleaning up container {container_name}: {e}")
        
        # Clear the list
        self.created_containers.clear()
        
        # Remove networks with better error handling
        for network in self.created_networks[:]:  # Create copy
            network_name = "unknown"
            try:
                if isinstance(network, str):
                    network_name = network
                    try:
                        network = self.client.networks.get(network)
                    except docker.errors.NotFound:
                        try:
                            self.created_networks.remove(network_name)
                        except ValueError:
                            pass
                        continue
                else:
                    network_name = getattr(network, 'name', 'unknown')
                
                # Disconnect containers from network first
                try:
                    network.reload()
                    for container in network.containers:
                        try:
                            network.disconnect(container, force=True)
                            logger.debug(f"Disconnected container {container.name} from network {network_name}")
                        except Exception as e:
                            logger.debug(f"Error disconnecting container from network: {e}")
                except Exception as e:
                    logger.debug(f"Error preparing network {network_name} for removal: {e}")
                
                # Remove network with retry
                for attempt in range(3):
                    try:
                        network.remove()
                        logger.debug(f"Removed network {network_name}")
                        break
                    except docker.errors.APIError as e:
                        if "has active endpoints" in str(e):
                            logger.debug(f"Network {network_name} has active endpoints, retrying...")
                            if attempt < 2:
                                time.sleep(2)
                                continue
                        elif "not found" in str(e).lower():
                            break  # Network already removed
                        else:
                            logger.debug(f"Error removing network {network_name}: {e}")
                            if attempt < 2:
                                time.sleep(1)
                    except docker.errors.NotFound:
                        break  # Network already removed
                    except Exception as e:
                        logger.debug(f"Error removing network {network_name}: {e}")
                        if attempt < 2:
                            time.sleep(1)
                
                # Remove from tracking list
                try:
                    self.created_networks.remove(network)
                except ValueError:
                    pass
                
            except docker.errors.NotFound:
                try:
                    self.created_networks.remove(network)
                except ValueError:
                    pass
            except Exception as e:
                logger.warning(f"Error cleaning up network {network_name}: {e}")
        
        # Clear the list
        self.created_networks.clear()
        
        # Remove images (optional - usually keep for caching)
        # for image_tag in self.created_images:
        #     try:
        #         self.client.images.remove(image_tag)
        #     except:
        #         pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def create_test_frigate_config(detector: str = 'cpu', num_detectors: int = 1,
                              mqtt_host: str = 'localhost', mqtt_port: int = 1883) -> Dict[str, Any]:
    """
    Create a test Frigate configuration.
    
    Args:
        detector: Type of detector (cpu, coral, tensorrt, hailo)
        num_detectors: Number of detectors to configure
        mqtt_host: MQTT broker host
        mqtt_port: MQTT broker port
        
    Returns:
        Frigate configuration dictionary
    """
    config = {
        'mqtt': {
            'enabled': True,
            'host': mqtt_host,
            'port': mqtt_port,
            'topic_prefix': 'frigate'
        },
        'detectors': {},
        'cameras': {},
        'model': {
            'width': 320,
            'height': 320,
            'labelmap_path': '/config/model_cache/labelmap.txt'
        }
    }
    
    # Configure detectors based on type
    if detector == 'coral':
        config['model']['path'] = '/config/model_cache/yolov8n_320_edgetpu.tflite'
        for i in range(num_detectors):
            config['detectors'][f'coral{i}'] = {
                'type': 'edgetpu',
                'device': f'pci:{i}' if i > 0 else 'pci'
            }
    elif detector == 'tensorrt':
        config['detectors']['tensorrt'] = {
            'type': 'tensorrt',
            'device': 0
        }
    elif detector == 'hailo':
        config['detectors']['hailo'] = {
            'type': 'hailo',
            'device': '/dev/hailo0'
        }
    else:
        config['detectors']['cpu'] = {
            'type': 'cpu',
            'num_threads': 4
        }
    
    # Add disabled test cameras to avoid RTSP errors
    for i in range(3):
        config['cameras'][f'test_camera_{i}'] = {
            'enabled': False,
            'ffmpeg': {
                'inputs': [{
                    'path': f'rtsp://localhost:554/test{i}',
                    'roles': ['detect']
                }]
            },
            'detect': {
                'width': 640,
                'height': 640,
                'fps': 5
            }
        }
    
    return config


def prepare_frigate_test_environment(config_dir: str = '/tmp/e2e_frigate_config',
                                   detector: str = 'cpu') -> str:
    """
    Prepare the test environment for Frigate.
    
    Args:
        config_dir: Directory to store config and models
        detector: Type of detector being used
        
    Returns:
        str: The actual config directory used (may differ if permissions required a temp dir)
    """
    import yaml
    import shutil
    
    # Create directory structure with proper permissions
    config_path = Path(config_dir)
    
    # If we can't write to the directory, use a temp directory instead
    try:
        config_path.mkdir(exist_ok=True, mode=0o755)
        # Test if we can write
        test_file = config_path / '.test_write'
        test_file.touch()
        test_file.unlink()
    except (PermissionError, OSError):
        # Use a temporary directory instead
        import tempfile
        config_dir = tempfile.mkdtemp(prefix='e2e_frigate_config_')
        config_path = Path(config_dir)
        config_path.mkdir(exist_ok=True, mode=0o755)
    
    model_cache_dir = config_path / 'model_cache'
    model_cache_dir.mkdir(exist_ok=True, mode=0o755)
    
    # Create labelmap
    labelmap_content = """0  person
1  fire
2  smoke
3  flame
"""
    with open(model_cache_dir / 'labelmap.txt', 'w') as f:
        f.write(labelmap_content)
    
    # Copy model if available for Coral
    if detector == 'coral':
        possible_models = [
            'converted_models/yolov8n_320_edgetpu.tflite',
            'converted_models/yolov8n_416_edgetpu.tflite',
            'converted_models/fire_320_edgetpu.tflite',
            'models/yolov8n_320_edgetpu.tflite'
        ]
        
        for model_path in possible_models:
            if Path(model_path).exists():
                shutil.copy(model_path, model_cache_dir / 'yolov8n_320_edgetpu.tflite')
                print(f"  Copied EdgeTPU model from {model_path}")
                break
        else:
            print("  Warning: No EdgeTPU model found, Frigate will use CPU fallback")
    
    # Return the actual config directory used
    return str(config_path)


# Utility for setting up test environment
@contextlib.contextmanager
def mqtt_test_environment(test_mqtt_broker, monkeypatch):
    """
    Context manager to set up MQTT environment variables for testing.
    
    Args:
        test_mqtt_broker: The test broker fixture
        monkeypatch: pytest monkeypatch fixture
        
    Usage:
        with mqtt_test_environment(test_mqtt_broker, monkeypatch):
            from consensus import FireConsensus
            consensus = FireConsensus()  # Will connect to test broker
    """
    # Set environment variables
    monkeypatch.setenv('MQTT_BROKER', test_mqtt_broker.host)
    monkeypatch.setenv('MQTT_PORT', str(test_mqtt_broker.port))
    monkeypatch.setenv('MQTT_TLS', 'false')
    
    # Optional: Set other common environment variables
    monkeypatch.setenv('LOG_LEVEL', 'DEBUG')
    
    yield
    
    # Cleanup happens automatically with monkeypatch


class ParallelTestContext:
    """
    Comprehensive context for parallel-safe integration testing.
    
    Provides:
    - Unique topic namespaces per worker
    - Dynamic port allocation
    - Worker-based container naming
    - Service configuration with isolation
    """
    
    def __init__(self, worker_id: str = 'master'):
        """
        Initialize parallel test context.
        
        Args:
            worker_id: pytest-xdist worker ID
        """
        self.worker_id = worker_id
        self.namespace = TopicNamespace(worker_id)
        
        # Port allocation
        self.base_mqtt_port = 20000
        self.mqtt_port = self._allocate_port(self.base_mqtt_port)
        self.mqtt_tls_port = self._allocate_port(self.base_mqtt_port + 5000)
        
        # Container naming
        self.container_prefix = f"wf-{worker_id}"
        
    def _allocate_port(self, base_port: int) -> int:
        """Allocate a port based on worker ID."""
        if self.worker_id == 'master':
            return base_port
        elif self.worker_id.startswith('gw'):
            try:
                worker_num = int(self.worker_id[2:])
                return base_port + (worker_num * 100)
            except ValueError:
                # Use hash of worker_id for consistent port allocation
                import hashlib
                hash_num = int(hashlib.md5(self.worker_id.encode()).hexdigest()[:4], 16)
                return base_port + (hash_num % 1000) * 10
        else:
            return base_port
    
    def container_name(self, service: str) -> str:
        """Get unique container name for a service."""
        return f"{self.container_prefix}-{service}"
    
    def network_name(self) -> str:
        """Get unique network name."""
        return f"{self.container_prefix}-network"
    
    def get_mqtt_config(self) -> Dict[str, Any]:
        """Get MQTT configuration for this worker."""
        return {
            'host': 'localhost',
            'port': self.mqtt_port,
            'tls_port': self.mqtt_tls_port,
            'topic_namespace': self.namespace.namespace
        }
    
    def get_service_env(self, service: str) -> Dict[str, str]:
        """
        Get environment variables for a service with isolation.
        
        Args:
            service: Service name (camera_detector, fire_consensus, etc.)
            
        Returns:
            Environment variables dict
        """
        base_env = {
            'MQTT_BROKER': 'localhost',
            'MQTT_PORT': str(self.mqtt_port),
            'MQTT_TLS': 'false',
            'MQTT_CLIENT_ID': f'{service}_{self.worker_id}',
            'TOPIC_PREFIX': self.namespace.namespace,
            'LOG_LEVEL': 'DEBUG'
        }
        
        # Service-specific configuration
        if service == 'camera_detector':
            base_env.update({
                'CAMERA_CREDENTIALS': 'admin:S3thrule',
                'DISCOVERY_INTERVAL': '30',
                'FRIGATE_CONFIG_PATH': f'/tmp/{self.worker_id}/frigate_config.yml'
            })
        elif service == 'fire_consensus':
            base_env.update({
                'CONSENSUS_THRESHOLD': '2',
                'TIME_WINDOW': '30',
                'MIN_CONFIDENCE': '0.6'
            })
        elif service == 'gpio_trigger':
            base_env.update({
                'GPIO_SIMULATION': 'true',
                'MAX_ENGINE_RUNTIME': '30'
            })
            
        return base_env
    
    def wrap_client(self, client: mqtt.Client) -> NamespacedMQTTClient:
        """Wrap an MQTT client with namespace support."""
        return NamespacedMQTTClient(client, self.namespace)
    
    def translate_topic(self, topic: str) -> str:
        """Translate a topic to namespaced version."""
        return self.namespace.topic(topic)
    
    def strip_topic(self, namespaced_topic: str) -> str:
        """Strip namespace from a topic."""
        return self.namespace.strip(namespaced_topic)


# Pytest fixture for parallel test context
@pytest.fixture
def parallel_test_context(worker_id):
    """Provide parallel test context for isolation."""
    return ParallelTestContext(worker_id)


# Pytest fixture for Docker container manager with worker isolation
@pytest.fixture
def docker_container_manager(worker_id):
    """Provide Docker container manager with worker isolation."""
    manager = DockerContainerManager(worker_id=worker_id)
    yield manager
    manager.cleanup()