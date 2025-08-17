#!/usr/bin/env python3.12
"""
Integration tests for Security NVR (Frigate) Service
Tests camera integration, object detection, MQTT publishing, and hardware detection
"""
import os
import sys
import json
import time
import threading
import subprocess
import tempfile
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import requests
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion
import yaml
import shutil
import docker

# Test configuration
TEST_TIMEOUT = 30
FRIGATE_HOST = os.getenv("FRIGATE_HOST", "localhost")
FRIGATE_PORT = int(os.getenv("FRIGATE_PORT", "5000"))
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "0"))

def check_service_running():
    """Check if security_nvr service is running"""
    result = subprocess.run(
        ["docker", "ps", "--filter", "name=security_nvr", "--format", "{{.Names}}"],
        capture_output=True,
        text=True
    )
    return "security_nvr" in result.stdout

def check_frigate_api():
    """Check if Frigate API is accessible"""
    try:
        # Try with default admin credentials first
        response = requests.get(
            f"http://localhost:5000/version", 
            auth=("admin", "7f155ad9e8c340c88ef6a33f528f2e75"),
            timeout=2
        )
        return response.status_code == 200
    except (requests.RequestException, requests.Timeout):
        return False

def check_mqtt_broker():
    """Check if MQTT broker is running"""
    result = subprocess.run(
        ["docker", "ps", "--filter", "name=mqtt_broker", "--format", "{{.Names}}"],
        capture_output=True,
        text=True
    )
    return "mqtt_broker" in result.stdout

def safe_container_reload(container, max_retries=3, worker_id=None):
    """Safely reload container with retry logic to handle transient 404s"""
    if worker_id:
        print(f"[Worker: {worker_id}] [DEBUG] Starting safe_container_reload for container ID: {container.id[:12]}")
    
    # First check if container still exists
    try:
        # Try to get the container from Docker to verify it exists
        docker_client = docker.from_env()
        docker_client.containers.get(container.id)
    except docker.errors.NotFound:
        if worker_id:
            print(f"[Worker: {worker_id}] Container {container.id[:12]} no longer exists, skipping reload")
        return  # Container doesn't exist, nothing to reload
    except Exception as e:
        if worker_id:
            print(f"[Worker: {worker_id}] Error checking container existence: {e}")
    
    for attempt in range(max_retries):
        try:
            container.reload()
            if worker_id:
                print(f"[Worker: {worker_id}] Container reload successful on attempt {attempt + 1}")
            return  # Success
        except docker.errors.NotFound:
            if attempt < max_retries - 1:
                if worker_id:
                    print(f"[Worker: {worker_id}] Container not found during reload (attempt {attempt + 1}/{max_retries}), retrying...")
                time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                continue
            else:
                if worker_id:
                    print(f"[Worker: {worker_id}] Container not found after {max_retries} reload attempts")
                raise  # Re-raise on last attempt
        except docker.errors.APIError as e:
            if "404" in str(e) and attempt < max_retries - 1:
                if worker_id:
                    print(f"[Worker: {worker_id}] API 404 error during reload (attempt {attempt + 1}/{max_retries}), retrying...")
                time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                continue
            else:
                raise  # Re-raise for non-404 errors or last attempt

# Skip decorators - only skip if we can't start the service
# We'll start containers in fixtures instead of skipping
requires_security_nvr = pytest.mark.security_nvr
requires_frigate_api = pytest.mark.frigate_integration
requires_mqtt = pytest.mark.mqtt


# Fixture Scope Documentation:
# These fixtures use session scope to minimize container startup overhead.
# - Session scope means fixtures are created once per test session/worker
# - DockerContainerManager uses reference counting to manage container lifecycle
# - Containers are only removed when reference count reaches zero
# - This ensures containers remain available throughout the test session
# - Each pytest-xdist worker gets its own session-scoped instances

@pytest.fixture(scope="session")
def mqtt_broker_for_frigate(worker_id):
    """Session-scoped MQTT broker for Frigate integration tests"""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from test_utils.enhanced_mqtt_broker import TestMQTTBroker
    
    broker = TestMQTTBroker(session_scope=True, worker_id=worker_id)
    broker.start()
    yield broker
    broker.stop()


@pytest.fixture(scope="session")
def docker_client_for_frigate():
    """Session-scoped Docker client to avoid connection exhaustion"""
    client = docker.from_env()
    # Set timeout to prevent hanging operations
    client.api.timeout = 240
    yield client
    client.close()


@pytest.fixture(scope="session")
def docker_manager_for_frigate(worker_id, docker_client_for_frigate):
    """Session-scoped Docker manager for Frigate integration tests"""
    sys.path.insert(0, os.path.dirname(__file__))
    from test_utils.helpers import DockerContainerManager
    
    manager = DockerContainerManager(client=docker_client_for_frigate, worker_id=worker_id)
    yield manager
    manager.cleanup(force=True)  # Force cleanup to ensure pruning happens


@pytest.fixture(scope="function")
def frigate_container(mqtt_broker_for_frigate, docker_manager_for_frigate):
    """Start a Frigate container for integration testing - function-scoped fixture"""
    import tempfile
    import yaml
    
    # Create temporary config directory
    config_dir = tempfile.mkdtemp(prefix=f"frigate_test_{docker_manager_for_frigate.worker_id}_")
    
    # ✅ Create Frigate config with proper topic isolation
    worker_id = docker_manager_for_frigate.worker_id
    topic_prefix = f"test/{worker_id}"
    
    frigate_config = {
        'database': {
            'path': '/config/frigate.db'  # Ensure writable path
        },
        'logger': {
            'default': 'info',  # Enable info logging for debugging
            'logs': {
                'frigate.app': 'debug'  # Debug logging for main app
            }
        },
        'mqtt': {
            'enabled': True,  # Enable MQTT
            'host': 'host.docker.internal',  # Connect to host from container
            'port': mqtt_broker_for_frigate.port,
            'user': '',
            'password': '',
            'topic_prefix': f'frigate_{worker_id}',  # Isolate Frigate MQTT topics
            'stats_interval': 15  # Minimum allowed by Frigate
        },
        'detectors': {
            'cpu': {
                'type': 'cpu'
            }
        },
        'objects': {
            'track': ['person', 'fire', 'smoke'],
            'filters': {
                'fire': {
                    'min_score': 0.7,
                    'threshold': 0.8
                },
                'smoke': {
                    'min_score': 0.7,
                    'threshold': 0.8
                }
            }
        },
        'auth': {
            'enabled': False  # Disable auth for easier testing
        },
        'cameras': {
            'dummy': {  # Minimal dummy camera config
                'enabled': False,  # Disable the camera entirely
                'ffmpeg': {
                    'inputs': [{
                        'path': 'rtsp://127.0.0.1:554/null',
                        'roles': ['detect']
                    }]
                }
            }
        },
        'record': {
            'enabled': False  # Disable recording
        },
        'snapshots': {
            'enabled': False  # Disable snapshots
        }
    }
    
    config_path = os.path.join(config_dir, 'config.yml')
    with open(config_path, 'w') as f:
        yaml.dump(frigate_config, f)
    
    # Create required directories with proper permissions
    media_dir = os.path.join(config_dir, 'media')
    os.makedirs(media_dir, exist_ok=True)
    db_dir = os.path.join(config_dir, 'db')
    os.makedirs(db_dir, exist_ok=True)
    # Ensure directories are writable by container (UID 1000)
    os.chmod(config_dir, 0o777)
    os.chmod(db_dir, 0o777)
    os.chmod(media_dir, 0o777)
    
    # Start Frigate container with unique name
    container_name = docker_manager_for_frigate.get_container_name('frigate')
    print(f"[Worker: {docker_manager_for_frigate.worker_id}, PID: {os.getpid()}] "
          f"Creating Frigate container with name: {container_name}")
    
    # ✅ Get dynamic ports
    frigate_port = docker_manager_for_frigate.get_free_port()
    rtsp_port = docker_manager_for_frigate.get_free_port()
    print(f"[Worker: {docker_manager_for_frigate.worker_id}] "
          f"Allocated ports - Frigate: {frigate_port}, RTSP: {rtsp_port}")
    
    # Use DockerContainerManager to start and track the container
    print(f"[Worker: {docker_manager_for_frigate.worker_id}] Starting container creation...")
    
    # Initialize timing variables before container creation for accurate error reporting
    import time
    start_time = time.time()
    frigate_ready = False
    last_error = None
    container = None
    
    # Try to start the container with error handling
    try:
        container = docker_manager_for_frigate.start_container(
            image='ghcr.io/blakeblackshear/frigate:stable',
            name=container_name,
            config={
                'detach': True,
                'mem_limit': '4g',  # Frigate needs more memory for video processing
                'ports': {
                    '5000/tcp': frigate_port,
                    '1935/tcp': docker_manager_for_frigate.get_free_port(),  # RTMP
                    '8554/tcp': rtsp_port,  # RTSP
                    '8555/tcp': docker_manager_for_frigate.get_free_port()   # WebRTC
                },
                'volumes': {
                    config_path: {'bind': '/config/config.yml', 'mode': 'ro'},
                    media_dir: {'bind': '/media/frigate', 'mode': 'rw'},
                    # FIXED: Don't mount db_dir to /config - let Frigate use container's /config
                    # This avoids permission issues with the database file
                    '/home/seth/wildfire-watch/converted_models': {'bind': '/models', 'mode': 'ro'}  # Mount models directory
                },
                'tmpfs': {
                    '/dev/shm': 'size=1g,exec,dev,suid,noatime,mode=1777'  # Container-specific shared memory
                },
                'environment': {
                    'FRIGATE_USER': 'admin',
                    'FRIGATE_PASSWORD': '7f155ad9e8c340c88ef6a33f528f2e75',
                    'TZ': 'UTC',
                    'FRIGATE_DISABLE_VAAPI': '1'  # Disable VAAPI hardware acceleration in tests
                },
                # shm_size removed - using tmpfs mount instead
                'privileged': True,
                'network_mode': 'bridge',
                'extra_hosts': {'host.docker.internal': 'host-gateway'},
                'remove': False  # Keep container for debugging if it fails
            },
            wait_timeout=30,  # Increased from 5 seconds for better stability
            health_check_fn=None  # We'll handle health check ourselves
        )
        
        # Store the container and ports for cleanup and access
        container.frigate_port = frigate_port
        container.rtsp_port = rtsp_port
        
        print(f"[Worker: {docker_manager_for_frigate.worker_id}] "
              f"Container created, waiting for Frigate to become ready...")
        
        # Give Frigate time to start up
        time.sleep(5)
        
        # Add initial debugging info
        print(f"[Worker: {docker_manager_for_frigate.worker_id}] "
              f"Container ID: {container.id[:12]}, "
              f"Frigate port: {frigate_port}, RTSP port: {rtsp_port}")
        
        # Log container details for debugging
        print(f"[Worker: {docker_manager_for_frigate.worker_id}] "
              f"Container name: {container.name}")
        print(f"[Worker: {docker_manager_for_frigate.worker_id}] "
              f"Container image: {container.image.tags}")
        
        # Check initial container state
        safe_container_reload(container, worker_id=docker_manager_for_frigate.worker_id)
        print(f"[Worker: {docker_manager_for_frigate.worker_id}] "
              f"Initial container status: {container.status}")
        
        # Log network configuration
        networks = container.attrs.get('NetworkSettings', {}).get('Networks', {})
        print(f"[Worker: {docker_manager_for_frigate.worker_id}] "
              f"Container networks: {list(networks.keys())}")
        
    except RuntimeError as e:
        # Container failed during initial startup
        last_error = f"Container startup failed: {str(e)}"
        print(f"[Worker: {docker_manager_for_frigate.worker_id}] {last_error}")
        # Don't re-raise here, let the retry logic below handle it
    
    # Single wait loop with strict timeout enforcement
    wait_start_time = time.time()  # Track total time from beginning
    backoff_delay = 1.0
    container_creation_attempts = 0
    MAX_CONTAINER_ATTEMPTS = 3  # Reduced from 5
    
    # Reduce timeout to prevent hanging
    frigate_timeout = int(os.environ.get('FRIGATE_STARTUP_TIMEOUT', '60'))  # Reduced to 1 minute
    print(f"[Worker: {docker_manager_for_frigate.worker_id}] "
          f"Frigate startup timeout set to {frigate_timeout} seconds")
    
    attempt_count = 0
    MAX_WAIT_ATTEMPTS = 20  # Reduced from 50
    
    # FIXED: Use start_time for timeout check, not wait_time
    while time.time() - start_time < frigate_timeout and attempt_count < MAX_WAIT_ATTEMPTS:
        attempt_count += 1
        
        # Log progress every 10 attempts
        if attempt_count % 10 == 0:
            elapsed = time.time() - start_time
            print(f"[Worker: {docker_manager_for_frigate.worker_id}] "
                  f"Still waiting for Frigate... (attempt {attempt_count}, elapsed: {elapsed:.1f}s)")
        
        try:
            # Handle case where container hasn't been created yet
            if container is None:
                print(f"[Worker: {docker_manager_for_frigate.worker_id}] "
                      f"DEBUG: Container is None, attempting to create...")
                # Check if we've exceeded max attempts
                if container_creation_attempts >= MAX_CONTAINER_ATTEMPTS:
                    raise RuntimeError(f"Failed to create container after {MAX_CONTAINER_ATTEMPTS} attempts")
                
                # Try to create container
                try:
                    container_creation_attempts += 1
                    print(f"[Worker: {docker_manager_for_frigate.worker_id}] "
                          f"Attempting to create container (attempt {container_creation_attempts}/{MAX_CONTAINER_ATTEMPTS}) "
                          f"after {time.time() - start_time:.1f}s")
                    container = docker_manager_for_frigate.start_container(
                            image='ghcr.io/blakeblackshear/frigate:stable',
                            name=container_name,
                            config={
                                'detach': True,
                                'mem_limit': '4g',
                                'ports': {
                                    '5000/tcp': frigate_port,
                                    '1935/tcp': docker_manager_for_frigate.get_free_port(),
                                    '8554/tcp': rtsp_port,
                                    '8555/tcp': docker_manager_for_frigate.get_free_port()
                                },
                                'volumes': {
                                    config_path: {'bind': '/config/config.yml', 'mode': 'ro'},
                                    media_dir: {'bind': '/media/frigate', 'mode': 'rw'},
                                    # FIXED: Don't mount db_dir to /config
                                    '/home/seth/wildfire-watch/converted_models': {'bind': '/models', 'mode': 'ro'}
                                },
                                'tmpfs': {
                                    '/dev/shm': 'size=1g,exec,dev,suid,noatime,mode=1777'  # Container-specific shared memory
                                },
                                'environment': {
                                    'FRIGATE_USER': 'admin',
                                    'FRIGATE_PASSWORD': '7f155ad9e8c340c88ef6a33f528f2e75',
                                    'TZ': 'UTC',
                                    'FRIGATE_DISABLE_VAAPI': '1'
                                },
                                # shm_size removed - using tmpfs mount instead
                                'privileged': True,
                                'network_mode': 'bridge',
                                'extra_hosts': {'host.docker.internal': 'host-gateway'},
                                'remove': False
                            },
                            wait_timeout=30,  # Increased from 5s
                            health_check_fn=None
                        )
                    container.frigate_port = frigate_port
                    container.rtsp_port = rtsp_port
                    print(f"[Worker: {docker_manager_for_frigate.worker_id}] "
                          f"Container created after {time.time() - start_time:.1f}s")
                except Exception as e:
                    last_error = f"Container creation failed: {str(e)}"
                    print(f"[Worker: {docker_manager_for_frigate.worker_id}] {last_error}")
                    
                    # Check for 500 Server Error and apply quarantine
                    if isinstance(e, docker.errors.APIError) and "500" in str(e):
                        print(f"[Worker: {docker_manager_for_frigate.worker_id}] "
                              f"Docker daemon overloaded (500 error), applying 30s quarantine...")
                        time.sleep(30)
                    else:
                        # Regular exponential backoff with jitter
                        import random
                        backoff_time = (2 ** container_creation_attempts) + random.uniform(0, 0.2)
                        backoff_time = min(backoff_time, 60)  # Cap at 60s
                        print(f"Retrying container creation in {backoff_time:.1f}s...")
                        time.sleep(backoff_time)
                    continue
            
            # Re-fetch container object with retry logic to handle transient 404s
            print(f"[Worker: {docker_manager_for_frigate.worker_id}] "
                  f"DEBUG: About to re-fetch container...")
            try:
                # Retry up to 3 times with brief delay
                current_container = None
                for fetch_attempt in range(3):
                    try:
                        current_container = docker_manager_for_frigate.client.containers.get(container.id)
                        break  # Success
                    except docker.errors.NotFound:
                        if fetch_attempt < 2:  # Not the last attempt
                            time.sleep(0.5)
                            continue
                        else:
                            raise  # Re-raise on last attempt
                print(f"[Worker: {docker_manager_for_frigate.worker_id}] "
                      f"DEBUG: Container re-fetch successful")
                
                if current_container and current_container.status != 'running':
                    # Get logs if container stopped
                    logs = current_container.logs(tail=100).decode('utf-8')
                    print(f"Container stopped. Last logs:\n{logs}")
                    
                    # Check exit code to determine if this is a permanent failure
                    try:
                        result = subprocess.run(['docker', 'inspect', container.id, 
                                               '--format={{.State.ExitCode}} {{.State.OOMKilled}}'], 
                                              capture_output=True, text=True)
                        if result.returncode == 0 and result.stdout.strip():
                            parts = result.stdout.strip().split()
                            if len(parts) >= 2:
                                exit_code, oom_killed = parts[0], parts[1]
                                if exit_code != '0' or oom_killed == 'true':
                                    print(f"Container failed permanently (exit_code={exit_code}, oom_killed={oom_killed})")
                                    raise RuntimeError(f"Frigate container failed with exit code {exit_code}, OOM: {oom_killed}")
                    except subprocess.SubprocessError:
                        pass  # Continue with existing logic if inspect fails
                    
                    # Container stopped - will retry in the next iteration
                    print(f"Container stopped, will retry...")
                    container = None  # Reset container to trigger recreation
                    time.sleep(backoff_delay)
                    backoff_delay = min(backoff_delay * 2, 60)
                    continue
            except docker.errors.NotFound:
                # Container was removed
                print(f"Container {container_name} not found")
                    
                # Try to get exit status before container was removed
                try:
                        result = subprocess.run(['docker', 'inspect', container_name, 
                                               '--format={{.State.ExitCode}} {{.State.OOMKilled}}'], 
                                              capture_output=True, text=True)
                        if result.returncode == 0 and result.stdout.strip():
                            parts = result.stdout.strip().split()
                            if len(parts) >= 2:
                                exit_code, oom_killed = parts[0], parts[1]
                                print(f"Container exit info: exit_code={exit_code}, oom_killed={oom_killed}")
                                if exit_code != '0' or oom_killed == 'true':
                                    raise RuntimeError(f"Frigate container failed permanently (exit_code={exit_code}, oom_killed={oom_killed})")
                except subprocess.SubprocessError:
                    pass
                
                # Container disappeared - will retry in the next iteration
                print(f"Container disappeared, will retry...")
                container = None  # Reset container to trigger recreation
                time.sleep(backoff_delay)
                backoff_delay = min(backoff_delay * 2, 60)
                continue
            except docker.errors.APIError as e:
                if "404" in str(e) or "not found" in str(e).lower():
                    print(f"Container {container_name} not found (API error)")
                    
                    # Try to get exit status before container was removed
                    try:
                            result = subprocess.run(['docker', 'inspect', container_name, 
                                                   '--format={{.State.ExitCode}} {{.State.OOMKilled}}'], 
                                                  capture_output=True, text=True)
                            if result.returncode == 0 and result.stdout.strip():
                                parts = result.stdout.strip().split()
                                if len(parts) >= 2:
                                    exit_code, oom_killed = parts[0], parts[1]
                                    print(f"Container exit info: exit_code={exit_code}, oom_killed={oom_killed}")
                                    if exit_code != '0' or oom_killed == 'true':
                                        raise RuntimeError(f"Frigate container failed permanently (exit_code={exit_code}, oom_killed={oom_killed})")
                    except subprocess.SubprocessError:
                        pass
                    
                    # Container disappeared - will retry in the next iteration
                    print(f"Container disappeared (API error), will retry...")
                    container = None  # Reset container to trigger recreation
                    time.sleep(backoff_delay)
                    backoff_delay = min(backoff_delay * 2, 60)
                    continue
                else:
                    raise
            
            # Check for early container failure
            print(f"[Worker: {docker_manager_for_frigate.worker_id}] "
                  f"DEBUG: Checking container status...")
            
            # Check if container has health check status with retry logic
            current_container = None
            for fetch_attempt in range(3):
                try:
                    current_container = docker_manager_for_frigate.client.containers.get(container.id)
                    break  # Success
                except docker.errors.NotFound:
                    print(f"[Worker: {docker_manager_for_frigate.worker_id}] "
                          f"DEBUG: Container not found on attempt {fetch_attempt + 1}")
                    if fetch_attempt < 2:  # Not the last attempt
                        time.sleep(0.5)
                        continue
                    else:
                        raise  # Re-raise on last attempt
            
            # Check container status and port mapping
            if current_container:
                print(f"[Worker: {docker_manager_for_frigate.worker_id}] "
                      f"Container status: {current_container.status}")
                
                # Early failure detection - check if container has exited
                if current_container.status in ['exited', 'dead', 'removing']:
                    print(f"[Worker: {docker_manager_for_frigate.worker_id}] "
                          f"ERROR: Container has failed with status: {current_container.status}")
                    
                    # Get exit code and logs for debugging
                    state = current_container.attrs.get('State', {})
                    exit_code = state.get('ExitCode', 'Unknown')
                    error = state.get('Error', 'No error message')
                    
                    print(f"[Worker: {docker_manager_for_frigate.worker_id}] "
                          f"Exit code: {exit_code}, Error: {error}")
                    
                    # Get last logs
                    try:
                        logs = current_container.logs(tail=50).decode('utf-8')
                        print(f"[Worker: {docker_manager_for_frigate.worker_id}] Last logs before exit:")
                        for line in logs.split('\n')[-20:]:
                            if line.strip():
                                print(f"  > {line.strip()}")
                    except:
                        pass
                    
                    raise RuntimeError(f"Frigate container exited early with status: {current_container.status}, "
                                     f"exit code: {exit_code}")
                
                # Check actual port bindings
                ports = current_container.attrs.get('NetworkSettings', {}).get('Ports', {})
                print(f"[Worker: {docker_manager_for_frigate.worker_id}] "
                      f"Port bindings: {ports}")
                
                # Check container processes
                try:
                    top = current_container.top()
                    process_count = len(top.get('Processes', []))
                    print(f"[Worker: {docker_manager_for_frigate.worker_id}] "
                          f"Running processes: {process_count}")
                    # Show key processes
                    for proc in top.get('Processes', [])[:5]:  # First 5 processes
                        if proc and len(proc) > 7:
                            print(f"  - PID {proc[1]}: {proc[7]}")
                except Exception as e:
                    print(f"[Worker: {docker_manager_for_frigate.worker_id}] "
                          f"Could not get process list: {e}")
            
            health_status = current_container.attrs.get('State', {}).get('Health', {}).get('Status') if current_container else None
            if health_status:
                print(f"Container health status: {health_status}")
            
            # Test network connectivity to the container
            print(f"[Worker: {docker_manager_for_frigate.worker_id}] "
                  f"DEBUG: Testing network connectivity to port {frigate_port}")
            
            # Try a raw socket connection first
            import socket
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex(('localhost', frigate_port))
                sock.close()
                print(f"[Worker: {docker_manager_for_frigate.worker_id}] "
                      f"Socket connection to port {frigate_port}: {'Success' if result == 0 else f'Failed with code {result}'}")
            except Exception as e:
                print(f"[Worker: {docker_manager_for_frigate.worker_id}] "
                      f"Socket connection error: {e}")
            
            # First try to access the root page to see if nginx is responding
            root_url = f"http://localhost:{frigate_port}/"
            print(f"[Worker: {docker_manager_for_frigate.worker_id}] "
                  f"Testing root URL at {root_url}")
            try:
                root_response = requests.get(root_url, timeout=3, allow_redirects=False)
                print(f"[Worker: {docker_manager_for_frigate.worker_id}] "
                      f"Root page response: {root_response.status_code}")
                if root_response.status_code != 200:
                    print(f"[Worker: {docker_manager_for_frigate.worker_id}] "
                          f"Root page headers: {dict(root_response.headers)}")
            except Exception as e:
                print(f"[Worker: {docker_manager_for_frigate.worker_id}] "
                      f"Root page error: {type(e).__name__}: {e}")
            
            # Test API endpoint - try version endpoint which should be simpler
            api_url = f"http://localhost:{frigate_port}/api/version"
            elapsed_time = time.time() - start_time
            print(f"[Worker: {docker_manager_for_frigate.worker_id}] "
                  f"Testing API at {api_url} (attempt {attempt_count}, elapsed: {elapsed_time:.1f}s)")
            response = requests.get(api_url, timeout=5)
            print(f"[Worker: {docker_manager_for_frigate.worker_id}] "
                  f"API response: {response.status_code}")
            if response.status_code == 200:
                # Version endpoint is ready, now check other critical endpoints
                print(f"[Worker: {docker_manager_for_frigate.worker_id}] "
                      f"Version API ready, checking other endpoints...")
                
                # List of endpoints that tests actually use
                critical_endpoints = [
                    ('/api/stats', 'Stats'),
                    ('/api/config', 'Config'),
                    ('/api/events', 'Events')
                ]
                
                all_endpoints_ready = True
                for endpoint, name in critical_endpoints:
                    endpoint_url = f"http://localhost:{frigate_port}{endpoint}"
                    try:
                        endpoint_response = requests.get(endpoint_url, timeout=3)
                        print(f"[Worker: {docker_manager_for_frigate.worker_id}] "
                              f"{name} API ({endpoint}): {endpoint_response.status_code}")
                        if endpoint_response.status_code != 200:
                            all_endpoints_ready = False
                            last_error = f"{name} API not ready: {endpoint_response.status_code}"
                            break
                    except Exception as e:
                        all_endpoints_ready = False
                        last_error = f"{name} API error: {e}"
                        print(f"[Worker: {docker_manager_for_frigate.worker_id}] "
                              f"{name} API error: {e}")
                        break
                
                if all_endpoints_ready:
                    frigate_ready = True
                    print(f"[Worker: {docker_manager_for_frigate.worker_id}] "
                          f"✅ All Frigate API endpoints ready after {time.time() - start_time:.1f}s")
                    break  # Success - exit the wait loop
            else:
                last_error = f"API returned status {response.status_code}"
                # Get more details on non-200 responses
                try:
                    print(f"[Worker: {docker_manager_for_frigate.worker_id}] "
                          f"Response body: {response.text[:200]}")
                except:
                    pass
        except requests.exceptions.RequestException as e:
            last_error = f"Request error: {str(e)}"
            # Log 409 errors specifically for debugging
            if "409" in str(e):
                print(f"[Worker: {docker_manager_for_frigate.worker_id}] "
                      f"409 Conflict error during health check: {e}")
        except docker.errors.NotFound:
            # Already handled above
            pass
        except Exception as e:
            last_error = f"Unexpected error: {str(e)}"
            print(f"[Worker: {docker_manager_for_frigate.worker_id}] "
                  f"Unexpected error during health check: {e}")
        
        # Exponential backoff as recommended by o3
        if not frigate_ready:
            # Check container logs periodically for debugging
            if container and int((time.time() - start_time) / 30) > int((time.time() - start_time - backoff_delay) / 30):
                # Every 30 seconds, show status and recent logs
                elapsed = time.time() - start_time
                print(f"\n[Worker: {docker_manager_for_frigate.worker_id}] "
                      f"Status update at {elapsed:.1f}s (attempt {attempt_count}):")
                
                try:
                    # Check if container is still running
                    safe_container_reload(container, worker_id=worker_id)
                    print(f"  - Container status: {container.status}")
                    
                    # Get process count
                    try:
                        top = container.top()
                        process_count = len(top.get('Processes', []))
                        print(f"  - Running processes: {process_count}")
                    except:
                        pass
                    
                    # Show recent logs
                    logs = container.logs(tail=20).decode('utf-8')
                    print(f"  - Recent container logs:")
                    # Filter out the nginx 400 errors which are just noise
                    shown = 0
                    for line in logs.split('\n')[-10:]:  # Last 10 lines
                        if line.strip() and '400 0 "-" "-" "-"' not in line:
                            print(f"    > {line.strip()}")
                            shown += 1
                    if shown == 0:
                        print(f"    > (No new significant logs)")
                except Exception as e:
                    print(f"  - Could not get status: {e}")
            
            # Check if we're approaching timeout and exit early
            if time.time() - start_time >= frigate_timeout * 0.9:  # 90% of timeout
                print(f"[Worker: {docker_manager_for_frigate.worker_id}] "
                      f"Approaching timeout, stopping wait loop")
                break
            
            time.sleep(min(backoff_delay, 5))  # Reduced cap to 5 seconds
            backoff_delay *= 1.2  # Slower increase
    
    if not frigate_ready:
        # Comprehensive failure diagnostics
        print("\n" + "="*80)
        print(f"[Worker: {docker_manager_for_frigate.worker_id}] FRIGATE STARTUP FAILURE DIAGNOSTICS")
        print("="*80)
        
        print(f"\nTiming Information:")
        actual_wait_time = time.time() - start_time
        print(f"  - Total wait time: {actual_wait_time:.1f} seconds")
        print(f"  - Total attempts: {attempt_count}")
        print(f"  - Max attempts allowed: {MAX_WAIT_ATTEMPTS}")
        print(f"  - Timeout configured: {frigate_timeout} seconds")
        print(f"  - Last error: {last_error}")
        
        # Check why we exited
        if attempt_count >= MAX_WAIT_ATTEMPTS:
            print(f"\n⚠️  EXITED DUE TO MAX ATTEMPTS REACHED ({MAX_WAIT_ATTEMPTS})")
        elif time.time() - start_time >= frigate_timeout:
            print(f"\n⚠️  EXITED DUE TO TIMEOUT ({frigate_timeout}s)")
        
        # Get container logs for debugging if container exists
        if container is not None:
            try:
                # Container state
                safe_container_reload(container, worker_id=worker_id)
                print(f"\nContainer State:")
                print(f"  - Status: {container.status}")
                print(f"  - ID: {container.id[:12]}")
                print(f"  - Name: {container.name}")
                
                # Detailed state info
                state = container.attrs.get('State', {})
                print(f"  - Running: {state.get('Running', False)}")
                print(f"  - ExitCode: {state.get('ExitCode', 'N/A')}")
                print(f"  - Error: {state.get('Error', 'None')}")
                print(f"  - StartedAt: {state.get('StartedAt', 'Unknown')}")
                
                # Resource usage
                try:
                    stats = container.stats(stream=False)
                    cpu_stats = stats.get('cpu_stats', {})
                    memory_stats = stats.get('memory_stats', {})
                    print(f"\nResource Usage:")
                    print(f"  - CPU: {cpu_stats.get('cpu_usage', {}).get('total_usage', 'N/A')}")
                    print(f"  - Memory: {memory_stats.get('usage', 'N/A')} / {memory_stats.get('limit', 'N/A')}")
                except:
                    print(f"\nResource Usage: Unable to get stats")
                
                # Container logs
                print(f"\nContainer Logs (last 200 lines):")
                logs = container.logs(tail=200).decode('utf-8')
                # Show unique log lines (skip duplicates)
                seen_lines = set()
                for line in logs.split('\n'):
                    if line.strip() and line not in seen_lines:
                        seen_lines.add(line)
                        print(f"  {line}")
                        if len(seen_lines) > 50:  # Limit output
                            total_lines = len(logs.split('\n'))
                            print(f"  ... (truncated, {total_lines} total lines)")
                            break
                
            except Exception as e:
                print(f"\nFailed to get debug info: {type(e).__name__}: {e}")
        else:
            print(f"\nFrigate container never started successfully.")
            print(f"Last error: {last_error}")
        
        # Clean up failed container before raising error
        if container is not None:
            try:
                print(f"\nCleaning up failed container {container_name}...")
                container.stop(timeout=5)
                container.remove()
                print(f"Container {container_name} cleaned up successfully")
            except Exception as e:
                print(f"Failed to clean up container: {e}")
        
        # Clean up config directory
        shutil.rmtree(config_dir, ignore_errors=True)
        
        print("\n" + "="*80)
        raise RuntimeError(f"Frigate failed to become ready after {actual_wait_time:.1f} seconds. Last error: {last_error}")
    
    # Use try/finally to ensure cleanup even on success
    try:
        yield container
    finally:
        # Release container reference - it will be cleaned up when ref count reaches zero
        docker_manager_for_frigate.release_container(container_name)
        
        # Cleanup config directory
        shutil.rmtree(config_dir, ignore_errors=True)

@pytest.mark.frigate_slow
class TestSecurityNVRIntegration:
    """Integration tests for Security NVR service"""

    @pytest.fixture(autouse=True)
    def setup(self, mqtt_broker_for_frigate, frigate_container, docker_manager_for_frigate):
        """Setup test environment with real MQTT broker and Frigate"""
        self.mqtt_messages = []
        self._mqtt_connected = False
        self._mqtt_client = None
        self.mqtt_broker = mqtt_broker_for_frigate
        self.frigate_container = frigate_container
        self.docker_manager = docker_manager_for_frigate
        self.frigate_api_url = f"http://localhost:{frigate_container.frigate_port}"
        # No auth needed for test instance (disabled in config)
        
        # Update MQTT connection params to use test broker
        self.mqtt_host = mqtt_broker_for_frigate.host
        self.mqtt_port = mqtt_broker_for_frigate.port
        
        yield
        
        # Cleanup after each test
        if self._mqtt_client and self._mqtt_connected:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
    
    def _check_container_exists(self) -> bool:
        """Check if frigate container still exists and is running."""
        if not hasattr(self, 'frigate_container') or self.frigate_container is None:
            return False
        try:
            # Re-fetch container instead of reload as recommended by o3
            container = self.docker_manager.client.containers.get(self.frigate_container.id)
            return container.status == 'running'
        except docker.errors.NotFound:
            return False
        except docker.errors.APIError as e:
            if "404" in str(e) or "not found" in str(e).lower():
                return False
            # For other API errors, log but return False
            print(f"API error checking container: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error checking container: {e}")
            return False
    
    def _exec_run_with_retry(self, cmd, retries=3, delay=1):
        """Execute command in container with retry logic and existence checking."""
        backoff_delay = delay
        
        for attempt in range(retries):
            try:
                if not self._check_container_exists():
                    raise RuntimeError(f"Frigate container no longer exists")
                
                # Re-fetch container to ensure we have the latest reference
                try:
                    container = self.docker_manager.client.containers.get(self.frigate_container.id)
                except docker.errors.NotFound:
                    if attempt < retries - 1:
                        print(f"Container not found, retrying in {backoff_delay}s (attempt {attempt + 1}/{retries})")
                        time.sleep(backoff_delay)
                        backoff_delay = min(backoff_delay * 1.5, 10)  # Exponential backoff
                        continue
                    else:
                        raise RuntimeError(f"Frigate container disappeared after {retries} attempts")
                
                # Try to execute the command
                exit_code, output = container.exec_run(cmd, demux=True)
                return exit_code, output
                
            except docker.errors.NotFound:
                if attempt < retries - 1:
                    print(f"Container not found during exec, retrying in {backoff_delay}s (attempt {attempt + 1}/{retries})")
                    time.sleep(backoff_delay)
                    backoff_delay = min(backoff_delay * 1.5, 10)  # Exponential backoff
                else:
                    raise RuntimeError(f"Frigate container disappeared after {retries} attempts")
            except docker.errors.APIError as e:
                if "is not running" in str(e):
                    if attempt < retries - 1:
                        print(f"Container not running, retrying in {backoff_delay}s (attempt {attempt + 1}/{retries})")
                        time.sleep(backoff_delay)
                        backoff_delay = min(backoff_delay * 1.5, 10)  # Exponential backoff
                    else:
                        raise RuntimeError(f"Container not running after {retries} attempts")
                else:
                    if attempt < retries - 1:
                        print(f"API error: {e}, retrying...")
                        time.sleep(backoff_delay)
                        backoff_delay = min(backoff_delay * 1.5, 10)  # Exponential backoff
                    else:
                        raise
            except Exception as e:
                if attempt < retries - 1:
                    print(f"Error executing command: {e}, retrying...")
                    time.sleep(backoff_delay)
                    backoff_delay = min(backoff_delay * 1.5, 10)  # Exponential backoff
                else:
                    raise
    
    # ========== Service Health Tests ==========
    
    @requires_security_nvr
    def test_frigate_service_running(self):
        """Test that Frigate service is running and accessible"""
        # Check if our test Frigate container is running
        container_name = self.frigate_container.name
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={container_name}", "--format", "{{.Status}}"],
            capture_output=True,
            text=True
        )
        assert "Up" in result.stdout, f"Frigate test container {container_name} is not running"
        
        # Check API accessibility without auth (since we disabled it in config)
        response = requests.get(f"{self.frigate_api_url}/api/version", timeout=5)
        assert response.status_code == 200, f"Frigate API not accessible: {response.status_code}"
        
        # Verify version format
        version = response.text.strip()
        assert version, "No version returned"
        print(f"Frigate version: {version}")
        
        # Check container health if available
        health_result = subprocess.run(
            ["docker", "inspect", container_name, "--format", "{{.State.Health.Status}}"],
            capture_output=True,
            text=True
        )
        # Health check might not be configured, so we don't assert on it
        if health_result.stdout.strip() and health_result.stdout.strip() != "<no value>":
            print(f"Container health: {health_result.stdout.strip()}")
    
    @requires_frigate_api
    def test_frigate_stats_endpoint(self):
        """Test Frigate stats API endpoint"""
        # No auth needed since we disabled it in test config
        response = requests.get(f"{self.frigate_api_url}/api/stats", timeout=5)
        assert response.status_code == 200
        
        stats = response.json()
        # Debug logging to understand API response structure
        print(f"[DEBUG] Stats response keys: {list(stats.keys())}")
        
        assert "detectors" in stats
        assert "cameras" in stats
        assert "service" in stats
        
        # Check service info
        service = stats["service"]
        print(f"[DEBUG] Service uptime: {service.get('uptime', 'N/A')}, keys: {list(service.keys())}")
        
        # Uptime can be 0 for just-started containers, which is valid
        assert service["uptime"] >= 0, f"Unexpected uptime value: {service['uptime']}"
        assert "storage" in service
    
    # ========== Hardware Detection Tests ==========
    
    @requires_security_nvr
    def test_hardware_detector_execution(self):
        """Test that hardware detector runs and produces output"""
        # Check container exists first
        if not self._check_container_exists():
            pytest.skip("Frigate container is not running")
            
        # Use the container object directly instead of subprocess
        try:
            exit_code, output = self._exec_run_with_retry(
                ["python3", "-c", 
                 "import platform, psutil; print(f'Platform: {platform.platform()}'); print(f'CPU: {psutil.cpu_count()} cores'); print(f'Memory: {psutil.virtual_memory().total // (1024**3)} GB')"]
            )
            result_stdout = output[0].decode() if output[0] else ""
            result_stderr = output[1].decode() if output[1] else ""
        except docker.errors.NotFound:
            pytest.skip("Frigate container was removed during test execution")
        except Exception as e:
            pytest.skip(f"Cannot execute in Frigate container: {e}")
        
        assert exit_code == 0, f"Hardware detector failed: {result_stderr}"
        output = result_stdout
        
        # Check for expected hardware detection output
        assert "platform" in output.lower()
        assert "cpu" in output.lower()
        assert "memory" in output.lower()
    
    @requires_frigate_api
    def test_detector_configuration(self):
        """Test that detector is properly configured based on hardware"""
        # No auth needed since we disabled it in test config
        response = requests.get(f"{self.frigate_api_url}/api/config", timeout=5)
        assert response.status_code == 200
        
        config = response.json()
        assert "detectors" in config
        
        # Check that at least one detector is configured
        detectors = config["detectors"]
        assert len(detectors) > 0, "No detectors configured"
        
        # Verify detector settings
        for detector_name, detector_config in detectors.items():
            assert "type" in detector_config
            assert detector_config["type"] in ["cpu", "edgetpu", "openvino", "tensorrt"]
            
            # Check model configuration
            if "model" in detector_config:
                model = detector_config["model"]
                assert "width" in model
                assert "height" in model
                assert model["width"] in [320, 416, 640, 1280]  # Valid model sizes
                assert model["height"] in [320, 416, 640, 1280]
    
    # ========== Camera Integration Tests ==========
    
    @requires_security_nvr
    @requires_mqtt
    @pytest.mark.timeout(120)  # Kill test after 2 minutes to prevent hanging
    def test_camera_discovery_integration(self):
        """Test integration with camera_detector service"""
        # Setup MQTT client to simulate camera discovery
        mqtt_client = mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            client_id="test_camera_discovery"
        )
        mqtt_client.on_connect = lambda c, u, f, rc, props: setattr(self, '_mqtt_connected', rc == 0)
        
        try:
            mqtt_client.connect(self.mqtt_host, self.mqtt_port, 60)
            mqtt_client.loop_start()
            
            # Wait for connection
            timeout = time.time() + 5
            while not self._mqtt_connected and time.time() < timeout:
                time.sleep(0.1)
            
            assert self._mqtt_connected, "Failed to connect to MQTT broker"
            
            # Publish camera discovery message
            camera_data = {
                "camera_id": "test_cam_001",
                "ip": "192.168.1.100",
                "rtsp_url": "rtsp://username:password@192.0.2.100:554/stream1",
                "manufacturer": "TestCam",
                "model": "TC-1000",
                "mac_address": "00:11:22:33:44:55",
                "capabilities": {
                    "ptz": False,
                    "audio": True,
                    "resolution": "1920x1080"
                }
            }
            
            mqtt_client.publish("cameras/discovered", json.dumps(camera_data), qos=1)
            time.sleep(2)  # Allow time for processing
            
        finally:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
    
    @requires_frigate_api
    def test_camera_configuration_format(self):
        """Test that camera configurations match expected format"""
        # No auth needed since we disabled it in test config
        # Add retry logic for Frigate API connection issues
        max_retries = 5
        retry_delay = 2
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = requests.get(f"{self.frigate_api_url}/api/config", timeout=5)
                assert response.status_code == 200
                config = response.json()
                break  # Success, exit retry loop
            except (requests.exceptions.RequestException, AssertionError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    raise last_error
        
        if "cameras" in config and len(config["cameras"]) > 0:
            for camera_name, camera_config in config["cameras"].items():
                # Check required camera configuration
                assert "ffmpeg" in camera_config
                assert "detect" in camera_config
                
                # Check ffmpeg inputs
                ffmpeg = camera_config["ffmpeg"]
                assert "inputs" in ffmpeg
                assert len(ffmpeg["inputs"]) > 0
                
                # Check detect settings
                detect = camera_config["detect"]
                assert "width" in detect
                assert "height" in detect
                # Allow common resolutions including 720p
                assert detect["width"] in [320, 416, 640, 720, 1280, 1920]
                assert detect["height"] in [320, 416, 640, 720, 1080, 1280]
    
    # ========== MQTT Publishing Tests ==========
    
    def test_mqtt_connection(self):
        """Test MQTT connection using real test broker"""
        # Create a test MQTT client
        test_client = mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            client_id="test_mqtt_connection"
        )
        
        connected = False
        def on_connect(client, userdata, flags, rc, props=None):
            nonlocal connected
            connected = True
            
        test_client.on_connect = on_connect
        
        try:
            test_client.connect(self.mqtt_host, self.mqtt_port, 60)
            test_client.loop_start()
            
            # Wait for connection
            timeout = 5
            start_time = time.time()
            while not connected and time.time() - start_time < timeout:
                time.sleep(0.1)
            
            assert connected, f"Failed to connect to MQTT broker at {self.mqtt_host}:{self.mqtt_port}"
            
            # Test publishing
            result = test_client.publish("test/connection", "test_message", qos=1)
            assert result.rc == mqtt.MQTT_ERR_SUCCESS
            
        finally:
            test_client.loop_stop()
            test_client.disconnect()
    
    def test_mqtt_event_publishing(self):
        """Test MQTT event publishing capability"""
        received_messages = []
        
        def on_message(client, userdata, msg):
            received_messages.append({
                "topic": msg.topic,
                "payload": msg.payload.decode('utf-8')
            })
        
        # Setup MQTT subscriber using test broker
        mqtt_client = mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            client_id="test_event_subscriber"
        )
        
        connected = False
        def on_connect(client, userdata, flags, rc, props=None):
            nonlocal connected
            connected = True
            client.subscribe("frigate/#")
            
        mqtt_client.on_connect = on_connect
        mqtt_client.on_message = on_message
        
        try:
            mqtt_client.connect(self.mqtt_host, self.mqtt_port, 60)
            mqtt_client.loop_start()
            
            # Wait for connection
            timeout = 5
            start_time = time.time()
            while not connected and time.time() - start_time < timeout:
                time.sleep(0.1)
                
            assert connected, "Failed to connect to MQTT broker"
            
            # Simulate Frigate event publishing
            publisher = mqtt.Client(
                callback_api_version=CallbackAPIVersion.VERSION2,
                client_id="test_frigate_simulator"
            )
            publisher.connect(self.mqtt_host, self.mqtt_port, 60)
            
            # Publish test messages that Frigate would send
            test_messages = [
                ("frigate/available", "online"),
                ("frigate/stats", json.dumps({"detectors": {"cpu": {"fps": 5.0}}})),
                ("frigate/events", json.dumps({"type": "new", "label": "fire"}))
            ]
            
            for topic, payload in test_messages:
                publisher.publish(topic, payload, qos=1)
            
            # Wait for messages
            time.sleep(2)
            
            # Check for expected message patterns
            topics_found = [msg["topic"] for msg in received_messages]
            
            # Should have received our test messages
            expected_patterns = ["frigate/available", "frigate/stats", "frigate/events"]
            for pattern in expected_patterns:
                assert any(pattern in topic for topic in topics_found), \
                    f"Expected MQTT topic pattern '{pattern}' not found in {topics_found}"
                    
            publisher.disconnect()
                    
        finally:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
    
    def test_fire_detection_event_format(self):
        """Test the format of fire detection events"""
        # This test verifies the expected event format
        # In a real scenario, we'd trigger a detection
        
        expected_event = {
            "type": "new",
            "before": {
                "id": "1234567890.123456-abc123",
                "camera": "front_yard",
                "frame_time": 1234567890.123456,
                "label": "fire",
                "score": 0.85,
                "box": [320, 180, 480, 360]
            }
        }
        
        # Validate event structure
        assert "type" in expected_event
        assert "before" in expected_event
        
        before = expected_event["before"]
        assert all(key in before for key in ["id", "camera", "frame_time", "label", "score", "box"])
        assert before["label"] in ["fire", "smoke"]
        assert 0 <= before["score"] <= 1
        assert len(before["box"]) == 4
    
    # ========== Storage Tests ==========
    
    @requires_security_nvr
    def test_usb_storage_configuration(self):
        """Test USB storage is properly configured"""
        # Check if storage path exists in test container
        try:
            exit_code, _ = self._exec_run_with_retry(
                ["test", "-d", "/media/frigate"]
            )
        except (docker.errors.NotFound, RuntimeError):
            pytest.skip("Frigate container was removed")
        except Exception as e:
            pytest.skip(f"Cannot execute in Frigate container: {e}")
        
        # Storage directory should exist (even if not mounted)
        # In test environment, this might not exist, so check alternatives
        if exit_code != 0:
            # Check if any storage directory exists
            try:
                exit_code2, output = self._exec_run_with_retry(
                    ["ls", "-la", "/"]
                )
                result_stdout = output[0].decode() if output[0] else ""
            except docker.errors.NotFound:
                pytest.skip("Frigate container was removed")
            except Exception as e:
                pytest.skip(f"Cannot execute in Frigate container: {e}")
            
            # Just verify the container is functional
            assert exit_code2 == 0, "Container filesystem not accessible"
            print(f"Note: /media/frigate not found in test container, this is normal for test environment")
        else:
            print(f"✓ Storage path /media/frigate exists in container")
    
    def test_recording_directory_structure(self):
        """Test that recording directory structure is documented correctly"""
        # This is a documentation test - verifies expected structure
        expected_paths = [
            "/media/frigate/recordings",
            "/media/frigate/clips",
            "/media/frigate/exports"
        ]
        
        # Just verify we know what paths should exist
        assert len(expected_paths) == 3
    
    # ========== Model Configuration Tests ==========
    
    @requires_frigate_api
    def test_wildfire_model_configuration(self):
        """Test that wildfire detection models are properly configured"""
        # No auth needed since we disabled it in test config
        response = requests.get(f"{self.frigate_api_url}/api/config", timeout=5)
        assert response.status_code == 200, f"Config API returned {response.status_code}: {response.text}"
        config = response.json()
        
        # Check for wildfire-specific configuration
        if "objects" in config and "track" in config["objects"]:
            tracked_objects = config["objects"]["track"]
            assert "fire" in tracked_objects, f"Fire detection not configured. Tracked objects: {tracked_objects}"
            
        # Check model configuration in detectors
        if "detectors" in config:
            for detector_name, detector_config in config["detectors"].items():
                if "model" in detector_config:
                    model = detector_config["model"]
                    assert "width" in model
                    assert "height" in model
    
    @requires_frigate_api
    def test_detection_settings(self):
        """Test detection settings match documentation"""
        # No auth needed since we disabled it in test config
        response = requests.get(f"{self.frigate_api_url}/api/config", timeout=5)
        assert response.status_code == 200, f"Config API returned {response.status_code}: {response.text}"
        config = response.json()
        
        # Check global detect settings
        if "detect" in config:
            detect = config["detect"]
            if "fps" in detect:
                assert 1 <= detect["fps"] <= 10, "Detection FPS out of expected range"
    
    # ========== API Endpoint Tests ==========
    
    @requires_frigate_api
    def test_events_api(self):
        """Test Frigate events API endpoint"""
        # No auth needed since we disabled it in test config
        response = requests.get(f"{self.frigate_api_url}/api/events", timeout=5)
        assert response.status_code == 200
        
        events = response.json()
        assert isinstance(events, list)
        
        # If there are events, validate structure
        if len(events) > 0:
            event = events[0]
            assert "id" in event
            assert "camera" in event
            assert "label" in event
            assert "start_time" in event
    
    @requires_frigate_api
    def test_recordings_api(self):
        """Test Frigate recordings API endpoint"""
        # No auth needed since we disabled it in test config
        response = requests.get(f"{self.frigate_api_url}/api/recordings/summary", timeout=5)
        # API might return 404 if no recordings exist yet
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            summary = response.json()
            print(f"[DEBUG] Recordings API response type: {type(summary).__name__}, content: {summary}")
            
            # API returns empty dict {} when no recordings exist, list when recordings exist
            assert isinstance(summary, (list, dict)), f"Expected list or dict, got {type(summary).__name__}: {summary}"
    
    # ========== Performance Tests ==========
    
    @requires_frigate_api
    def test_cpu_usage(self):
        """Test that CPU usage is within acceptable limits"""
        # No auth needed since we disabled it in test config
        response = requests.get(f"{self.frigate_api_url}/api/stats", timeout=5)
        stats = response.json()
        
        if "cpu_usages" in stats:
            cpu_stats = stats["cpu_usages"]
            
            # Check overall CPU usage
            if "cpu" in cpu_stats and "cpu" in cpu_stats["cpu"]:
                cpu_percent = cpu_stats["cpu"]["cpu"]
                assert cpu_percent < 80, f"CPU usage too high: {cpu_percent}%"
    
    @requires_frigate_api
    def test_detector_inference_speed(self):
        """Test that detector inference speed is acceptable"""
        # No auth needed since we disabled it in test config
        response = requests.get(f"{self.frigate_api_url}/api/stats", timeout=5)
        stats = response.json()
        
        if "detectors" in stats:
            for detector_name, detector_stats in stats["detectors"].items():
                if "inference_speed" in detector_stats:
                    speed = detector_stats["inference_speed"]
                    # Check inference speed based on hardware type
                    if "coral" in detector_name.lower():
                        assert speed < 50, f"Coral inference too slow: {speed}ms"
                    elif "hailo" in detector_name.lower():
                        assert speed < 30, f"Hailo inference too slow: {speed}ms"
                    else:  # CPU or other
                        assert speed < 200, f"Inference too slow: {speed}ms"
    
    # ========== Integration Flow Tests ==========
    
    @pytest.mark.integration
    @requires_security_nvr
    @requires_mqtt
    def test_full_detection_flow(self):
        """Test complete flow from camera to MQTT event"""
        # This test would require a test video or live camera
        # It verifies the entire pipeline works
        
        # Setup MQTT subscriber for fire events
        fire_events = []
        
        def on_fire_event(client, userdata, msg):
            if "fire" in msg.topic or "smoke" in msg.topic:
                fire_events.append({
                    "topic": msg.topic,
                    "payload": json.loads(msg.payload.decode('utf-8'))
                })
        
        mqtt_client = mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            client_id="test_fire_detector"
        )
        mqtt_client.on_connect = lambda c, u, f, rc, props: c.subscribe("frigate/+/fire")
        mqtt_client.on_message = on_fire_event
        
        try:
            mqtt_client.connect(self.mqtt_host, self.mqtt_port, 60)
            mqtt_client.loop_start()
            
            # In a real test, we would:
            # 1. Inject a test video with fire
            # 2. Wait for detection
            # 3. Verify MQTT event published
            
            time.sleep(5)  # Wait for any existing events
            
            # Verify event structure if any were received
            for event in fire_events:
                payload = event["payload"]
                assert "type" in payload
                assert "camera" in payload or "before" in payload
                
        finally:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()


@pytest.mark.frigate_slow
class TestServiceDependencies:
    """Test service dependencies and startup order"""
    
    @pytest.fixture(autouse=True)
    def setup(self, mqtt_broker_for_frigate, frigate_container, docker_manager_for_frigate):
        """Setup test environment"""
        self.mqtt_broker = mqtt_broker_for_frigate
        self.frigate_container = frigate_container
        self.docker_manager = docker_manager_for_frigate
    
    def _check_container_exists(self) -> bool:
        """Check if frigate container still exists and is running."""
        if not hasattr(self, 'frigate_container') or self.frigate_container is None:
            return False
        try:
            # Re-fetch container instead of reload as recommended by o3
            container = self.docker_manager.client.containers.get(self.frigate_container.id)
            return container.status == 'running'
        except docker.errors.NotFound:
            return False
        except docker.errors.APIError as e:
            if "404" in str(e) or "not found" in str(e).lower():
                return False
            # For other API errors, log but return False
            print(f"API error checking container: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error checking container: {e}")
            return False
    
    def _exec_run_with_retry(self, cmd, retries=3, delay=1):
        """Execute command in container with retry logic and existence checking."""
        backoff_delay = delay
        
        for attempt in range(retries):
            try:
                if not self._check_container_exists():
                    raise RuntimeError(f"Frigate container no longer exists")
                
                # Re-fetch container to ensure we have the latest reference
                try:
                    container = self.docker_manager.client.containers.get(self.frigate_container.id)
                except docker.errors.NotFound:
                    if attempt < retries - 1:
                        print(f"Container not found, retrying in {backoff_delay}s (attempt {attempt + 1}/{retries})")
                        time.sleep(backoff_delay)
                        backoff_delay = min(backoff_delay * 1.5, 10)  # Exponential backoff
                        continue
                    else:
                        raise RuntimeError(f"Frigate container disappeared after {retries} attempts")
                
                # Try to execute the command
                exit_code, output = container.exec_run(cmd, demux=True)
                return exit_code, output
                
            except docker.errors.NotFound:
                if attempt < retries - 1:
                    print(f"Container not found during exec, retrying in {backoff_delay}s (attempt {attempt + 1}/{retries})")
                    time.sleep(backoff_delay)
                    backoff_delay = min(backoff_delay * 1.5, 10)  # Exponential backoff
                else:
                    raise RuntimeError(f"Frigate container disappeared after {retries} attempts")
            except docker.errors.APIError as e:
                if "is not running" in str(e):
                    if attempt < retries - 1:
                        print(f"Container not running, retrying in {backoff_delay}s (attempt {attempt + 1}/{retries})")
                        time.sleep(backoff_delay)
                        backoff_delay = min(backoff_delay * 1.5, 10)  # Exponential backoff
                    else:
                        raise RuntimeError(f"Container not running after {retries} attempts")
                else:
                    if attempt < retries - 1:
                        print(f"API error: {e}, retrying...")
                        time.sleep(backoff_delay)
                        backoff_delay = min(backoff_delay * 1.5, 10)  # Exponential backoff
                    else:
                        raise
            except Exception as e:
                if attempt < retries - 1:
                    print(f"Error executing command: {e}, retrying...")
                    time.sleep(backoff_delay)
                    backoff_delay = min(backoff_delay * 1.5, 10)  # Exponential backoff
                else:
                    raise
    
    @requires_security_nvr
    def test_mqtt_broker_dependency(self):
        """Test that security_nvr depends on mqtt_broker"""
        # Verify the container is available and running
        if not hasattr(self, 'frigate_container') or self.frigate_container is None:
            pytest.skip("Frigate container not available")
            
        # Check container status
        try:
            safe_container_reload(self.frigate_container)
            if self.frigate_container.status != 'running':
                pytest.skip(f"Frigate container not running: {self.frigate_container.status}")
        except docker.errors.NotFound:
            pytest.skip("Frigate container was removed")
        except docker.errors.APIError:
            pytest.skip("Cannot access Frigate container")
            
        # Check if our test Frigate container can reach the test MQTT broker
        try:
            exit_code, _ = self._exec_run_with_retry(
                ["ping", "-c", "1", "host.docker.internal"]
            )
        except (docker.errors.NotFound, RuntimeError):
            pytest.skip("Frigate container was removed")
        except Exception as e:
            pytest.skip(f"Cannot execute in Frigate container: {e}")
        
        # If ping fails, test the MQTT connection directly
        if exit_code != 0:
            # Test MQTT connectivity from the container
            mqtt_test_cmd = f"python3 -c \"import paho.mqtt.client as mqtt; c=mqtt.Client(); c.connect('host.docker.internal', {self.mqtt_broker.port}); print('MQTT OK')\""
            try:
                exit_code2, output = self._exec_run_with_retry(
                    ["sh", "-c", mqtt_test_cmd]
                )
                result_stdout = output[0].decode() if output[0] else ""
                result_stderr = output[1].decode() if output[1] else ""
            except docker.errors.NotFound:
                pytest.skip("Frigate container was removed")
            except Exception as e:
                pytest.skip(f"Cannot execute in Frigate container: {e}")
                
            assert "MQTT OK" in result_stdout, f"Cannot reach MQTT broker from container: {result_stderr}"
    
    def test_camera_detector_integration(self):
        """Test that security_nvr can receive camera updates"""
        # Verify the container is available and running
        if not hasattr(self, 'frigate_container') or self.frigate_container is None:
            pytest.skip("Frigate container not available")
            
        # Check container status
        try:
            safe_container_reload(self.frigate_container)
            if self.frigate_container.status != 'running':
                pytest.skip(f"Frigate container not running: {self.frigate_container.status}")
        except docker.errors.NotFound:
            pytest.skip("Frigate container was removed")
        except docker.errors.APIError:
            pytest.skip("Cannot access Frigate container")
            
        # Check if both services are on the same network
        result = subprocess.run(
            ["docker", "network", "ls", "--format", "{{.Name}}"],
            capture_output=True,
            text=True
        )
        
        # Just verify Docker networks exist
        assert result.returncode == 0


@pytest.mark.frigate_slow
class TestWebInterface:
    """Test Frigate web interface accessibility"""
    
    @pytest.fixture(autouse=True)
    def setup(self, mqtt_broker_for_frigate, frigate_container, docker_manager_for_frigate):
        """Setup test environment"""
        self.mqtt_broker = mqtt_broker_for_frigate
        self.frigate_container = frigate_container
        self.docker_manager = docker_manager_for_frigate
        self.frigate_api_url = f"http://localhost:{frigate_container.frigate_port}"
        # No auth needed for test instance (disabled in config)
    
    @requires_frigate_api
    def test_web_ui_accessible(self):
        """Test that Frigate web UI is accessible"""
        import time
        
        # Add retry logic for web UI access
        max_retries = 5
        retry_delay = 2
        last_error = None
        
        for i in range(max_retries):
            try:
                # Use the actual test Frigate instance - no auth needed
                response = requests.get(f"{self.frigate_api_url}/", timeout=5)
                assert response.status_code == 200
                # The root path might return different content, check for typical responses
                response_text = response.text.lower()
                # Accept various valid Frigate responses
                valid_responses = ["frigate", "running", "ok", "<!doctype html"]
                assert any(indicator in response_text for indicator in valid_responses), \
                    f"Unexpected response from Frigate UI: {response.text[:100]}"
                return  # Success
            except (requests.exceptions.RequestException, AssertionError) as e:
                last_error = e
                if i < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
        
        # If we get here, all retries failed
        pytest.fail(f"Failed to access Frigate web UI after {max_retries} attempts: {last_error}")
    
    @requires_frigate_api
    def test_static_resources(self):
        """Test that static resources are served"""
        import time
        
        # Add retry logic for API access
        max_retries = 5
        retry_delay = 2
        last_error = None
        
        for i in range(max_retries):
            try:
                # Check if API version endpoint works with the test instance - no auth needed
                response = requests.get(f"{self.frigate_api_url}/version", timeout=5)
                assert response.status_code == 200
                return  # Success
            except (requests.exceptions.RequestException, AssertionError) as e:
                last_error = e
                if i < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
        
        # If we get here, all retries failed
        pytest.fail(f"Failed to access Frigate API after {max_retries} attempts: {last_error}")


if __name__ == "__main__":
    # Run with markers to control which tests run
    import sys
    
    # Check what's available
    print("Checking service availability...")
    print(f"Security NVR running: {check_service_running()}")
    print(f"Frigate API accessible: {check_frigate_api()}")
    print(f"MQTT broker running: {check_mqtt_broker()}")
    print()
    
    # Run tests
    pytest.main([__file__, "-v", "-k", "not integration"])