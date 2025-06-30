import sys
import os
import subprocess
from pathlib import Path
import pytest
import time
import uuid
import paho.mqtt.client as mqtt

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

def has_camera_on_network():
    """Check if there's a camera on the network"""
    # Check environment variable
    if os.getenv('CAMERA_CREDENTIALS'):
        return True
    return False

def pytest_ignore_collect(path, config):
    python_version = sys.version_info
    path = Path(str(path))
    if "hailo" in path.name and python_version[:2] != (3, 10):
        return True
    if "coral" in path.name and python_version[:2] != (3, 8):
        return True
    return False


# ─────────────────────────────────────────────────────────────
# Shared MQTT Test Infrastructure
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def test_mqtt_broker():
    """Provide a test MQTT broker using enhanced broker with session reuse"""
    # Import here to avoid circular dependencies
    sys.path.insert(0, os.path.dirname(__file__))
    from enhanced_mqtt_broker import TestMQTTBroker
    
    broker = TestMQTTBroker(session_scope=True)
    broker.start()
    
    # Reset state for this test
    broker.reset_state()
    
    yield broker
    
    # Don't stop session broker (it's reused)


@pytest.fixture
def mqtt_client_factory(test_mqtt_broker):
    """Factory for creating MQTT clients with automatic cleanup"""
    clients = []
    
    def create_client(client_id=None):
        if not client_id:
            client_id = f"test_client_{uuid.uuid4().hex[:8]}"
            
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        conn_params = test_mqtt_broker.get_connection_params()
        client.connect(conn_params['host'], conn_params['port'], 60)
        client.loop_start()
        
        # Track for cleanup
        clients.append(client)
        
        # Wait for connection
        start_time = time.time()
        while not client.is_connected() and time.time() - start_time < 5:
            time.sleep(0.1)
            
        return client
    
    yield create_client
    
    # Cleanup all created clients
    for client in clients:
        try:
            client.loop_stop()
            client.disconnect()
        except:
            pass


@pytest.fixture
def mqtt_topic_factory():
    """Generate unique MQTT topics for parallel test safety"""
    def generate_topic(base_topic):
        unique_id = uuid.uuid4().hex[:8]
        return f"test_{unique_id}/{base_topic}"
    return generate_topic
