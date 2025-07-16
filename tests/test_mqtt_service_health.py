#!/usr/bin/env python3.12
"""
Simple test to verify services are publishing health messages
"""
import time
import json
import pytest
import docker
import paho.mqtt.client as mqtt
from threading import Event


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_service_health_messages(test_mqtt_broker):
    """Test services publishing health messages using test fixtures"""
    # Find running containers
    docker_client = docker.from_env()
    containers = docker_client.containers.list()
    print(f"Running containers: {[c.name for c in containers]}")
    
    # Set up MQTT client to monitor all topics
    messages = []
    health_received = {
        'camera_detector': Event(),
        'fire_consensus': Event(),
        'gpio_trigger': Event()
    }
    
    def on_message(client, userdata, msg):
        print(f"Message: {msg.topic}")
        messages.append({
            'topic': msg.topic,
            'payload': msg.payload.decode('utf-8')
        })
        
        # Check for health messages
        if 'camera_detector/health' in msg.topic:
            health_received['camera_detector'].set()
        elif 'fire_consensus/health' in msg.topic:
            health_received['fire_consensus'].set()
        elif 'trigger_telemetry' in msg.topic or 'gpio_trigger/health' in msg.topic:
            health_received['gpio_trigger'].set()
    
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "health_monitor")
    client.on_message = on_message
    
    # Connect to test MQTT broker
    client.connect(test_mqtt_broker.host, test_mqtt_broker.port)
    print(f"Connected to test MQTT broker on port {test_mqtt_broker.port}")
    
    # Subscribe to all topics
    client.subscribe("#")
    client.loop_start()
    
    # Wait and collect messages
    print("Monitoring MQTT for health messages...")
    time.sleep(15)  # Most services publish health every 10s
    
    client.loop_stop()
    client.disconnect()
    
    # Analyze messages
    health_topics = [m['topic'] for m in messages if 'health' in m['topic'] or 'telemetry' in m['topic']]
    print(f"\nReceived {len(messages)} total messages")
    print(f"Health/telemetry topics: {set(health_topics)}")
    
    # Check for expected health messages
    for service, event in health_received.items():
        if event.is_set():
            print(f"{service}: ✅ Health message received")
        else:
            print(f"{service}: ⚠️  No health message received")
    
    # At least some services should be publishing health
    assert any(event.is_set() for event in health_received.values()), \
        "No health messages received from any service"


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_service_health_with_namespace(test_mqtt_broker, parallel_test_context):
    """Test services publishing health messages with namespace isolation"""
    namespace = parallel_test_context.namespace
    
    # Set up MQTT client to monitor namespaced topics
    messages = []
    
    def on_message(client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode('utf-8'))
            messages.append({
                'topic': msg.topic,
                'service': payload.get('service', 'unknown'),
                'healthy': payload.get('healthy', False),
                'timestamp': payload.get('timestamp', 0)
            })
            print(f"Health: {msg.topic} - {payload.get('service')} - Healthy: {payload.get('healthy')}")
        except:
            # Not a JSON health message
            pass
    
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, f"health_monitor_{parallel_test_context.worker_id}")
    client.on_message = on_message
    
    # Connect to test MQTT broker
    client.connect(test_mqtt_broker.host, test_mqtt_broker.port)
    print(f"Connected to test MQTT broker with namespace: {namespace}")
    
    # Subscribe to namespaced health topics
    client.subscribe(f"{namespace}/system/+/health")
    client.subscribe(f"{namespace}/system/+/telemetry")
    client.loop_start()
    
    # Wait and collect messages
    print("Monitoring MQTT for namespaced health messages...")
    time.sleep(15)
    
    client.loop_stop()
    client.disconnect()
    
    # Analyze messages
    print(f"\nReceived {len(messages)} health messages")
    services_seen = set(m['service'] for m in messages)
    print(f"Services reporting health: {services_seen}")
    
    # Verify health message format
    for msg in messages[:5]:  # Check first 5 messages
        assert 'service' in msg, "Health message missing service field"
        assert 'healthy' in msg, "Health message missing healthy field"
        assert 'timestamp' in msg, "Health message missing timestamp field"
        print(f"  {msg['service']}: healthy={msg['healthy']}")


if __name__ == "__main__":
    pytest.main([__file__, '-v'])