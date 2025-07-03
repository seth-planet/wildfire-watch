#!/usr/bin/env python3.12
"""
Simple test to verify services are publishing health messages
"""
import time
import docker
import paho.mqtt.client as mqtt


def test_service_health_direct():
    """Test services publishing health without namespace complexity"""
    docker_client = docker.from_env()
    
    # Find running containers
    containers = docker_client.containers.list()
    print(f"Running containers: {[c.name for c in containers]}")
    
    # Set up MQTT client to monitor all topics
    messages = []
    
    def on_message(client, userdata, msg):
        print(f"Message: {msg.topic}")
        messages.append(msg.topic)
    
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "health_monitor")
    client.on_message = on_message
    
    # Try multiple ports where broker might be running
    ports = [1883, 11883, 20000]
    connected = False
    
    for port in ports:
        try:
            client.connect("localhost", port)
            print(f"Connected to MQTT on port {port}")
            connected = True
            break
        except:
            continue
    
    if not connected:
        print("Could not connect to any MQTT broker")
        return
    
    # Subscribe to all topics
    client.subscribe("#")
    client.loop_start()
    
    # Wait and collect messages
    print("Monitoring MQTT for 30 seconds...")
    time.sleep(30)
    
    client.loop_stop()
    client.disconnect()
    
    # Analyze messages
    health_topics = [t for t in messages if 'health' in t or 'telemetry' in t]
    print(f"\nReceived {len(messages)} total messages")
    print(f"Health/telemetry topics: {health_topics}")
    
    # Check for expected topics
    expected = ['camera_detector_health', 'consensus_telemetry', 'trigger_telemetry']
    for exp in expected:
        found = any(exp in t for t in health_topics)
        print(f"{exp}: {'✅ Found' if found else '❌ Not found'}")


if __name__ == "__main__":
    test_service_health_direct()