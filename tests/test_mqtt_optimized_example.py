#!/usr/bin/env python3.12
"""
Example test file demonstrating optimized MQTT test fixtures.
Shows how to use topic isolation and client management for fast, reliable tests.
"""
import json
import time
import pytest
from threading import Event


class TestOptimizedMQTT:
    """Example tests using the optimized MQTT fixtures"""
    
    def test_basic_publish_subscribe(self, mqtt_client, mqtt_topic_factory):
        """
        Demonstrates basic publish/subscribe with topic isolation.
        Each test gets its own unique topic namespace.
        """
        # Create isolated topics for this test
        sensor_topic = mqtt_topic_factory("sensors/temperature")
        
        # Set up message capture
        received_messages = []
        message_event = Event()
        
        def on_message(client, userdata, msg):
            received_messages.append({
                'topic': msg.topic,
                'payload': msg.payload.decode()
            })
            message_event.set()
        
        # Subscribe to the isolated topic
        mqtt_client.on_message = on_message
        mqtt_client.subscribe(sensor_topic)
        
        # Publish a test message
        test_payload = json.dumps({"temperature": 25.5, "unit": "celsius"})
        mqtt_client.publish(sensor_topic, test_payload)
        
        # Wait for message (with timeout)
        assert message_event.wait(timeout=2.0), "Timeout waiting for message"
        
        # Verify message received
        assert len(received_messages) == 1
        msg = received_messages[0]
        assert msg['topic'] == sensor_topic
        assert json.loads(msg['payload']) == {"temperature": 25.5, "unit": "celsius"}
    
    def test_multiple_clients_isolated(self, session_mqtt_broker, mqtt_topic_factory):
        """
        Demonstrates that multiple clients in the same test are isolated
        from other tests running in parallel.
        """
        import paho.mqtt.client as mqtt
        import uuid
        
        # Create two clients for this test
        client1 = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, f"test_multi_1_{uuid.uuid4().hex[:8]}")
        client2 = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, f"test_multi_2_{uuid.uuid4().hex[:8]}")
        
        try:
            # Connect both clients
            client1.connect(session_mqtt_broker.host, session_mqtt_broker.port, 60)
            client2.connect(session_mqtt_broker.host, session_mqtt_broker.port, 60)
            
            client1.loop_start()
            client2.loop_start()
            
            # Use isolated topics
            chat_topic = mqtt_topic_factory("chat/room")
            
            # Set up message capture for client2
            messages = []
            msg_event = Event()
            
            def on_message(client, userdata, msg):
                messages.append(msg.payload.decode())
                msg_event.set()
            
            client2.on_message = on_message
            client2.subscribe(chat_topic)
            time.sleep(0.1)  # Allow subscription to register
            
            # Client1 publishes
            client1.publish(chat_topic, "Hello from client1")
            
            # Client2 should receive it
            assert msg_event.wait(timeout=2.0), "Message not received"
            assert messages[0] == "Hello from client1"
            
        finally:
            # Clean up both clients
            client1.loop_stop()
            client2.loop_stop()
            client1.disconnect()
            client2.disconnect()
    
    def test_qos_levels(self, mqtt_client, mqtt_topic_factory):
        """Test different QoS levels with the optimized broker"""
        control_topic = mqtt_topic_factory("control/commands")
        
        received = []
        message_event = Event()
        
        def on_message(client, userdata, msg):
            received.append({
                'payload': msg.payload.decode(),
                'qos': msg.qos
            })
            if len(received) >= 3:
                message_event.set()
        
        mqtt_client.on_message = on_message
        mqtt_client.subscribe(control_topic, qos=2)
        time.sleep(0.2)  # Give subscription time to register
        
        # Test QoS 0 (at most once)
        info0 = mqtt_client.publish(control_topic, "QoS 0 message", qos=0)
        info0.wait_for_publish()
        
        # Test QoS 1 (at least once)
        info1 = mqtt_client.publish(control_topic, "QoS 1 message", qos=1)
        info1.wait_for_publish()
        
        # Test QoS 2 (exactly once)
        info2 = mqtt_client.publish(control_topic, "QoS 2 message", qos=2)
        info2.wait_for_publish()
        
        # Wait for all messages to be received
        assert message_event.wait(timeout=5.0), f"Timeout waiting for messages. Received: {received}"
        
        # Verify all messages received
        assert len(received) >= 3, f"Expected at least 3 messages, got {len(received)}"
        payloads = [m['payload'] for m in received]
        assert "QoS 0 message" in payloads
        assert "QoS 1 message" in payloads
        assert "QoS 2 message" in payloads
    
    @pytest.mark.parametrize("payload_size", [100, 1000, 10000])
    def test_various_payload_sizes(self, mqtt_client, mqtt_topic_factory, payload_size):
        """Test that the optimized broker handles various payload sizes efficiently"""
        data_topic = mqtt_topic_factory("data/stream")
        
        # Create payload of specified size
        test_data = "x" * payload_size
        
        received_event = Event()
        received_size = None
        
        def on_message(client, userdata, msg):
            nonlocal received_size
            received_size = len(msg.payload)
            received_event.set()
        
        mqtt_client.on_message = on_message
        mqtt_client.subscribe(data_topic)
        time.sleep(0.1)
        
        # Publish the payload
        mqtt_client.publish(data_topic, test_data)
        
        # Verify reception
        assert received_event.wait(timeout=5.0), f"Timeout receiving {payload_size} byte payload"
        assert received_size == payload_size


def test_parallel_safety(mqtt_topic_factory):
    """
    This test demonstrates that topic isolation makes tests safe for parallel execution.
    Even if another test is using similar topic patterns, this test's topics are isolated.
    """
    # These topics are guaranteed to be unique to this test run
    fire_detection = mqtt_topic_factory("fire/detection")
    fire_trigger = mqtt_topic_factory("fire/trigger")
    
    # Even if another test uses the same base topics, they won't collide
    # because each test gets a unique prefix like "test/abc12345/fire/detection"
    assert fire_detection.startswith("test/")
    assert fire_trigger.startswith("test/")
    assert fire_detection.split("/")[1] == fire_trigger.split("/")[1]  # Same unique prefix


if __name__ == "__main__":
    # Run with: pytest -n auto tests/test_mqtt_optimized_example.py
    # The -n auto flag enables parallel execution across CPU cores
    pytest.main([__file__, "-v"])