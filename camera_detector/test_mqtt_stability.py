#!/usr/bin/env python3.12
"""Test script for MQTT stability improvements"""

import time
import threading
import logging
import json
from datetime import datetime, timezone

from mqtt_stability_fix import StableMQTTHandler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def simulate_blocking_operation(duration: float):
    """Simulate a blocking network operation"""
    logger.info(f"Starting blocking operation for {duration}s")
    time.sleep(duration)
    logger.info("Blocking operation complete")

def test_stability():
    """Test MQTT stability during blocking operations"""
    
    # Create MQTT handler
    handler = StableMQTTHandler(
        broker="localhost",  # Change to your broker
        port=1883,
        client_id="test-stability",
        keepalive=30
    )
    
    # Track connection state
    connection_events = []
    
    def on_connect():
        event = {
            'type': 'connect',
            'time': datetime.now(timezone.utc).isoformat()
        }
        connection_events.append(event)
        logger.info("MQTT Connected")
        
    def on_disconnect():
        event = {
            'type': 'disconnect', 
            'time': datetime.now(timezone.utc).isoformat()
        }
        connection_events.append(event)
        logger.warning("MQTT Disconnected")
    
    handler.on_connect_callback = on_connect
    handler.on_disconnect_callback = on_disconnect
    
    # Start handler
    handler.start()
    
    # Wait for connection
    if not handler.wait_for_connection(timeout=5.0):
        logger.error("Failed to connect to MQTT broker")
        return
    
    # Test 1: Rapid publishing
    logger.info("\n=== Test 1: Rapid Publishing ===")
    start_time = time.time()
    for i in range(100):
        payload = json.dumps({
            'test': 'rapid',
            'index': i,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        handler.publish(f"test/rapid/{i}", payload)
    
    elapsed = time.time() - start_time
    logger.info(f"Published 100 messages in {elapsed:.2f}s")
    logger.info(f"Queue size: {handler.message_queue.qsize()}")
    
    # Wait for queue to drain
    time.sleep(2)
    
    # Test 2: Publishing during blocking operation
    logger.info("\n=== Test 2: Publishing During Blocking Operation ===")
    
    # Start blocking operation in main thread
    blocking_thread = threading.Thread(
        target=simulate_blocking_operation,
        args=(35,)  # Longer than keepalive
    )
    blocking_thread.start()
    
    # Publish messages while blocked
    for i in range(10):
        payload = json.dumps({
            'test': 'blocking',
            'index': i,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        handler.publish(f"test/blocking/{i}", payload)
        time.sleep(1)
    
    blocking_thread.join()
    
    # Check if still connected
    if handler.is_connected():
        logger.info("✅ MQTT remained connected during blocking operation!")
    else:
        logger.error("❌ MQTT disconnected during blocking operation")
    
    # Test 3: Queue overflow handling
    logger.info("\n=== Test 3: Queue Overflow Handling ===")
    overflow_count = 0
    for i in range(2000):  # More than queue size
        payload = json.dumps({
            'test': 'overflow',
            'index': i,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        if not handler.publish(f"test/overflow/{i}", payload):
            overflow_count += 1
    
    logger.info(f"Queue overflow count: {overflow_count}")
    logger.info(f"Queue size: {handler.message_queue.qsize()}")
    
    # Wait and check final state
    time.sleep(5)
    
    # Print connection events
    logger.info("\n=== Connection Events ===")
    for event in connection_events:
        logger.info(f"{event['type']} at {event['time']}")
    
    # Calculate uptime
    if handler.is_connected() and connection_events:
        first_connect = next((e for e in connection_events if e['type'] == 'connect'), None)
        if first_connect:
            uptime = (datetime.now(timezone.utc) - 
                     datetime.fromisoformat(first_connect['time'].replace('Z', '+00:00')))
            logger.info(f"\nTotal uptime: {uptime}")
    
    # Cleanup
    handler.stop()
    logger.info("\nTest complete")

def test_reconnection():
    """Test MQTT reconnection behavior"""
    logger.info("\n=== Testing Reconnection Behavior ===")
    
    handler = StableMQTTHandler(
        broker="localhost",
        port=1883,
        client_id="test-reconnect",
        keepalive=10  # Short for testing
    )
    
    reconnect_count = 0
    
    def on_connect():
        nonlocal reconnect_count
        reconnect_count += 1
        logger.info(f"Connected (count: {reconnect_count})")
        
    def on_disconnect():
        logger.warning("Disconnected")
    
    handler.on_connect_callback = on_connect
    handler.on_disconnect_callback = on_disconnect
    
    # Start handler
    handler.start()
    
    # Wait for initial connection
    if not handler.wait_for_connection(timeout=5.0):
        logger.error("Failed to connect to MQTT broker")
        return
    
    logger.info("Initial connection established")
    
    # Simulate broker restart by disconnecting
    logger.info("Simulating broker disconnect...")
    handler.client.disconnect()
    
    # Wait for reconnection
    time.sleep(15)
    
    if handler.is_connected():
        logger.info(f"✅ Successfully reconnected! Total connections: {reconnect_count}")
    else:
        logger.error("❌ Failed to reconnect")
    
    handler.stop()

if __name__ == "__main__":
    print("Testing MQTT Stability Improvements")
    print("===================================\n")
    
    # Run stability test
    test_stability()
    
    # Run reconnection test
    test_reconnection()
    
    print("\nAll tests complete!")