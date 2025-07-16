#!/usr/bin/env python3.12
"""Test connectivity from inside container"""
import os
import socket
import sys
import time
import paho.mqtt.client as mqtt

def test_connectivity():
    broker = os.environ.get('MQTT_BROKER', 'localhost')
    port = int(os.environ.get('MQTT_PORT', '1883'))
    
    print(f"\n=== Container Connectivity Test ===")
    print(f"MQTT_BROKER: {broker}")
    print(f"MQTT_PORT: {port}")
    print(f"MQTT_TLS: {os.environ.get('MQTT_TLS', 'not set')}")
    
    # Test DNS
    print(f"\n1. DNS Resolution:")
    try:
        ip = socket.gethostbyname(broker)
        print(f"   ✓ {broker} -> {ip}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Test port
    print(f"\n2. Port Connectivity:")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((broker, port))
        sock.close()
        if result == 0:
            print(f"   ✓ Port {port} is open")
        else:
            print(f"   ✗ Port {port} is closed (error: {result})")
            return False
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Test MQTT
    print(f"\n3. MQTT Connection:")
    connected = False
    
    def on_connect(client, userdata, flags, rc, properties=None):
        nonlocal connected
        if rc == 0:
            print(f"   ✓ Connected successfully!")
            connected = True
        else:
            print(f"   ✗ Connection failed with code: {rc}")
    
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="container_test")
    client.on_connect = on_connect
    
    try:
        print(f"   Attempting connection...")
        client.connect(broker, port, 60)
        client.loop_start()
        
        start_time = time.time()
        while not connected and time.time() - start_time < 10:
            time.sleep(0.1)
        
        client.loop_stop()
        client.disconnect()
        
        if connected:
            print(f"   ✓ MQTT test passed")
            return True
        else:
            print(f"   ✗ MQTT connection timeout")
            return False
            
    except Exception as e:
        print(f"   ✗ MQTT error: {e}")
        return False

if __name__ == "__main__":
    success = test_connectivity()
    sys.exit(0 if success else 1)