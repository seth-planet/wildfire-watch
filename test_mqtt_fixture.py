#!/usr/bin/env python3.12
"""Test MQTT broker fixture directly"""
import sys
sys.path.insert(0, 'tests')

try:
    print("Importing mqtt_test_broker...")
    from mqtt_test_broker import TestMQTTBroker
    print("Import successful!")
    
    print("\nCreating broker instance...")
    broker = TestMQTTBroker()
    print(f"Broker created: {broker}")
    
    print("\nStarting broker...")
    broker.start()
    print("Broker started!")
    
    print(f"\nIs running: {broker.is_running()}")
    print(f"Connection params: {broker.get_connection_params()}")
    
    print("\nStopping broker...")
    broker.stop()
    print("Broker stopped!")
    
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()