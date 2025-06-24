#!/usr/bin/env python3.12
"""
Simple debug to understand consensus service issue
"""
import subprocess
import time
import json
import paho.mqtt.client as mqtt

def test_simple_consensus():
    """Test consensus with verbose output"""
    print("=== SIMPLE CONSENSUS DEBUG ===")
    
    # 1. Start mosquitto
    print("1. Starting mosquitto...")
    subprocess.run(['pkill', '-f', 'mosquitto'], capture_output=True)
    time.sleep(1)
    
    mosquitto = subprocess.Popen(['mosquitto', '-p', '1883', '-v'])
    time.sleep(2)
    
    # 2. Start consensus with debug output
    print("2. Starting consensus service with debug...")
    consensus_proc = subprocess.Popen([
        'python3.12', 'fire_consensus/consensus.py'
    ], env={
        'MQTT_BROKER': 'localhost',
        'MQTT_PORT': '1883', 
        'CONSENSUS_THRESHOLD': '1',
        'SINGLE_CAMERA_TRIGGER': 'true',
        'MIN_CONFIDENCE': '0.7',
        'LOG_LEVEL': 'DEBUG',
        'PYTHONUNBUFFERED': '1'  # Force unbuffered output
    }, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    time.sleep(5)  # Give consensus time to connect
    
    # 3. Send test messages
    print("3. Sending test fire detections...")
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="debug-sender")
    client.connect('localhost', 1883, 60)
    client.loop_start()
    
    for i in range(8):
        detection = {
            'camera_id': 'test_cam',
            'object': 'fire',
            'object_id': 'fire_1',
            'confidence': 0.8 + i * 0.01,
            'bounding_box': [0.1, 0.1, 0.05 + i * 0.01, 0.05 + i * 0.008],
            'timestamp': time.time() + i * 0.5
        }
        client.publish('fire/detection', json.dumps(detection), qos=1)
        print(f"  Sent detection {i+1}")
        time.sleep(0.5)
    
    # 4. Monitor for output
    print("4. Checking consensus output...")
    time.sleep(5)
    
    # Get consensus output
    try:
        output, _ = consensus_proc.communicate(timeout=2)
        print("Consensus output:")
        print(output)
    except subprocess.TimeoutExpired:
        consensus_proc.terminate()
        output = consensus_proc.stdout.read()
        print("Consensus output (partial):")
        print(output)
    
    # Cleanup
    client.loop_stop()
    client.disconnect()
    consensus_proc.terminate()
    mosquitto.terminate()
    subprocess.run(['pkill', '-f', 'mosquitto'], capture_output=True)
    
    print("\nDone.")

if __name__ == "__main__":
    test_simple_consensus()