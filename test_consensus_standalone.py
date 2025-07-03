#!/usr/bin/env python3.12
"""
Test consensus service standalone
"""
import subprocess
import time
import json
import paho.mqtt.client as mqtt

def test_consensus_standalone():
    """Test consensus with verbose logging"""
    print("=== STANDALONE CONSENSUS TEST ===")
    
    # 1. Start mosquitto
    print("1. Starting mosquitto...")
    subprocess.run(['pkill', '-f', 'mosquitto'], capture_output=True)
    time.sleep(1)
    
    mosquitto = subprocess.Popen(['mosquitto', '-p', '1883', '-v'])
    time.sleep(2)
    
    # 2. Start consensus in subprocess with debug
    print("2. Starting consensus service...")
    consensus_proc = subprocess.Popen([
        'python3.12', 'fire_consensus/consensus.py'
    ], env={
        'MQTT_BROKER': 'localhost',
        'MQTT_PORT': '1883',
        'CONSENSUS_THRESHOLD': '1',
        'SINGLE_CAMERA_TRIGGER': 'true',
        'MIN_CONFIDENCE': '0.7',
        'LOG_LEVEL': 'DEBUG',
        'PYTHONUNBUFFERED': '1'
    })
    
    time.sleep(5)  # Give consensus time to start
    
    # 3. Monitor for trigger messages
    triggered = False
    def on_message(client, userdata, msg):
        nonlocal triggered
        print(f"TRIGGER RECEIVED: {msg.topic}")
        triggered = True
    
    monitor = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="monitor")
    monitor.on_message = on_message
    monitor.connect('localhost', 1883)
    monitor.subscribe('fire/trigger', qos=1)
    monitor.loop_start()
    
    # 4. Send test detections
    print("3. Sending test fire detections...")
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="test-sender")
    client.connect('localhost', 1883, 60)
    client.loop_start()
    
    # Send 8 growing fire detections
    for i in range(8):
        detection = {
            'camera_id': 'test_camera',
            'object': 'fire',
            'object_id': 'fire_1',
            'confidence': 0.75 + i * 0.02,
            'bounding_box': [0.1, 0.1, 0.04 + i * 0.01, 0.04 + i * 0.008],
            'timestamp': time.time() + i * 0.5
        }
        client.publish('fire/detection', json.dumps(detection), qos=1)
        print(f"  Sent detection {i+1}: confidence={detection['confidence']:.2f}, area={detection['bounding_box'][2]*detection['bounding_box'][3]:.6f}")
        time.sleep(0.5)
    
    # 5. Wait and check
    print("4. Waiting for consensus trigger...")
    time.sleep(10)
    
    # Check result
    if triggered:
        print("✅ SUCCESS: Fire trigger received!")
    else:
        print("❌ FAILED: No fire trigger received")
    
    # Cleanup
    client.loop_stop()
    client.disconnect()
    monitor.loop_stop()
    monitor.disconnect()
    consensus_proc.terminate()
    mosquitto.terminate()
    subprocess.run(['pkill', '-f', 'mosquitto'], capture_output=True)
    
    return triggered

if __name__ == "__main__":
    success = test_consensus_standalone()
    exit(0 if success else 1)