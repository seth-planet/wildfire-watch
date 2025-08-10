#!/usr/bin/env python3.12
"""Check which ports are being used by mosquitto processes"""
import subprocess
import re

def check_mqtt_ports():
    """Check all mosquitto processes and their ports"""
    print("Checking mosquitto processes and ports...")
    
    # Get all mosquitto processes
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        mosquitto_procs = [line for line in result.stdout.splitlines() if 'mosquitto' in line and 'grep' not in line]
        
        print(f"\nFound {len(mosquitto_procs)} mosquitto processes:")
        for proc in mosquitto_procs:
            print(f"  {proc}")
            
        # Get network connections for mosquitto
        result = subprocess.run(['ss', '-tlnp'], capture_output=True, text=True)
        if result.returncode == 0:
            print("\nMosquitto network listeners:")
            for line in result.stdout.splitlines():
                if 'mosquitto' in line:
                    print(f"  {line}")
        
        # Check specific ports
        print("\nChecking common MQTT ports:")
        for port in [1883, 8883, 11883, 20000, 30000]:
            result = subprocess.run(['lsof', '-i', f':{port}'], capture_output=True, text=True)
            if result.stdout.strip():
                print(f"  Port {port} is in use:")
                print(f"    {result.stdout.strip()}")
            else:
                print(f"  Port {port} is free")
                
    except Exception as e:
        print(f"Error checking ports: {e}")

if __name__ == "__main__":
    check_mqtt_ports()