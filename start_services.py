#!/usr/bin/env python3.12
"""Start wildfire-watch services using docker-compose programmatically"""
import subprocess
import time
import sys

def run_docker_compose(args):
    """Run docker-compose command with proper environment"""
    cmd = ["python3.12", "-m", "compose"] + args
    env = {
        "COMPOSE_PROJECT_NAME": "wildfire-watch",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONPATH": "/home/seth/.local/lib/python3.12/site-packages"
    }
    
    try:
        result = subprocess.run(cmd, 
                              capture_output=True, 
                              text=True, 
                              cwd="/home/seth/wildfire-watch",
                              env=dict(os.environ, **env))
        
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return False
        
        print(result.stdout)
        return True
    except Exception as e:
        print(f"Exception: {e}")
        return False

if __name__ == "__main__":
    import os
    os.chdir("/home/seth/wildfire-watch")
    
    print("Starting MQTT broker...")
    if not run_docker_compose(["up", "-d", "mqtt_broker"]):
        print("Failed to start MQTT broker")
        sys.exit(1)
    
    print("Waiting for MQTT broker to be healthy...")
    time.sleep(10)
    
    print("Starting camera detector...")
    if not run_docker_compose(["up", "-d", "camera_detector"]):
        print("Failed to start camera detector")
        sys.exit(1)
    
    print("Starting security NVR...")
    if not run_docker_compose(["up", "-d", "security_nvr"]):
        print("Failed to start security NVR")
        sys.exit(1)
    
    print("\nAll services started successfully!")
    print("Running docker ps to verify...")
    subprocess.run(["docker", "ps", "--format", "table {{.Names}}\t{{.Status}}"])