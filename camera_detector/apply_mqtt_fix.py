#!/usr/bin/env python3.12
"""Apply MQTT stability fix to detect.py

This script modifies detect.py to use the stable MQTT implementation.
Run this to update the camera detector with the stability improvements.
"""

import os
import sys
import shutil
from datetime import datetime

def apply_mqtt_stability_fix():
    """Apply the MQTT stability fix to detect.py"""
    
    detect_file = "detect.py"
    backup_file = f"detect.py.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if not os.path.exists(detect_file):
        print(f"Error: {detect_file} not found in current directory")
        return False
    
    # Create backup
    print(f"Creating backup: {backup_file}")
    shutil.copy2(detect_file, backup_file)
    
    # Read the original file
    with open(detect_file, 'r') as f:
        content = f.read()
    
    # Define the modifications
    modifications = []
    
    # 1. Add import for StableMQTTHandler after other imports
    import_marker = "from utils.command_runner import run_command, CommandError"
    import_addition = """from utils.command_runner import run_command, CommandError

# Import stable MQTT handler
try:
    from mqtt_stability_fix import StableMQTTHandler
    STABLE_MQTT_AVAILABLE = True
except ImportError:
    STABLE_MQTT_AVAILABLE = False
    safe_log("StableMQTTHandler not available, using standard MQTT", logging.WARNING)"""
    
    modifications.append((import_marker, import_addition))
    
    # 2. Replace _setup_mqtt method
    setup_mqtt_start = '    def _setup_mqtt(self):\n        """Setup MQTT client with resilient connection"""'
    setup_mqtt_new = '''    def _setup_mqtt(self):
        """Setup MQTT client with resilient connection"""
        if STABLE_MQTT_AVAILABLE:
            # Use stable MQTT handler
            self.mqtt_handler = StableMQTTHandler(
                broker=self.config.MQTT_BROKER,
                port=self.config.MQTT_PORT,
                client_id=self.config.SERVICE_ID,
                keepalive=30,  # Reduced from 60 for faster detection
                tls_enabled=self.config.MQTT_TLS,
                ca_cert_path=self.config.TLS_CA_PATH if self.config.MQTT_TLS else None
            )
            
            # Keep mqtt_client reference for compatibility
            self.mqtt_client = self.mqtt_handler.client
            
            # Set callbacks
            self.mqtt_handler.on_connect_callback = lambda: self._on_mqtt_connect(None, None, None, 0)
            self.mqtt_handler.on_disconnect_callback = lambda: self._on_mqtt_disconnect(None, None, 1)
            
            # Set LWT
            lwt_topic = f"{self.config.TOPIC_HEALTH}/{self.config.NODE_ID}/lwt"
            lwt_payload = json.dumps({
                'node_id': self.config.NODE_ID,
                'service': 'camera_detector',
                'status': 'offline',
                'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            })
            self.mqtt_handler.set_will(lwt_topic, lwt_payload, qos=1, retain=True)
            
            # Start handler
            self.mqtt_handler.start()
            
            # Wait for initial connection
            if not self.mqtt_handler.wait_for_connection(timeout=10.0):
                safe_log("Initial MQTT connection timeout - running in degraded mode", logging.WARNING)
            
            return'''
    
    # Find the end of the original _setup_mqtt method
    setup_mqtt_end_marker = "self._mqtt_connect_with_retry()"
    
    # 3. Add mqtt_publish method after _setup_mqtt
    mqtt_publish_method = '''
    
    def mqtt_publish(self, topic: str, payload, qos: int = 0, retain: bool = False) -> bool:
        """Thread-safe publish to MQTT"""
        if STABLE_MQTT_AVAILABLE and hasattr(self, 'mqtt_handler'):
            # Use stable handler
            if not isinstance(payload, str):
                payload = json.dumps(payload)
            return self.mqtt_handler.publish(topic, payload, qos, retain)
        else:
            # Fallback to standard client
            try:
                if not isinstance(payload, (str, bytes)):
                    payload = json.dumps(payload)
                result = self.mqtt_client.publish(topic, payload, qos=qos, retain=retain)
                return result.rc == mqtt.MQTT_ERR_SUCCESS
            except Exception as e:
                safe_log(f"MQTT publish error: {e}", logging.ERROR)
                return False'''
    
    # 4. Replace mqtt_client.publish calls
    publish_replacements = [
        ('self.mqtt_client.publish(', 'self.mqtt_publish('),
        ('.publish(', '.mqtt_publish('),
    ]
    
    # 5. Update cleanup method for MQTT handler
    cleanup_mqtt_old = '''        # Disconnect MQTT
        try:
            if self.mqtt_client:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
                # Clear the reference to prevent further use
                self.mqtt_client = None
        except Exception as e:
            safe_log(f"Error during MQTT cleanup: {e}")'''
    
    cleanup_mqtt_new = '''        # Disconnect MQTT
        try:
            if STABLE_MQTT_AVAILABLE and hasattr(self, 'mqtt_handler'):
                # Publish final offline status
                lwt_payload = json.dumps({
                    'node_id': self.config.NODE_ID,
                    'service': 'camera_detector',
                    'status': 'offline',
                    'timestamp': datetime.now(timezone.utc).isoformat() + 'Z'
                })
                self.mqtt_publish(
                    f"{self.config.TOPIC_HEALTH}/{self.config.NODE_ID}/lwt",
                    lwt_payload,
                    qos=1,
                    retain=True
                )
                time.sleep(0.5)  # Allow final message to send
                self.mqtt_handler.stop()
            elif self.mqtt_client:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
                self.mqtt_client = None
        except Exception as e:
            safe_log(f"Error during MQTT cleanup: {e}")'''
    
    # Apply modifications
    modified_content = content
    
    # Apply import addition
    modified_content = modified_content.replace(import_marker, import_addition)
    
    # Apply _setup_mqtt replacement
    # Find the complete method
    setup_start_idx = modified_content.find(setup_mqtt_start)
    if setup_start_idx != -1:
        # Find the end of the method (next method definition or class end)
        setup_end_idx = modified_content.find("\n    def _mqtt_connect_with_retry", setup_start_idx)
        if setup_end_idx != -1:
            # Replace the method
            modified_content = (modified_content[:setup_start_idx] + 
                              setup_mqtt_new + 
                              modified_content[setup_end_idx:])
            
            # Add mqtt_publish method after _setup_mqtt
            insert_idx = modified_content.find("\n    def _mqtt_connect_with_retry")
            if insert_idx != -1:
                modified_content = (modified_content[:insert_idx] + 
                                  mqtt_publish_method + 
                                  modified_content[insert_idx:])
    
    # Apply publish replacements
    for old, new in publish_replacements:
        # Skip the mqtt_publish method itself
        lines = modified_content.split('\n')
        new_lines = []
        in_mqtt_publish = False
        
        for line in lines:
            if 'def mqtt_publish(' in line:
                in_mqtt_publish = True
            elif line.strip().startswith('def ') and in_mqtt_publish:
                in_mqtt_publish = False
            
            if not in_mqtt_publish and old in line and 'mqtt_publish' not in line:
                line = line.replace(old, new)
            new_lines.append(line)
        
        modified_content = '\n'.join(new_lines)
    
    # Apply cleanup replacement
    modified_content = modified_content.replace(cleanup_mqtt_old, cleanup_mqtt_new)
    
    # Write the modified content
    with open(detect_file, 'w') as f:
        f.write(modified_content)
    
    print(f"✅ Successfully applied MQTT stability fix to {detect_file}")
    print(f"   Backup saved as: {backup_file}")
    print("\nChanges made:")
    print("- Added StableMQTTHandler import")
    print("- Modified _setup_mqtt to use stable handler")
    print("- Added mqtt_publish method for thread-safe publishing")
    print("- Updated all publish calls to use mqtt_publish")
    print("- Enhanced cleanup method for proper shutdown")
    
    return True

def verify_fix():
    """Verify the fix was applied correctly"""
    with open("detect.py", 'r') as f:
        content = f.read()
    
    checks = [
        ("StableMQTTHandler import", "from mqtt_stability_fix import StableMQTTHandler" in content),
        ("mqtt_publish method", "def mqtt_publish(self" in content),
        ("Stable handler in _setup_mqtt", "self.mqtt_handler = StableMQTTHandler" in content),
        ("Updated cleanup", "mqtt_handler.stop()" in content),
    ]
    
    print("\n📋 Verification:")
    all_good = True
    for check_name, result in checks:
        status = "✅" if result else "❌"
        print(f"   {status} {check_name}")
        if not result:
            all_good = False
    
    return all_good

if __name__ == "__main__":
    print("MQTT Stability Fix Applicator")
    print("=============================\n")
    
    # Check if mqtt_stability_fix.py exists
    if not os.path.exists("mqtt_stability_fix.py"):
        print("Error: mqtt_stability_fix.py not found!")
        print("Please ensure mqtt_stability_fix.py is in the same directory")
        sys.exit(1)
    
    # Apply the fix
    if apply_mqtt_stability_fix():
        # Verify the fix
        if verify_fix():
            print("\n✨ All modifications verified successfully!")
            print("\nNext steps:")
            print("1. Test the modified detect.py in development")
            print("2. Monitor MQTT connection stability")
            print("3. Check logs for any issues")
        else:
            print("\n⚠️  Some verifications failed. Please check the modifications.")
    else:
        print("\n❌ Failed to apply modifications")