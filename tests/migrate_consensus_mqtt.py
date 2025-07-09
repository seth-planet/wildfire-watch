#!/usr/bin/env python3.12
"""
Script to migrate test_consensus.py to use optimized MQTT infrastructure
"""

import re
import sys
from pathlib import Path

def migrate_consensus_tests():
    """Migrate consensus tests to use new MQTT infrastructure"""
    
    test_file = Path(__file__).parent / "test_consensus.py"
    
    # Read the current file
    with open(test_file, 'r') as f:
        content = f.read()
    
    # Backup original
    backup_file = test_file.with_suffix('.py.backup')
    with open(backup_file, 'w') as f:
        f.write(content)
    
    print(f"Created backup: {backup_file}")
    
    # Migration steps
    replacements = [
        # 1. Remove class-scoped broker fixture
        (
            r'@pytest\.fixture\(scope="class"\)\s*\ndef class_mqtt_broker\(\):[^}]+yield broker[^}]+broker\.stop\(\)',
            ''
        ),
        
        # 2. Replace class_mqtt_broker with test_mqtt_broker
        (
            r'class_mqtt_broker',
            'test_mqtt_broker'
        ),
        
        # 3. Remove mqtt_test_broker import
        (
            r'from mqtt_test_broker import MQTTTestBroker\s*\n',
            ''
        ),
        
        # 4. Update fixture parameters
        (
            r'def (mqtt_publisher|trigger_monitor|message_monitor|consensus_service)\(test_mqtt_broker\)',
            r'def \1(test_mqtt_broker, mqtt_topic_factory)'
        ),
        
        # 5. Add topic isolation to fixtures
        (
            r'def mqtt_publisher\(test_mqtt_broker, mqtt_topic_factory\):',
            '''def mqtt_publisher(test_mqtt_broker, mqtt_topic_factory):
    """Create MQTT publisher for test message injection with topic isolation"""'''
        ),
    ]
    
    # Apply replacements
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
    
    # Add topic mapping for consensus service
    consensus_service_update = '''@pytest.fixture
def consensus_service(test_mqtt_broker, mqtt_topic_factory, monkeypatch):
    """Create FireConsensus service with real MQTT broker and topic isolation"""
    # Get connection parameters from the test broker
    conn_params = test_mqtt_broker.get_connection_params()
    
    # Map topics for isolation
    fire_topic = mqtt_topic_factory("fire/detection")
    trigger_topic = mqtt_topic_factory("trigger/fire_detected")
    telemetry_topic = mqtt_topic_factory("system/camera_telemetry")
    
    # Set topic prefix for service
    topic_prefix = fire_topic.rsplit('/', 2)[0]  # Get test/worker_id prefix
    monkeypatch.setenv("TOPIC_PREFIX", topic_prefix)'''
    
    # Replace the consensus_service fixture definition
    content = re.sub(
        r'@pytest\.fixture\s*\ndef consensus_service\(test_mqtt_broker, mqtt_topic_factory, monkeypatch\):[^@]+?(?=@pytest\.fixture|\nclass\s|\ndef\s|\Z)',
        consensus_service_update + '\n\n    ',
        content,
        flags=re.MULTILINE | re.DOTALL
    )
    
    # Write migrated content
    with open(test_file, 'w') as f:
        f.write(content)
    
    print(f"Migration complete: {test_file}")
    print("\nKey changes:")
    print("1. Removed class-scoped broker fixture")
    print("2. Updated all fixtures to use test_mqtt_broker")
    print("3. Added mqtt_topic_factory for topic isolation")
    print("4. Updated consensus_service to use isolated topics")
    
    return True

def verify_migration():
    """Verify the migration was successful"""
    test_file = Path(__file__).parent / "test_consensus.py"
    
    with open(test_file, 'r') as f:
        content = f.read()
    
    issues = []
    
    # Check for remaining class_mqtt_broker references
    if 'class_mqtt_broker' in content:
        issues.append("Found remaining class_mqtt_broker references")
    
    # Check for MQTTTestBroker usage
    if 'MQTTTestBroker()' in content:
        issues.append("Found direct MQTTTestBroker instantiation")
    
    # Check for proper fixture usage
    if not re.search(r'def \w+\(.*test_mqtt_broker.*\)', content):
        issues.append("No fixtures using test_mqtt_broker found")
    
    if issues:
        print("\nVerification issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\nVerification passed!")
        return True

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        verify_migration()
    else:
        if migrate_consensus_tests():
            verify_migration()