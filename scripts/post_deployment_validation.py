#!/usr/bin/env python3.12
"""Post-deployment validation script for Wildfire Watch services.

This script performs comprehensive validation after deploying refactored services,
checking health reporting, message flow, and system functionality.
"""

import json
import time
import sys
import argparse
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
import paho.mqtt.client as mqtt
from collections import defaultdict
import statistics

class PostDeploymentValidator:
    """Validates Wildfire Watch services after deployment."""
    
    def __init__(self, mqtt_broker: str, mqtt_port: int, test_duration: int = 60):
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.test_duration = test_duration
        
        # Test results
        self.health_messages: Dict[str, List[dict]] = defaultdict(list)
        self.message_latencies: Dict[str, List[float]] = defaultdict(list)
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
        # Expected services
        self.expected_services = {
            'camera_detector', 'fire_consensus', 'gpio_trigger'
        }
        
        # MQTT client
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "validator")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        
    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """Handle MQTT connection."""
        if rc == 0:
            print("✓ Connected to MQTT broker")
            # Subscribe to all relevant topics
            topics = [
                ('system/+/health', 1),
                ('system/+/lwt', 1),
                ('system/trigger_telemetry', 1),  # Legacy topic
                ('fire/+', 1),
                ('cameras/+', 1),
                ('trigger/+', 1)
            ]
            for topic, qos in topics:
                client.subscribe(topic, qos)
                print(f"  Subscribed to: {topic}")
        else:
            self.errors.append(f"Failed to connect to MQTT: {rc}")
            
    def _on_message(self, client, userdata, msg):
        """Process received messages."""
        try:
            topic = msg.topic
            
            # Track message receipt time
            receipt_time = time.time()
            
            # Parse payload
            try:
                payload = json.loads(msg.payload.decode())
            except:
                payload = msg.payload.decode()
                
            # Handle health messages
            if '/health' in topic or topic == 'system/trigger_telemetry':
                service = self._extract_service_name(topic)
                if service:
                    # Check for timestamp in payload
                    if isinstance(payload, dict) and 'timestamp' in payload:
                        try:
                            msg_time = datetime.fromisoformat(payload['timestamp'].replace('Z', '+00:00'))
                            latency = receipt_time - msg_time.timestamp()
                            self.message_latencies[service].append(latency)
                        except:
                            pass
                    
                    self.health_messages[service].append({
                        'topic': topic,
                        'payload': payload,
                        'time': receipt_time
                    })
                    
            # Check for errors in payload
            if isinstance(payload, dict):
                if payload.get('error') or payload.get('state') == 'ERROR':
                    self.errors.append(f"Error state in {topic}: {payload}")
                    
        except Exception as e:
            self.errors.append(f"Error processing message from {msg.topic}: {e}")
            
    def _extract_service_name(self, topic: str) -> Optional[str]:
        """Extract service name from topic."""
        if topic == 'system/trigger_telemetry':
            return 'gpio_trigger'
        elif '/health' in topic:
            parts = topic.split('/')
            if len(parts) >= 3:
                return parts[1]
        return None
        
    def run_validation(self) -> Dict[str, any]:
        """Run the validation tests."""
        print(f"\nStarting post-deployment validation for {self.test_duration} seconds...")
        
        # Connect to MQTT
        try:
            self.client.connect(self.mqtt_broker, self.mqtt_port, 60)
            self.client.loop_start()
        except Exception as e:
            self.errors.append(f"Failed to connect to MQTT broker: {e}")
            return self._generate_report()
            
        # Wait and collect data
        start_time = time.time()
        while time.time() - start_time < self.test_duration:
            elapsed = int(time.time() - start_time)
            print(f"\rCollecting data... {elapsed}/{self.test_duration}s", end='', flush=True)
            time.sleep(1)
            
        print("\n\nAnalyzing results...")
        
        # Disconnect
        self.client.loop_stop()
        self.client.disconnect()
        
        return self._generate_report()
        
    def _generate_report(self) -> Dict[str, any]:
        """Generate validation report."""
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'test_duration': self.test_duration,
            'overall_status': 'PASS',
            'services': {},
            'errors': self.errors,
            'warnings': self.warnings,
            'summary': {}
        }
        
        # Check each expected service
        for service in self.expected_services:
            service_report = self._validate_service(service)
            report['services'][service] = service_report
            
            if service_report['status'] != 'HEALTHY':
                report['overall_status'] = 'FAIL'
                
        # Add summary statistics
        report['summary'] = {
            'services_healthy': sum(1 for s in report['services'].values() if s['status'] == 'HEALTHY'),
            'services_total': len(self.expected_services),
            'total_health_messages': sum(len(msgs) for msgs in self.health_messages.values()),
            'total_errors': len(self.errors),
            'total_warnings': len(self.warnings)
        }
        
        return report
        
    def _validate_service(self, service: str) -> Dict[str, any]:
        """Validate individual service."""
        messages = self.health_messages.get(service, [])
        
        if not messages:
            return {
                'status': 'MISSING',
                'message_count': 0,
                'details': 'No health messages received'
            }
            
        # Calculate message interval
        if len(messages) > 1:
            intervals = []
            for i in range(1, len(messages)):
                interval = messages[i]['time'] - messages[i-1]['time']
                intervals.append(interval)
            avg_interval = statistics.mean(intervals)
        else:
            avg_interval = None
            
        # Check for standard vs legacy topics
        topics_used = list(set(msg['topic'] for msg in messages))
        using_standard = any('health' in topic for topic in topics_used)
        using_legacy = 'system/trigger_telemetry' in topics_used
        
        # Calculate latencies
        latencies = self.message_latencies.get(service, [])
        avg_latency = statistics.mean(latencies) if latencies else None
        
        # Determine status
        status = 'HEALTHY'
        details = []
        
        if len(messages) < 2:
            status = 'DEGRADED'
            details.append('Insufficient messages for interval calculation')
            
        if avg_interval and avg_interval > 120:
            status = 'DEGRADED'
            details.append(f'Health interval too long: {avg_interval:.1f}s')
            
        if service == 'gpio_trigger' and using_legacy and not using_standard:
            self.warnings.append(f'{service} still using legacy topic')
            details.append('Using legacy health topic')
            
        if avg_latency and avg_latency > 1.0:
            self.warnings.append(f'{service} high message latency: {avg_latency:.2f}s')
            
        return {
            'status': status,
            'message_count': len(messages),
            'topics': topics_used,
            'using_standard_topic': using_standard,
            'using_legacy_topic': using_legacy,
            'average_interval': avg_interval,
            'average_latency': avg_latency,
            'details': ' | '.join(details) if details else 'All checks passed'
        }


def print_report(report: Dict[str, any]):
    """Print formatted validation report."""
    print("\n" + "="*60)
    print("POST-DEPLOYMENT VALIDATION REPORT")
    print("="*60)
    print(f"Timestamp: {report['timestamp']}")
    print(f"Duration: {report['test_duration']}s")
    print(f"Overall Status: {report['overall_status']}")
    print()
    
    # Service status
    print("SERVICE STATUS:")
    print("-"*60)
    for service, status in report['services'].items():
        status_icon = "✓" if status['status'] == 'HEALTHY' else "✗"
        print(f"{status_icon} {service:20} {status['status']:10} ({status['message_count']} messages)")
        if status['using_standard_topic']:
            print(f"  - Using standard health topic")
        if status['using_legacy_topic']:
            print(f"  - Using legacy topic (migration needed)")
        if status['average_interval']:
            print(f"  - Avg interval: {status['average_interval']:.1f}s")
        if status['average_latency']:
            print(f"  - Avg latency: {status['average_latency']:.3f}s")
        if status['details'] and status['details'] != 'All checks passed':
            print(f"  - {status['details']}")
        print()
    
    # Summary
    print("SUMMARY:")
    print("-"*60)
    summary = report['summary']
    print(f"Services Healthy: {summary['services_healthy']}/{summary['services_total']}")
    print(f"Health Messages: {summary['total_health_messages']}")
    print(f"Errors: {summary['total_errors']}")
    print(f"Warnings: {summary['total_warnings']}")
    
    # Errors and warnings
    if report['errors']:
        print("\nERRORS:")
        print("-"*60)
        for error in report['errors']:
            print(f"✗ {error}")
            
    if report['warnings']:
        print("\nWARNINGS:")
        print("-"*60)
        for warning in report['warnings']:
            print(f"⚠ {warning}")
            
    print("\n" + "="*60)
    
    # Save detailed report
    report_file = f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed report saved to: {report_file}")
    

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Post-deployment validation for Wildfire Watch')
    parser.add_argument('--broker', default='localhost', help='MQTT broker address')
    parser.add_argument('--port', type=int, default=1883, help='MQTT broker port')
    parser.add_argument('--duration', type=int, default=60, help='Test duration in seconds')
    parser.add_argument('--json', action='store_true', help='Output JSON report only')
    
    args = parser.parse_args()
    
    # Run validation
    validator = PostDeploymentValidator(args.broker, args.port, args.duration)
    report = validator.run_validation()
    
    # Output results
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)
        
    # Exit code based on status
    sys.exit(0 if report['overall_status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()