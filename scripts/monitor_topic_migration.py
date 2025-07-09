#!/usr/bin/env python3.12
"""Monitor topic migration progress from legacy to standard health topics."""

import json
import time
import sys
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, Set
import paho.mqtt.client as mqtt


class TopicMigrationMonitor:
    """Monitor and report on topic migration progress."""
    
    def __init__(self, mqtt_broker: str = "localhost", mqtt_port: int = 1883):
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        
        # Topic mappings
        self.topic_mappings = {
            'system/trigger_telemetry': 'system/gpio_trigger/health',
            # Add more legacy -> new mappings as needed
        }
        
        # Tracking data
        self.message_counts: Dict[str, int] = defaultdict(int)
        self.last_seen: Dict[str, datetime] = {}
        self.unique_publishers: Dict[str, Set[str]] = defaultdict(set)
        self.start_time = datetime.now()
        
        # MQTT setup
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "migration_monitor")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        
    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """Subscribe to all relevant topics on connection."""
        if rc == 0:
            # Subscribe to all legacy and new topics
            for legacy, new in self.topic_mappings.items():
                client.subscribe(legacy, 1)
                client.subscribe(new, 1)
            # Also monitor all health topics
            client.subscribe('system/+/health', 1)
            print("✓ Connected and subscribed to topics")
        else:
            print(f"✗ Connection failed: {rc}")
            
    def _on_message(self, client, userdata, msg):
        """Track messages on each topic."""
        topic = msg.topic
        self.message_counts[topic] += 1
        self.last_seen[topic] = datetime.now()
        
        # Try to extract publisher info
        try:
            payload = json.loads(msg.payload.decode())
            if 'service_id' in payload:
                self.unique_publishers[topic].add(payload['service_id'])
            elif 'host' in payload:
                self.unique_publishers[topic].add(payload['host'])
        except:
            pass
            
    def run(self, duration: int = 300):
        """Monitor topics for specified duration."""
        print(f"\nMonitoring topic migration for {duration} seconds...")
        print("Press Ctrl+C to stop early\n")
        
        try:
            self.client.connect(self.mqtt_broker, self.mqtt_port, 60)
            self.client.loop_start()
            
            # Monitor and update display
            end_time = time.time() + duration
            while time.time() < end_time:
                self._display_status()
                time.sleep(5)
                
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")
        finally:
            self.client.loop_stop()
            self.client.disconnect()
            
        self._generate_report()
        
    def _display_status(self):
        """Display current migration status."""
        # Clear screen and show header
        print("\033[2J\033[H")  # Clear screen and move cursor to top
        print("TOPIC MIGRATION MONITOR")
        print("=" * 80)
        print(f"Runtime: {datetime.now() - self.start_time}")
        print()
        
        # Show migration status for each mapping
        for legacy, new in self.topic_mappings.items():
            legacy_count = self.message_counts.get(legacy, 0)
            new_count = self.message_counts.get(new, 0)
            total = legacy_count + new_count
            
            if total > 0:
                migration_percent = (new_count / total) * 100
            else:
                migration_percent = 0
                
            print(f"Migration: {legacy} → {new}")
            print(f"  Legacy:  {legacy_count:5d} messages ({100-migration_percent:5.1f}%)")
            print(f"  New:     {new_count:5d} messages ({migration_percent:5.1f}%)")
            print(f"  Status:  {self._get_status_emoji(migration_percent)}")
            print()
            
        # Show all health topics
        print("All Health Topics:")
        print("-" * 80)
        health_topics = [t for t in self.message_counts.keys() if '/health' in t or t == 'system/trigger_telemetry']
        for topic in sorted(health_topics):
            count = self.message_counts[topic]
            last = self.last_seen.get(topic)
            if last:
                age = (datetime.now() - last).seconds
                status = "✓" if age < 120 else "⚠"
            else:
                age = "?"
                status = "✗"
            publishers = len(self.unique_publishers.get(topic, set()))
            print(f"  {status} {topic:40} {count:5d} msgs, {publishers:2d} publishers, last: {age}s ago")
            
    def _get_status_emoji(self, percent: float) -> str:
        """Get status emoji based on migration percentage."""
        if percent == 0:
            return "❌ Not Started"
        elif percent < 25:
            return "🔴 Early Stage"
        elif percent < 50:
            return "🟡 In Progress"
        elif percent < 75:
            return "🟢 Majority Migrated"
        elif percent < 100:
            return "🔵 Nearly Complete"
        else:
            return "✅ Fully Migrated"
            
    def _generate_report(self):
        """Generate final migration report."""
        print("\n" + "=" * 80)
        print("MIGRATION REPORT")
        print("=" * 80)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'duration': (datetime.now() - self.start_time).total_seconds(),
            'migrations': {}
        }
        
        for legacy, new in self.topic_mappings.items():
            legacy_count = self.message_counts.get(legacy, 0)
            new_count = self.message_counts.get(new, 0)
            total = legacy_count + new_count
            
            if total > 0:
                migration_data = {
                    'legacy_topic': legacy,
                    'new_topic': new,
                    'legacy_messages': legacy_count,
                    'new_messages': new_count,
                    'migration_percentage': (new_count / total) * 100,
                    'unique_legacy_publishers': list(self.unique_publishers.get(legacy, set())),
                    'unique_new_publishers': list(self.unique_publishers.get(new, set())),
                    'recommendation': self._get_recommendation(new_count / total)
                }
                report['migrations'][legacy] = migration_data
                
                print(f"\n{legacy} → {new}")
                print(f"  Migration: {migration_data['migration_percentage']:.1f}%")
                print(f"  Recommendation: {migration_data['recommendation']}")
                
        # Save report
        report_file = f"migration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n\nDetailed report saved to: {report_file}")
        
    def _get_recommendation(self, migration_ratio: float) -> str:
        """Get recommendation based on migration progress."""
        if migration_ratio == 0:
            return "Begin migration - update service to publish to new topic"
        elif migration_ratio < 0.5:
            return "Continue migration - ensure all instances are updated"
        elif migration_ratio < 1.0:
            return "Nearly complete - identify and update remaining legacy publishers"
        else:
            return "Ready to deprecate legacy topic - all publishers migrated"


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor MQTT topic migration progress')
    parser.add_argument('--broker', default='localhost', help='MQTT broker address')
    parser.add_argument('--port', type=int, default=1883, help='MQTT broker port')
    parser.add_argument('--duration', type=int, default=300, help='Monitoring duration in seconds')
    
    args = parser.parse_args()
    
    monitor = TopicMigrationMonitor(args.broker, args.port)
    monitor.run(args.duration)


if __name__ == '__main__':
    main()