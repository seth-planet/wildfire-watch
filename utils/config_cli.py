#!/usr/bin/env python3.12
"""Command-line interface for configuration management.

This tool provides utilities for validating, exporting, and checking
configuration across all Wildfire Watch services.

Usage:
    python -m utils.config_cli validate
    python -m utils.config_cli export --format yaml
    python -m utils.config_cli check-compatibility
    python -m utils.config_cli show --service camera_detector
"""

import sys
import argparse
import logging
from typing import Optional
from .service_configs import load_all_configs
from .config_base import export_all_configs, ConfigValidationError

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def validate_command(args):
    """Validate all service configurations."""
    try:
        configs = load_all_configs()
        print("✅ All configurations valid")
        print(f"\nLoaded configurations for {len(configs)} services:")
        for service in configs:
            print(f"  - {service}")
        return 0
    except ConfigValidationError as e:
        print(f"❌ Configuration validation failed: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 2


def export_command(args):
    """Export all configurations."""
    try:
        configs = load_all_configs()
        output = export_all_configs(configs, args.format)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"✅ Configuration exported to {args.output}")
        else:
            print(output)
        return 0
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return 1


def show_command(args):
    """Show configuration for a specific service."""
    try:
        configs = load_all_configs()
        
        if args.service not in configs:
            print(f"❌ Unknown service '{args.service}'")
            print(f"Available services: {', '.join(configs.keys())}")
            return 1
            
        config = configs[args.service]
        print(f"Configuration for {args.service}:")
        print(config.export(args.format))
        return 0
    except Exception as e:
        print(f"❌ Error showing configuration: {e}")
        return 1


def check_compatibility_command(args):
    """Check cross-service configuration compatibility."""
    try:
        configs = load_all_configs()
        
        # Check critical safety parameters
        gpio_config = configs.get('gpio_trigger')
        if gpio_config:
            runtime_minutes = gpio_config.max_engine_runtime / 60
            capacity = gpio_config.tank_capacity_gallons
            flow_rate = gpio_config.pump_flow_rate_gpm
            
            print("🔍 Safety Check Results:")
            print(f"  Tank capacity: {capacity} gallons")
            print(f"  Pump flow rate: {flow_rate} GPM")
            print(f"  Max runtime: {runtime_minutes:.1f} minutes")
            print(f"  Safe runtime: {(capacity/flow_rate):.1f} minutes at full flow")
            
        # Check consensus settings
        consensus_config = configs.get('fire_consensus')
        if consensus_config:
            print("\n🔍 Consensus Configuration:")
            print(f"  Threshold: {consensus_config.consensus_threshold} cameras")
            print(f"  Single camera override: {consensus_config.single_camera_trigger}")
            print(f"  Cooldown period: {consensus_config.cooldown_period}s")
            
        # Check MQTT consistency
        mqtt_settings = set()
        for service, config in configs.items():
            if hasattr(config, 'mqtt_broker'):
                mqtt_key = f"{config.mqtt_broker}:{config.mqtt_port}"
                mqtt_settings.add(mqtt_key)
                
        if len(mqtt_settings) > 1:
            print("\n⚠️  Warning: Services using different MQTT settings!")
            for setting in mqtt_settings:
                print(f"    - {setting}")
        else:
            print("\n✅ All services using consistent MQTT settings")
            
        return 0
    except Exception as e:
        print(f"❌ Compatibility check failed: {e}")
        return 1


def diff_command(args):
    """Show configuration differences between services."""
    try:
        configs = load_all_configs()
        
        if args.service1 not in configs or args.service2 not in configs:
            print(f"❌ Unknown service(s)")
            print(f"Available services: {', '.join(configs.keys())}")
            return 1
            
        config1 = configs[args.service1]
        config2 = configs[args.service2]
        
        # Find common configuration keys
        common_keys = set(config1.SCHEMA.keys()) & set(config2.SCHEMA.keys())
        
        print(f"Configuration differences between {args.service1} and {args.service2}:")
        print(f"Common configuration keys: {len(common_keys)}")
        
        differences = config1.get_diff(config2)
        if differences:
            print("\nDifferences in common keys:")
            for key, (val1, val2) in differences.items():
                print(f"  {key}:")
                print(f"    {args.service1}: {val1}")
                print(f"    {args.service2}: {val2}")
        else:
            print("\nNo differences in common configuration keys")
            
        return 0
    except Exception as e:
        print(f"❌ Diff failed: {e}")
        return 1


def main():
    """Main entry point for configuration CLI."""
    parser = argparse.ArgumentParser(
        description='Wildfire Watch Configuration Management'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Validate command
    validate_parser = subparsers.add_parser(
        'validate',
        help='Validate all service configurations'
    )
    validate_parser.set_defaults(func=validate_command)
    
    # Export command
    export_parser = subparsers.add_parser(
        'export',
        help='Export all configurations'
    )
    export_parser.add_argument(
        '--format',
        choices=['json', 'yaml'],
        default='yaml',
        help='Export format'
    )
    export_parser.add_argument(
        '--output',
        help='Output file (default: stdout)'
    )
    export_parser.set_defaults(func=export_command)
    
    # Show command
    show_parser = subparsers.add_parser(
        'show',
        help='Show configuration for a specific service'
    )
    show_parser.add_argument(
        '--service',
        required=True,
        help='Service name'
    )
    show_parser.add_argument(
        '--format',
        choices=['json', 'yaml'],
        default='yaml',
        help='Output format'
    )
    show_parser.set_defaults(func=show_command)
    
    # Check compatibility command
    compat_parser = subparsers.add_parser(
        'check-compatibility',
        help='Check cross-service configuration compatibility'
    )
    compat_parser.set_defaults(func=check_compatibility_command)
    
    # Diff command
    diff_parser = subparsers.add_parser(
        'diff',
        help='Show configuration differences between services'
    )
    diff_parser.add_argument(
        'service1',
        help='First service to compare'
    )
    diff_parser.add_argument(
        'service2',
        help='Second service to compare'
    )
    diff_parser.set_defaults(func=diff_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
        
    # Execute command
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())