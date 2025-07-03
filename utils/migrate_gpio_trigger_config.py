#!/usr/bin/env python3.12
"""Migration script to update GPIO Trigger service to use new configuration system.

This script demonstrates how to migrate the gpio_trigger service from its
current dictionary-based configuration to the new ConfigBase system.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.service_configs import GPIOTriggerConfig


def generate_migrated_config():
    """Generate the migrated configuration code for gpio_trigger."""
    
    migration_code = '''#!/usr/bin/env python3.12
"""GPIO-based fire suppression pump controller with comprehensive safety systems.

[Original docstring content remains the same...]
"""
import os
import time
import json
import socket
import threading
import logging
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Optional, Dict, Any, Callable

import paho.mqtt.client as mqtt
from dotenv import load_dotenv

# Import new configuration system
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.service_configs import GPIOTriggerConfig

load_dotenv()

# ─────────────────────────────────────────────────────────────
# Configuration - MIGRATED TO NEW SYSTEM
# ─────────────────────────────────────────────────────────────
# Load configuration using new system
try:
    CONFIG = GPIOTriggerConfig()
    logger = logging.getLogger(__name__)
    logger.info("Configuration loaded successfully using new system")
    
    # Log critical safety parameters
    logger.info(f"MAX_ENGINE_RUNTIME: {CONFIG.max_engine_runtime}s")
    logger.info(f"Tank capacity: {CONFIG.tank_capacity_gallons} gallons")
    logger.info(f"Flow rate: {CONFIG.pump_flow_rate_gpm} GPM")
    
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    logger.error("Using fallback configuration - SAFETY CRITICAL!")
    
    # Fallback to safe defaults if configuration fails
    class FallbackConfig:
        # MQTT Settings
        mqtt_broker = 'mqtt_broker'
        mqtt_port = 1883
        mqtt_tls = False
        tls_ca_path = '/mnt/data/certs/ca.crt'
        
        # Topics
        trigger_topic = 'fire/trigger'
        emergency_topic = 'fire/emergency'
        telemetry_topic = 'system/trigger_telemetry'
        
        # GPIO Pins
        main_valve_pin = 18
        ign_start_pin = 23
        ign_on_pin = 24
        ign_off_pin = 25
        refill_valve_pin = 22
        
        # Critical Safety - Conservative defaults
        max_engine_runtime = 600.0  # 10 minutes - very conservative
        refill_multiplier = 40.0
        fire_off_delay = 1800.0
        max_dry_run_time = 180.0  # 3 minutes
        
        # Tank configuration - small tank assumed for safety
        tank_capacity_gallons = 200.0
        pump_flow_rate_gpm = 30.0
        
    CONFIG = FallbackConfig()
    logger.warning("Using conservative fallback configuration")

# Create compatibility mapping for old CONFIG dictionary access
# This allows gradual migration of code that uses CONFIG['key'] syntax
CONFIG_DICT = {
    'MQTT_BROKER': CONFIG.mqtt_broker,
    'MQTT_PORT': CONFIG.mqtt_port,
    'MQTT_TLS': CONFIG.mqtt_tls,
    'TLS_CA_PATH': CONFIG.tls_ca_path,
    'TRIGGER_TOPIC': CONFIG.trigger_topic,
    'EMERGENCY_TOPIC': CONFIG.emergency_topic,
    'TELEMETRY_TOPIC': CONFIG.telemetry_topic,
    'MAIN_VALVE_PIN': CONFIG.main_valve_pin,
    'IGN_START_PIN': CONFIG.ign_start_pin,
    'IGN_ON_PIN': CONFIG.ign_on_pin,
    'IGN_OFF_PIN': CONFIG.ign_off_pin,
    'REFILL_VALVE_PIN': CONFIG.refill_valve_pin,
    'MAX_ENGINE_RUNTIME': CONFIG.max_engine_runtime,
    'REFILL_MULTIPLIER': CONFIG.refill_multiplier,
    'FIRE_OFF_DELAY': CONFIG.fire_off_delay,
    'MAX_DRY_RUN_TIME': CONFIG.max_dry_run_time,
    # Add other mappings as needed...
}

# Rest of the trigger.py code remains the same, but update references:
# Change: CONFIG['MQTT_BROKER'] -> CONFIG.mqtt_broker
# Or use CONFIG_DICT for compatibility during migration
'''
    
    return migration_code


def show_migration_steps():
    """Show step-by-step migration instructions."""
    
    print("GPIO Trigger Configuration Migration Steps:")
    print("=" * 50)
    print()
    print("1. The new configuration system provides:")
    print("   - Type validation and range checking")
    print("   - Safety validation (tank capacity vs runtime)")
    print("   - Cross-service compatibility checks")
    print("   - Configuration export and documentation")
    print()
    print("2. Key changes:")
    print("   - CONFIG dictionary -> CONFIG object with attributes")
    print("   - Automatic validation of safety parameters")
    print("   - GPIO pin conflict detection")
    print("   - Tank capacity and flow rate configuration")
    print()
    print("3. Migration approach:")
    print("   - Import GPIOTriggerConfig from utils.service_configs")
    print("   - Create CONFIG instance with validation")
    print("   - Provide fallback for safety if config fails")
    print("   - Update references throughout code")
    print()
    print("4. New safety features:")
    print("   - Validates max_engine_runtime against tank capacity")
    print("   - Ensures GPIO pins don't conflict")
    print("   - Validates timing relationships")
    print()
    
    # Test loading the configuration
    print("Testing configuration load...")
    try:
        config = GPIOTriggerConfig()
        print("✅ Configuration loaded successfully")
        print(f"   Max runtime: {config.max_engine_runtime}s")
        print(f"   Tank capacity: {config.tank_capacity_gallons} gallons")
        print(f"   Safe runtime: {(config.tank_capacity_gallons/config.pump_flow_rate_gpm)*60:.0f}s")
    except Exception as e:
        print(f"❌ Configuration error: {e}")


if __name__ == '__main__':
    show_migration_steps()
    
    # Optionally generate migration code
    if len(sys.argv) > 1 and sys.argv[1] == 'generate':
        code = generate_migrated_config()
        output_file = 'trigger_migrated.py'
        with open(output_file, 'w') as f:
            f.write(code)
        print(f"\nMigration code written to {output_file}")