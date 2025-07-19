#!/usr/bin/env python3.12
"""GPIO-based fire suppression pump controller with comprehensive safety systems.

This module implements the physical control layer for the Wildfire Watch system,
managing a gasoline/diesel pump engine and water valves through GPIO pins. It
receives fire detection triggers via MQTT and activates the suppression system
with multiple layers of safety protection.

Critical Safety Features:
    1. State Machine Control: Enforces valid state transitions only
    2. Refill Lockout: Prevents starting with low water levels
    3. Maximum Runtime Limit: Prevents pump damage from empty reservoir
    4. Dry Run Protection: Monitors water flow and shuts down if absent
    5. Valve-First Startup: Opens valves before engine to prevent deadheading
    6. Emergency Override: Manual control via MQTT for unusual situations

The Refill Cycle (Most Important Safety Feature):
    - Refill valve opens IMMEDIATELY when pump starts (not after)
    - System enters REFILLING state after pump stops
    - Cannot start again until refill completes (time or float switch)
    - This prevents repeated starts with progressively lower water levels

State Machine:
    IDLE -> PRIMING -> STARTING -> RUNNING -> STOPPING -> REFILLING -> IDLE
    Any state can transition to ERROR on critical failure

Hardware Configuration:
    All GPIO pins and logic levels are configurable via environment variables.
    The system auto-detects Raspberry Pi hardware and enters simulation mode
    if not available, allowing development and testing on any platform.

Communication Flow:
    1. Subscribes to 'fire/trigger' for consensus-based activation
    2. Subscribes to 'fire/emergency' for manual override commands
    3. Publishes detailed telemetry to 'system/trigger_telemetry'
    4. Monitors optional hardware sensors (float switch, pressure switch)

MQTT Topics:
    Subscribed:
        - fire/trigger: Fire suppression activation command
        - fire/emergency: Manual override (start/stop/reset)
        
    Published:
        - system/trigger_telemetry: Detailed status and events
        - system/trigger_telemetry/{NODE_ID}/lwt: Last will testament

Thread Model:
    - Main thread: MQTT message handling and state management
    - Timer threads: Scheduled state transitions (priming, shutdown, refill)
    - Monitor thread: Continuous dry-run protection
    - All state access synchronized via single RLock

Critical Parameters:
    - MAX_ENGINE_RUNTIME: MUST be set based on tank capacity/flow rate!
    - REFILL_MULTIPLIER: Determines refill duration (runtime * multiplier)
    - DRY_RUN_TIME: Maximum time without water flow before emergency stop
    - PRIMING_TIME: Time for pump to build pressure before starting

GPIO Pin Functions:
    - ENGINE_START_PIN: Momentary signal to start engine
    - ENGINE_STOP_PIN: Sustained signal to stop engine
    - MAIN_VALVE_PIN: Controls main water output valve
    - PRIMING_VALVE_PIN: Small valve for pump priming
    - REFILL_VALVE_PIN: Controls reservoir refill valve
    - RESERVOIR_FLOAT_PIN: Optional water level sensor
    - LINE_PRESSURE_PIN: Optional output pressure sensor

Example:
    Run standalone:
        $ python3.12 trigger.py
        
    Run in Docker:
        $ docker-compose up gpio-trigger
        
    Test in simulation mode:
        $ GPIO_SIMULATION=true python3.12 trigger.py

Warning:
    This controls physical hardware that could cause property damage or injury
    if misconfigured. Always test thoroughly in simulation mode first and 
    ensure MAX_ENGINE_RUNTIME is set conservatively for your water capacity.
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

# Import configuration base
try:
    from utils.config_base import ConfigBase, ConfigSchema, ConfigValidationError
except ImportError:
    # For standalone testing
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.config_base import ConfigBase, ConfigSchema, ConfigValidationError

# Import safety wrappers
try:
    from gpio_safety import SafeGPIO, ThreadSafeStateMachine, SafeTimerManager, HardwareError, GPIOVerificationError
except ImportError:
    # Fallback if gpio_safety not available (for backwards compatibility)
    SafeGPIO = None
    ThreadSafeStateMachine = object
    SafeTimerManager = None
    HardwareError = Exception
    GPIOVerificationError = Exception

load_dotenv()

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

class PumpControllerConfig(ConfigBase):
    """Configuration for GPIO trigger pump controller.
    
    This configuration class replaces the global CONFIG dictionary to enable
    runtime configuration loading and proper test isolation.
    """
    
    SCHEMA = {
        # MQTT Configuration
        'mqtt_broker': ConfigSchema(str, default='mqtt_broker', description="MQTT broker hostname"),
        'mqtt_port': ConfigSchema(int, default=1883, min=1, max=65535, description="MQTT broker port"),
        'mqtt_tls': ConfigSchema(bool, default=False, description="Enable TLS for MQTT"),
        'tls_ca_path': ConfigSchema(str, default='/mnt/data/certs/ca.crt', description="CA certificate path"),
        
        # Topic Configuration
        'topic_prefix': ConfigSchema(str, default='', description="Topic prefix for test isolation"),
        'trigger_topic': ConfigSchema(str, default='fire/trigger', description="Fire trigger topic"),
        'emergency_topic': ConfigSchema(str, default='fire/emergency', description="Emergency command topic"),
        'telemetry_topic': ConfigSchema(str, default='system/trigger_telemetry', description="Telemetry topic"),
        
        # GPIO Pins - Control
        'main_valve_pin': ConfigSchema(int, default=18, description="Main water valve control pin"),
        'ign_start_pin': ConfigSchema(int, default=23, description="Ignition start signal pin"),
        'ign_on_pin': ConfigSchema(int, default=24, description="Ignition on signal pin"),
        'ign_off_pin': ConfigSchema(int, default=25, description="Ignition off signal pin"),
        'refill_valve_pin': ConfigSchema(int, default=22, description="Refill valve control pin"),
        'priming_valve_pin': ConfigSchema(int, default=26, description="Priming valve control pin"),
        'rpm_reduce_pin': ConfigSchema(int, default=27, description="RPM reduction control pin"),
        
        # GPIO Pins - Monitoring (Optional - 0 means disabled)
        'reservoir_float_pin': ConfigSchema(int, default=0, description="Reservoir float switch pin (0=disabled)"),
        'line_pressure_pin': ConfigSchema(int, default=0, description="Line pressure sensor pin (0=disabled)"),
        'flow_sensor_pin': ConfigSchema(int, default=0, description="Flow sensor pin (0=disabled)"),
        'emergency_button_pin': ConfigSchema(int, default=0, description="Emergency button pin (0=disabled)"),
        
        # Timing Configuration
        'pre_open_delay': ConfigSchema(float, default=2.0, min=0.0, description="Valve pre-open delay"),
        'ignition_start_duration': ConfigSchema(float, default=5.0, min=0.1, description="Ignition start signal duration"),
        'fire_off_delay': ConfigSchema(float, default=1800.0, min=1.0, description="Auto shutoff delay"),
        'valve_close_delay': ConfigSchema(float, default=600.0, min=0.0, description="Valve close delay after stop"),
        'ignition_off_duration': ConfigSchema(float, default=5.0, min=0.1, description="Ignition off signal duration"),
        'max_engine_runtime': ConfigSchema(float, default=1800.0, min=60.0, description="Maximum engine runtime (safety limit)"),
        'refill_multiplier': ConfigSchema(float, default=40.0, min=1.0, description="Refill duration multiplier"),
        'priming_duration': ConfigSchema(float, default=180.0, min=0.1, description="Pump priming duration"),
        'rpm_reduction_lead': ConfigSchema(float, default=15.0, min=0.0, description="RPM reduction lead time"),
        'pressure_check_delay': ConfigSchema(float, default=60.0, min=0.0, description="Pressure check delay after priming"),
        'health_interval': ConfigSchema(float, default=60.0, min=1.0, description="Health reporting interval"),
        'action_retry_interval': ConfigSchema(float, default=60.0, min=1.0, description="Action retry interval"),
        'cooldown_duration': ConfigSchema(float, default=60.0, min=0.0, description="Cooldown duration after shutdown"),
        'rpm_reduction_duration': ConfigSchema(float, default=5.0, min=0.1, description="RPM reduction phase duration"),
        
        # Safety Configuration
        'reservoir_float_active_low': ConfigSchema(bool, default=True, description="Float switch active low logic"),
        'line_pressure_active_low': ConfigSchema(bool, default=True, description="Pressure switch active low logic"),
        'emergency_button_active_low': ConfigSchema(bool, default=True, description="Emergency button active low logic"),
        
        # Hardware Validation
        'hardware_validation_enabled': ConfigSchema(bool, default=False, description="Enable hardware validation"),
        'relay_feedback_pins': ConfigSchema(list, default=[], description="Relay feedback pins (JSON array)"),
        'hardware_check_interval': ConfigSchema(float, default=30.0, min=1.0, description="Hardware check interval"),
        
        # Dry Run Protection
        'max_dry_run_time': ConfigSchema(float, default=300.0, min=30.0, description="Maximum dry run time"),
        
        # Status Reporting
        'enhanced_status_enabled': ConfigSchema(bool, default=True, description="Enable enhanced status"),
        'simulation_mode_warnings': ConfigSchema(bool, default=True, description="Show simulation warnings"),
    }
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        # Map old environment variable names to new schema keys
        self._apply_env_mappings()
        
        # Call parent to load configuration
        super().__init__()
        
        # Apply topic prefix if configured
        self._apply_topic_prefix()
        
        # Convert pin 0 to None for optional pins
        self._normalize_optional_pins()
    
    def _load_value(self, env_key: str, schema: ConfigSchema) -> Any:
        """Load and convert a single configuration value.
        
        Overrides parent to handle empty strings for pin values.
        """
        raw_value = os.getenv(env_key)
        
        # Special handling for pin values - convert empty string to 0 (disabled)
        if (schema.type == int and 
            env_key.endswith('_PIN') and 
            raw_value == ''):
            return 0
        
        # Use parent implementation for everything else
        return super()._load_value(env_key, schema)
    
    def _apply_env_mappings(self):
        """Map legacy environment variable names to new schema."""
        mappings = {
            'MQTT_TOPIC_PREFIX': 'TOPIC_PREFIX',
            'IGNITION_START_PIN': 'IGN_START_PIN',
            'IGNITION_ON_PIN': 'IGN_ON_PIN',
            'IGNITION_OFF_PIN': 'IGN_OFF_PIN',
            'VALVE_PRE_OPEN_DELAY': 'PRE_OPEN_DELAY',
            'FIRE_OFF_DELAY': 'FIRE_OFF_DELAY',
            'VALVE_CLOSE_DELAY': 'VALVE_CLOSE_DELAY',
            'TELEMETRY_INTERVAL': 'HEALTH_INTERVAL',
        }
        
        for old_name, new_name in mappings.items():
            if old_name in os.environ and new_name not in os.environ:
                os.environ[new_name] = os.environ[old_name]
    
    def _apply_topic_prefix(self):
        """Apply topic prefix to MQTT topics if configured."""
        if self.topic_prefix:
            self.trigger_topic = f"{self.topic_prefix}/{self.trigger_topic}"
            self.emergency_topic = f"{self.topic_prefix}/{self.emergency_topic}"
            self.telemetry_topic = f"{self.topic_prefix}/{self.telemetry_topic}"
    
    def _normalize_optional_pins(self):
        """Convert pin 0 to None for optional sensor pins."""
        optional_pins = [
            'reservoir_float_pin', 'line_pressure_pin', 
            'flow_sensor_pin', 'emergency_button_pin'
        ]
        
        for pin_name in optional_pins:
            if getattr(self, pin_name) == 0:
                setattr(self, pin_name, None)



# ─────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────
try:
    from utils.logging_config import setup_logging
    from utils.safe_logging import register_logger_for_cleanup
    logger = setup_logging("gpio_trigger")
    register_logger_for_cleanup(logger)
except ImportError:
    # Fallback for standalone testing
    log_level = getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# GPIO Setup
# ─────────────────────────────────────────────────────────────
try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO_AVAILABLE = True
except (ImportError, RuntimeError):
    GPIO_AVAILABLE = False
    if os.getenv('SIMULATION_MODE_WARNINGS', 'true').lower() == 'true':
        logger.critical("⚠️  HARDWARE SIMULATION MODE ACTIVE ⚠️")
        logger.critical("⚠️  GPIO CONTROL IS SIMULATED - NO PHYSICAL HARDWARE CONTROL ⚠️")
        logger.critical("⚠️  PUMP AND VALVES WILL NOT OPERATE IN WILDFIRE EMERGENCY ⚠️")
    else:
        logger.warning("RPi.GPIO unavailable; using simulation mode")
    
    class GPIO:
        BCM = "BCM"
        OUT = "OUT"
        IN = "IN"
        PUD_UP = "PUD_UP"
        PUD_DOWN = "PUD_DOWN"
        HIGH = True
        LOW = False
        _state = {}
        _lock = threading.RLock()  # Add GPIO state lock
        
        @classmethod
        def setmode(cls, mode):
            pass
        
        @classmethod
        def setwarnings(cls, warnings):
            pass
        
        @classmethod
        def setup(cls, pin, mode, initial=None, pull_up_down=None):
            with cls._lock:
                if mode == cls.OUT:
                    cls._state[pin] = initial if initial is not None else cls.LOW
                else:
                    # Input pins default based on pull resistor
                    if pull_up_down == cls.PUD_UP:
                        cls._state[pin] = cls.HIGH
                    else:
                        cls._state[pin] = cls.LOW
        
        @classmethod
        def output(cls, pin, value):
            with cls._lock:
                # Convert GPIO constants to boolean consistently
                if value == cls.HIGH or value is True:
                    cls._state[pin] = True
                elif value == cls.LOW or value is False:
                    cls._state[pin] = False
                else:
                    cls._state[pin] = bool(value)
        
        @classmethod
        def input(cls, pin):
            with cls._lock:
                return cls._state.get(pin, cls.LOW)
        
        @classmethod
        def cleanup(cls):
            with cls._lock:
                cls._state.clear()

# ─────────────────────────────────────────────────────────────
# State Machine
# ─────────────────────────────────────────────────────────────
class PumpState(Enum):
    """State machine states for pump controller.
    
    The pump controller enforces strict state transitions to ensure safe operation.
    Invalid transitions are rejected, preventing dangerous conditions like starting
    an already-running pump or stopping during refill.
    
    Valid State Transitions:
        IDLE -> PRIMING: Fire detected, begin startup sequence
        PRIMING -> STARTING: Priming complete, start engine
        STARTING -> RUNNING: Engine started successfully
        RUNNING -> STOPPING: Fire extinguished or max runtime reached
        STOPPING -> REFILLING: Engine stopped, begin refill cycle
        REFILLING -> IDLE: Refill complete (time or float switch)
        
        Any state -> ERROR: Critical failure detected
        ERROR -> IDLE: Manual reset command received
        
    States:
        IDLE: System ready, monitoring for fire detection
        PRIMING: Priming valve open, building line pressure
        STARTING: Sending start signal to engine
        RUNNING: Engine running, main valve open, water flowing
        REDUCING_RPM: Unused in current implementation
        STOPPING: Shutting down engine, closing valves
        COOLDOWN: Brief pause after shutdown
        REFILLING: Refill valve open, waiting for reservoir to fill
        ERROR: Critical failure, manual intervention required
        LOW_PRESSURE: Pressure loss detected, transitioning to safe state
    """
    IDLE = auto()          # System ready, no fire detected
    PRIMING = auto()       # Valve open, priming before engine start
    STARTING = auto()      # Starting engine sequence
    RUNNING = auto()       # Engine running, pumping water
    REDUCING_RPM = auto()  # Reducing RPM before shutdown
    STOPPING = auto()      # Shutting down engine
    COOLDOWN = auto()      # Post-shutdown cooldown
    REFILLING = auto()     # Refilling reservoir
    ERROR = auto()         # Error state requiring manual intervention
    LOW_PRESSURE = auto()  # Low line pressure detected

# ─────────────────────────────────────────────────────────────
# PumpController Class
# ─────────────────────────────────────────────────────────────
class PumpController(ThreadSafeStateMachine if ThreadSafeStateMachine is not object else object):
    """Thread-safe pump controller with comprehensive safety systems.
    
    This class implements the complete control logic for a fire suppression pump
    system, including gasoline/diesel engine control, valve management, and
    safety monitoring. It uses a strict state machine to ensure safe operation
    and prevent hardware damage.
    
    Safety Systems:
        1. Refill Lockout: Cannot start if refill cycle is incomplete
        2. Maximum Runtime: Automatic shutdown to prevent empty reservoir
        3. Dry Run Protection: Monitors water flow and shuts down if absent
        4. State Machine: Only allows valid, safe state transitions
        5. Emergency Override: Manual control for unusual situations
        6. Sensor Monitoring: Optional float and pressure switches
        
    The Refill Cycle (Critical Safety Feature):
        The refill valve opens IMMEDIATELY when the pump starts, not after it
        stops. This ensures the reservoir begins refilling while water is being
        used. After pump shutdown, the system enters REFILLING state and will
        not allow restart until either:
        - The calculated refill time expires (runtime * REFILL_MULTIPLIER)
        - The reservoir float switch indicates full (if installed)
        
    Attributes:
        cfg (CONFIG): Configuration object
        _lock (RLock): Thread synchronization for all state access
        _state (PumpState): Current state machine state
        _timers (Dict[str, Timer]): Active timer threads
        _engine_start_time (float): When engine started (for runtime limit)
        _refill_complete (bool): Whether refill cycle is complete
        _total_runtime (float): Cumulative runtime for maintenance tracking
        
    Hardware Control:
        All GPIO operations go through _set_pin() which includes retry logic
        and handles both real hardware and simulation mode transparently.
        
    MQTT Integration:
        - Receives commands on fire/trigger and fire/emergency topics
        - Publishes detailed telemetry including state, sensors, and warnings
        - Sets last will testament for disconnection detection
        
    Thread Safety:
        All public methods acquire self._lock before accessing state.
        Timer callbacks and monitor threads also use proper locking.
        The single RLock pattern prevents deadlocks and ensures consistency.
        
    Error Handling:
        - Transient errors: Retry with backoff
        - Critical failures: Transition to ERROR state
        - Emergency recovery: Aggressive retry procedures
        - Safe failure: ERROR state requires manual reset
    """
    
    def __init__(self, config: Optional[PumpControllerConfig] = None, auto_connect=True):
        # Initialize parent class if using ThreadSafeStateMachine
        if ThreadSafeStateMachine is not object:
            super().__init__()
        
        # Use provided config or create default
        self.config = config or PumpControllerConfig()
        self._lock = threading.RLock()
        self._auto_connect = auto_connect
        
        # Validate timing configuration before proceeding
        self._validate_timing_config()
        
        # Initialize safety wrappers if available
        if SafeGPIO:
            self.gpio = SafeGPIO(GPIO, simulation_mode=not GPIO_AVAILABLE)
            logger.info("Using SafeGPIO wrapper for enhanced hardware safety")
        else:
            self.gpio = None
            logger.warning("SafeGPIO not available - using direct GPIO access")
            
        if SafeTimerManager:
            self.timer_manager = SafeTimerManager()
            logger.info("Using SafeTimerManager for thread-safe timer operations")
        else:
            self.timer_manager = None
            logger.warning("SafeTimerManager not available - using direct timers")
        
        self._state = PumpState.IDLE
        
        # Only create _timers dict if not using SafeTimerManager
        if not self.timer_manager:
            self._internal_timers: Dict[str, threading.Timer] = {}
        self._last_trigger_time = 0
        self._engine_start_time: Optional[float] = None
        self._total_runtime = 0
        self._shutting_down = False
        self._refill_complete = True  # Start assuming tank is full
        self._low_pressure_detected = False
        self._current_runtime = 0  # Track current session runtime
        
        # Hardware validation state
        self._hardware_status = {}
        self._last_hardware_check = 0
        self._hardware_failures = 0
        
        # Dry run protection state  
        self._pump_start_time = None
        self._water_flow_detected = False
        self._dry_run_warnings = 0
        self._dry_run_start_time = None  # Track when dry run condition started
        
        # Shutdown flag for clean thread termination
        self._shutdown = False
        self._shutting_down = False  # Flag for quick thread termination during tests
        
        # Track background threads for cleanup
        self._background_threads = []
        
        # Error tracking
        self._last_error = ""  # Store last error reason
        
        # Initialize GPIO
        self._init_gpio()
        
        # Setup MQTT only if auto_connect is True
        if self._auto_connect:
            self._setup_mqtt()
            
            # Start monitoring tasks
            self._start_monitoring_tasks()
            
            # Start health monitoring
            self._schedule_timer('health', self._publish_health, self.config.health_interval)
        else:
            # Initialize client but don't connect
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, clean_session=True)
            self.client.on_connect = self._on_connect
            self.client.on_message = self._on_message
            self.client.on_disconnect = self._on_disconnect
        
        logger.info(f"PumpController initialized in {self._state.name} state")
    
    def connect(self):
        """Connect to MQTT broker and start monitoring tasks.
        
        This method is used when auto_connect=False to manually establish
        the MQTT connection after the test environment is properly set up.
        """
        if not self._auto_connect and not hasattr(self, '_mqtt_connected'):
            logger.info("Manually connecting to MQTT broker")
            self._setup_mqtt()
            
            # Start monitoring tasks
            self._start_monitoring_tasks()
            
            # Start health monitoring
            self._schedule_timer('health', self._publish_health, self.config.health_interval)
            
            self._mqtt_connected = True
    
    def _validate_timing_config(self):
        """Validate timing configuration for self-consistency.
        
        Ensures that all timing parameters make sense together and will
        result in proper system operation. Emits warnings or raises errors
        for invalid configurations.
        """
        cfg = self.config
        
        # Critical validations that would break the system
        errors = []
        
        # MAX_ENGINE_RUNTIME must be reasonable
        if cfg.max_engine_runtime < 30:
            errors.append(f"MAX_ENGINE_RUNTIME ({cfg.max_engine_runtime}s) is dangerously short. Minimum recommended: 30s")
        
        # RPM_REDUCTION_LEAD must be less than MAX_ENGINE_RUNTIME
        if cfg.rpm_reduction_lead >= cfg.max_engine_runtime:
            errors.append(f"RPM_REDUCTION_LEAD ({cfg.rpm_reduction_lead}s) must be less than MAX_ENGINE_RUNTIME ({cfg.max_engine_runtime}s)")
        
        # RPM reduction should happen with enough time before shutdown
        rpm_reduction_time = cfg.max_engine_runtime - cfg.rpm_reduction_lead
        if rpm_reduction_time < 10:
            errors.append(f"RPM reduction would happen only {rpm_reduction_time}s after engine start - too early! Increase MAX_ENGINE_RUNTIME or decrease RPM_REDUCTION_LEAD")
        
        # Startup sequence timing
        startup_time = cfg.priming_duration + cfg.ignition_start_duration
        if startup_time > cfg.max_engine_runtime * 0.5:
            errors.append(f"Startup sequence ({startup_time}s) takes more than half of MAX_ENGINE_RUNTIME ({cfg.max_engine_runtime}s)")
        
        # Warnings for suboptimal but not critical issues
        warnings = []
        
        # Check if there's enough runtime after startup
        effective_runtime = cfg.max_engine_runtime - startup_time
        if effective_runtime < 60:
            warnings.append(f"Effective pump runtime is only {effective_runtime}s after startup sequence")
        
        # Priming duration check
        if cfg.priming_duration < 30:
            warnings.append(f"PRIMING_DURATION ({cfg.priming_duration}s) may be too short for proper priming")
        elif cfg.priming_duration > 300:
            warnings.append(f"PRIMING_DURATION ({cfg.priming_duration}s) seems excessively long")
        
        # Engine start duration check
        if cfg.ignition_start_duration < 3:
            warnings.append(f"IGNITION_START_DURATION ({cfg.ignition_start_duration}s) may be too short for reliable engine start")
        elif cfg.ignition_start_duration > 30:
            warnings.append(f"IGNITION_START_DURATION ({cfg.ignition_start_duration}s) seems excessively long")
        
        # Refill multiplier check
        if cfg.refill_multiplier < 1.5:
            warnings.append(f"REFILL_MULTIPLIER ({cfg.refill_multiplier}) may not provide enough refill time")
        
        # Valve timing checks
        if cfg.valve_close_delay < cfg.ignition_off_duration:
            warnings.append(f"VALVE_CLOSE_DELAY ({cfg.valve_close_delay}s) should be longer than IGNITION_OFF_DURATION ({cfg.ignition_off_duration}s)")
        
        # Health interval check
        if cfg.health_interval > cfg.max_engine_runtime / 3:
            warnings.append(f"HEALTH_INTERVAL ({cfg.health_interval}s) is too long - pump could run for {cfg.max_engine_runtime}s with only {int(cfg.max_engine_runtime / cfg.health_interval)} health reports")
        
        # Fire off delay vs max runtime
        if cfg.fire_off_delay > cfg.max_engine_runtime:
            warnings.append(f"FIRE_OFF_DELAY ({cfg.fire_off_delay}s) is longer than MAX_ENGINE_RUNTIME ({cfg.max_engine_runtime}s) - pump will always hit safety timeout")
        
        # Emit all warnings
        for warning in warnings:
            logger.warning(f"Timing configuration warning: {warning}")
        
        # Raise error if critical issues found
        if errors:
            error_msg = "Critical timing configuration errors:\n" + "\n".join(f"  - {e}" for e in errors)
            logger.error(error_msg)
            
            # In production, we might want to refuse to start with invalid config
            # For now, we'll allow it but with strong warnings
            if not GPIO_AVAILABLE:  # In simulation mode, be more lenient
                logger.critical("TIMING CONFIGURATION ERRORS DETECTED - SYSTEM MAY NOT OPERATE CORRECTLY")
            else:
                # In real hardware mode, we should be stricter
                raise ValueError(error_msg)
        
        # Log timing summary
        logger.info("Timing configuration summary:")
        logger.info(f"  Startup: PRIMING={cfg.priming_duration}s + IGNITION_START={cfg.ignition_start_duration}s = {startup_time}s total")
        logger.info(f"  Runtime: MAX={cfg.max_engine_runtime}s, RPM reduction at {rpm_reduction_time}s")
        logger.info(f"  Effective pump runtime: {effective_runtime}s")
        logger.info(f"  Refill time after {cfg.max_engine_runtime}s runtime: {cfg.max_engine_runtime * cfg.refill_multiplier}s")
    
    def _init_gpio(self):
        """Initialize all GPIO pins to safe state"""
        # Output pins
        output_pins = {
            'MAIN_VALVE_PIN': GPIO.LOW,
            'IGN_START_PIN': GPIO.LOW,
            'IGN_ON_PIN': GPIO.LOW,
            'IGN_OFF_PIN': GPIO.LOW,
            'REFILL_VALVE_PIN': GPIO.LOW,
            'PRIMING_VALVE_PIN': GPIO.LOW,
            'RPM_REDUCE_PIN': GPIO.LOW,
        }
        
        for pin_name, initial_state in output_pins.items():
            pin = getattr(self.config, pin_name.lower())
            GPIO.setup(pin, GPIO.OUT, initial=initial_state)
            logger.debug(f"Initialized {pin_name} (pin {pin}) to {initial_state}")
        
        # Input pins (optional monitoring)
        if self.config.reservoir_float_pin:
            pull = GPIO.PUD_DOWN if self.config.reservoir_float_active_low else GPIO.PUD_UP
            GPIO.setup(self.config.reservoir_float_pin, GPIO.IN, pull_up_down=pull)
            logger.info(f"Reservoir float switch on pin {self.config.reservoir_float_pin}")
        
        if self.config.line_pressure_pin:
            pull = GPIO.PUD_DOWN if self.config.line_pressure_active_low else GPIO.PUD_UP
            GPIO.setup(self.config.line_pressure_pin, GPIO.IN, pull_up_down=pull)
            logger.info(f"Line pressure switch on pin {self.config.line_pressure_pin}")
        
        # Optional hardware validation pins
        if self.config.relay_feedback_pins:
            for i, pin_str in enumerate(self.config.relay_feedback_pins):
                if pin_str.strip():
                    pin = int(pin_str.strip())
                    GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
                    logger.info(f"Relay feedback pin {i+1} configured on pin {pin}")
        
        # Optional flow sensor pin
        if self.config.flow_sensor_pin:
            GPIO.setup(self.config.flow_sensor_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
            logger.info(f"Water flow sensor on pin {self.config.flow_sensor_pin}")
        
        # Optional emergency button pin
        if self.config.emergency_button_pin:
            pull = GPIO.PUD_UP if self.config.emergency_button_active_low else GPIO.PUD_DOWN
            GPIO.setup(self.config.emergency_button_pin, GPIO.IN, pull_up_down=pull)
            logger.info(f"Emergency button on pin {self.config.emergency_button_pin}")
    
    def _setup_mqtt(self):
        """Setup MQTT client with TLS if configured"""
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, clean_session=True)
        
        # Set callbacks
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        
        # Configure TLS if enabled
        if self.config.mqtt_tls:
            import ssl
            self.client.tls_set(
                ca_certs=self.config.tls_ca_path,
                cert_reqs=ssl.CERT_REQUIRED,
                tls_version=ssl.PROTOCOL_TLS
            )
            logger.info("MQTT TLS enabled")
        
        # Set LWT
        lwt_topic = f"{self.config.telemetry_topic}/{socket.gethostname()}/lwt"
        lwt_payload = json.dumps({
            'host': socket.gethostname(),
            'status': 'offline',
            'timestamp': self._now_iso()
        })
        self.client.will_set(lwt_topic, payload=lwt_payload, qos=1, retain=True)
        
        # Connect
        self._mqtt_connect_with_retry()
    
    def _mqtt_connect_with_retry(self, max_retries=None):
        """Connect to MQTT with retry logic"""
        retry_count = 0
        # Default to 10 retries if not specified
        if max_retries is None:
            max_retries = 10
            
        while not self._shutdown and retry_count < max_retries:
            try:
                port = 8883 if self.config.mqtt_tls else self.config.mqtt_port
                self.client.connect(self.config.mqtt_broker, port, keepalive=60)
                self.client.loop_start()
                logger.info(f"MQTT client connected to {self.config.mqtt_broker}:{port}")
                break
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"MQTT connection failed after {max_retries} attempts: {e}")
                    raise
                logger.error(f"MQTT connection failed (attempt {retry_count}): {e}")
                # Use shorter sleep for tests if max_retries is set
                sleep_time = 0.1 if max_retries is not None else 5
                time.sleep(sleep_time)
    
    def _now_iso(self) -> str:
        """Get current UTC timestamp in ISO format"""
        return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    
    def _get_state_snapshot(self) -> Dict[str, Any]:
        """Get current state of all pins and system"""
        with self._lock:
            snapshot = {
                'state': self._state.name,
                'engine_on': GPIO.input(self.config.ign_on_pin),
                'main_valve': GPIO.input(self.config.main_valve_pin),
                'refill_valve': GPIO.input(self.config.refill_valve_pin),
                'priming_valve': GPIO.input(self.config.priming_valve_pin),
                'rpm_reduced': GPIO.input(self.config.rpm_reduce_pin),
                'total_runtime': self._total_runtime,
                'current_runtime': self._current_runtime,
                'shutting_down': self._shutting_down,
                'refill_complete': self._refill_complete,
                'active_timers': self.timer_manager.get_active_timers() if self.timer_manager else (list(self._internal_timers.keys()) if hasattr(self, '_internal_timers') else []),
            }
            
            # Add monitoring status if available
            if self.config.reservoir_float_pin:
                snapshot['reservoir_full'] = self._is_reservoir_full()
            
            if self.config.line_pressure_pin:
                snapshot['line_pressure_ok'] = self._is_line_pressure_ok()
            
            return snapshot
    
    def _publish_event(self, action: str, extra_data: Optional[Dict] = None):
        """Publish telemetry event"""
        payload = {
            'host': socket.gethostname(),
            'timestamp': self._now_iso(),
            'action': action,
            'gpio_available': GPIO_AVAILABLE,
            'system_state': self._get_state_snapshot(),
        }
        
        if extra_data:
            payload.update(extra_data)
        
        logger.info(f"Event: {action} | State: {self._state.name}")
        
        try:
            self.client.publish(
                self.config.telemetry_topic,
                json.dumps(payload),
                qos=1
            )
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
    
    def _set_pin(self, pin_name: str, state: bool, max_retries: int = 3) -> bool:
        """Set GPIO pin state with error handling and retry logic.
        
        This is the central hardware control method that all GPIO operations
        go through. It provides consistent error handling, retry logic, and
        works transparently with both real hardware and simulation mode.
        
        Args:
            pin_name: Configuration key for pin (e.g., 'MAIN_VALVE', 'ENGINE_START')
            state: Desired state (True=HIGH, False=LOW)
            max_retries: Number of retry attempts for transient failures
            
        Returns:
            bool: True if successful, False if all retries failed
            
        Error Handling:
            - Retries with progressive backoff: 0.1s, 0.5s, 2.0s
            - Critical pins verified after setting
            - Publishes failure events for monitoring
            
        Critical Pins:
            MAIN_VALVE, IGN_ON, IGN_START are verified after setting
            to ensure the hardware actually responded.
            
        Thread Safety:
            Acquires lock during GPIO operations to prevent races
        """
        # Use SafeGPIO if available
        if self.gpio:
            try:
                pin = getattr(self.config, f'{pin_name.lower()}_pin')
                result = self.gpio.safe_write(pin, state, pin_name=pin_name, retries=max_retries)
                
                if result:
                    logger.debug(f"SafeGPIO: Set {pin_name} (pin {pin}) to {'HIGH' if state else 'LOW'}")
                else:
                    self._publish_event('gpio_failure_final', {
                        'pin': pin_name, 
                        'attempts': max_retries,
                        'error': 'SafeGPIO write failed'
                    })
                
                return result
                
            except GPIOVerificationError as e:
                logger.critical(f"Critical GPIO verification failure for {pin_name}: {e}")
                self._publish_event('gpio_critical_failure', {
                    'pin': pin_name,
                    'error': str(e),
                    'type': 'verification_failure'
                })
                # Critical pins failing verification may require ERROR state
                if pin_name in ['MAIN_VALVE', 'IGN_ON']:
                    self._enter_error_state(f"Critical GPIO {pin_name} verification failed")
                return False
                
            except HardwareError as e:
                logger.error(f"Hardware error for {pin_name}: {e}")
                self._publish_event('gpio_hardware_error', {
                    'pin': pin_name,
                    'error': str(e)
                })
                # Critical pins failing with hardware error require ERROR state
                if pin_name in ['MAIN_VALVE', 'IGN_ON', 'IGN_START']:
                    self._enter_error_state(f"Critical GPIO {pin_name} hardware failure: {e}")
                return False
                
            except Exception as e:
                # Catch all other exceptions from SafeGPIO
                logger.error(f"GPIO operation failed for {pin_name}: {e}")
                self._publish_event('gpio_operation_failed', {
                    'pin': pin_name,
                    'error': str(e),
                    'type': type(e).__name__
                })
                # For non-critical pins, just return False
                if pin_name in ['MAIN_VALVE', 'IGN_ON', 'IGN_START']:
                    # Critical pins need special handling
                    logger.critical(f"Critical GPIO {pin_name} failed: {e}")
                    self._enter_error_state(f"Critical GPIO {pin_name} operation failed: {e}")
                return False
        
        # Fallback to original implementation if SafeGPIO not available
        pin = getattr(self.config, f'{pin_name.lower()}_pin')
        retry_delays = [0.1, 0.5, 2.0]  # Progressive delays
        
        for attempt in range(max_retries):
            try:
                # Use controller lock to prevent concurrent pin changes
                with self._lock:
                    GPIO.output(pin, GPIO.HIGH if state else GPIO.LOW)
                    
                    # Verify operation succeeded for critical pins
                    if pin_name in ['MAIN_VALVE', 'IGN_ON', 'IGN_START']:
                        time.sleep(0.05)  # Small delay for hardware response
                        actual_state = GPIO.input(pin)
                        expected_state = bool(state)  # Ensure both are boolean
                        if actual_state != expected_state:
                            # Debug logging
                            logger.debug(f"Pin verification issue - pin_name: {pin_name}, pin: {pin}, "
                                       f"requested: {state}, expected: {expected_state}, actual: {actual_state}")
                            logger.debug(f"GPIO state dict: {getattr(GPIO, '_state', {})}")
                            raise Exception(f"Pin state verification failed: expected {expected_state}, got {actual_state}")
                
                logger.debug(f"Set {pin_name} (pin {pin}) to {'HIGH' if state else 'LOW'}")
                return True
                
            except Exception as e:
                attempt_num = attempt + 1
                if attempt_num < max_retries:
                    delay = retry_delays[attempt] if attempt < len(retry_delays) else retry_delays[-1]
                    logger.warning(f"GPIO {pin_name} attempt {attempt_num}/{max_retries} failed: {e}. Retrying in {delay}s...")
                    self._publish_event('gpio_retry', {
                        'pin': pin_name, 
                        'attempt': attempt_num,
                        'max_retries': max_retries,
                        'delay': delay,
                        'error': str(e)
                    })
                    time.sleep(delay)
                else:
                    logger.error(f"GPIO {pin_name} failed after {max_retries} attempts: {e}")
                    self._publish_event('gpio_failure_final', {
                        'pin': pin_name, 
                        'attempts': max_retries,
                        'error': str(e)
                    })
                    return False
        
        return False
    
    def _is_reservoir_full(self) -> bool:
        """Check if reservoir is full (float switch)"""
        if not self.config.reservoir_float_pin:
            return True  # Assume full if no sensor
        
        state = GPIO.input(self.config.reservoir_float_pin)
        # Active low means LOW = full, HIGH = not full
        if self.config.reservoir_float_active_low:
            return not state
        else:
            return state
    
    def _is_line_pressure_ok(self) -> bool:
        """Check if line pressure is adequate"""
        if not self.config.line_pressure_pin:
            return True  # Assume OK if no sensor
        
        state = GPIO.input(self.config.line_pressure_pin)
        # Active low means LOW = pressure OK, HIGH = low pressure
        if self.config.line_pressure_active_low:
            return not state
        else:
            return state
    
    def _schedule_timer(self, name: str, func: Callable, delay: float):
        """Schedule a timer, canceling any existing timer with same name"""
        # Define critical timers that must transition to ERROR on failure
        critical_timers = {'start_engine', 'emergency_stop', 'ignition_off'}
        
        if self.timer_manager:
            # Use SafeTimerManager for thread-safe timer operations
            def error_handler(timer_name: str, error: Exception):
                logger.error(f"Timer {timer_name} failed: {error}")
                self._publish_event('timer_error', {'timer': timer_name, 'error': str(error)})
                # CRITICAL: If a timer function (especially one related to GPIO)
                # raises a hardware error, enter ERROR state.
                if isinstance(error, (HardwareError, GPIOVerificationError)):
                    self._enter_error_state(f"Critical timer '{timer_name}' failed: {str(error)}")
                elif timer_name in critical_timers:
                    # Any exception in critical timers should cause ERROR state
                    self._enter_error_state(f"Critical timer '{timer_name}' failed: {str(error)}")
            
            self.timer_manager.schedule(name, func, delay, error_handler)
        else:
            # Fallback to original implementation
            self._cancel_timer(name)
            
            def wrapped_func():
                with self._lock:
                    self._internal_timers.pop(name, None)
                    try:
                        func()
                    except Exception as e:
                        logger.error(f"Timer {name} failed: {e}")
                        self._publish_event('timer_error', {'timer': name, 'error': str(e)})
                        # CRITICAL: Same logic for internal timers
                        if isinstance(e, (HardwareError, GPIOVerificationError)):
                            self._enter_error_state(f"Critical timer '{name}' failed: {str(e)}")
                        elif name in critical_timers:
                            # Any exception in critical timers should cause ERROR state
                            self._enter_error_state(f"Critical timer '{name}' failed: {str(e)}")
            
            timer = threading.Timer(delay, wrapped_func)
            timer.daemon = True
            timer.start()
            self._internal_timers[name] = timer
            logger.debug(f"Scheduled timer '{name}' for {delay}s")
    
    def _cancel_timer(self, name: str):
        """Cancel a timer if it exists"""
        if self.timer_manager:
            self.timer_manager.cancel(name)
        else:
            timer = self._internal_timers.pop(name, None)
            if timer and timer.is_alive():
                timer.cancel()
                logger.debug(f"Cancelled timer '{name}'")
    
    def _cancel_all_timers(self):
        """Cancel all active timers"""
        if self.timer_manager:
            self.timer_manager.cancel_all()
        else:
            for name in list(self._internal_timers.keys()):
                self._cancel_timer(name)
    
    def _has_timer(self, name: str) -> bool:
        """Check if a timer with given name is scheduled"""
        if self.timer_manager:
            return name in self.timer_manager.get_active_timers()
        else:
            return name in self._internal_timers and self._internal_timers[name].is_alive()
    
    @property
    def _timers(self):
        """Property for backward compatibility with tests"""
        if self.timer_manager:
            # Return a dict-like object that supports 'in' operator
            class TimerDict:
                def __init__(self, timer_manager):
                    self.timer_manager = timer_manager
                
                def __contains__(self, key):
                    return key in self.timer_manager.get_active_timers()
                
                def __iter__(self):
                    return iter(self.timer_manager.get_active_timers())
                
                def keys(self):
                    return self.timer_manager.get_active_timers()
            
            return TimerDict(self.timer_manager)
        else:
            # Return actual _timers dict if it exists
            return getattr(self, '_internal_timers', {})
    
    def _start_monitoring_tasks(self):
        """Start background monitoring tasks"""
        # Monitor reservoir level during refill
        if self.config.reservoir_float_pin:
            threading.Thread(target=self._monitor_reservoir_level, daemon=True).start()
        
        # Start hardware validation monitoring if enabled
        if self.config.hardware_validation_enabled:
            self._schedule_timer('hardware_check', self._validate_hardware, self.config.hardware_check_interval)
        
        # Start dry run protection monitoring (always enabled for safety)
        dry_run_thread = threading.Thread(target=self._monitor_dry_run_protection, daemon=True)
        dry_run_thread.start()
        self._background_threads.append(dry_run_thread)
        
        # Start emergency button monitoring if configured
        if self.config.emergency_button_pin:
            emergency_thread = threading.Thread(target=self._monitor_emergency_button, daemon=True)
            emergency_thread.start()
            self._background_threads.append(emergency_thread)
    
    def _monitor_reservoir_level(self):
        """Monitor reservoir level during refill operations"""
        while not self._shutdown:
            try:
                if self._state == PumpState.REFILLING:
                    if self._is_reservoir_full():
                        logger.info("Reservoir full detected, stopping refill")
                        with self._lock:
                            self._set_pin('REFILL_VALVE', False)
                            self._refill_complete = True
                            if self._state == PumpState.REFILLING:
                                self._enter_idle()
                            self._publish_event('refill_complete_float_switch')
                            self._cancel_timer('close_refill_valve')
            except Exception as e:
                logger.error(f"Reservoir monitoring error: {e}")
            
            # Sleep with shutdown check
            for _ in range(10):
                if self._shutdown:
                    return  # Exit immediately
                time.sleep(0.1)
    
    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """MQTT connection callback"""
        if rc == 0:
            client.subscribe([
                (self.config.trigger_topic, 0),
                (self.config.emergency_topic, 0)
            ])
            logger.info(f"Subscribed to {self.config.trigger_topic} and {self.config.emergency_topic}")
            self._publish_event('mqtt_connected')
        else:
            logger.error(f"MQTT connection failed with code {rc}")
            # Don't enter ERROR state - MQTT already has retry logic
            # Just log the issue and let the retry mechanism handle it
            self._publish_event('mqtt_connection_failed', {'rc': rc})
    
    def _on_disconnect(self, client, userdata, rc, properties=None, reasoncode=None):
        """MQTT disconnection callback"""
        logger.warning(f"MQTT disconnected with code {rc}")
        if rc != 0:
            self._publish_event('mqtt_disconnected', {'rc': rc})
    
    def _on_message(self, client, userdata, msg):
        """MQTT message callback"""
        try:
            if msg.topic == self.config.trigger_topic:
                logger.info(f"Received fire trigger on {msg.topic}")
                self.handle_fire_trigger()
            elif msg.topic == self.config.emergency_topic:
                logger.warning(f"Received emergency command on {msg.topic}")
                self.handle_emergency_command(msg.payload.decode())
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    def handle_fire_trigger(self):
        """Handle fire detection trigger with safety checks.
        
        This is the primary entry point when consensus is reached. It implements
        the refill lockout safety system and handles triggers appropriately based
        on current state.
        
        Safety Checks:
            1. Refill Lockout: Ignores trigger if refill incomplete
            2. State Validation: Only starts pump from appropriate states
            3. Emergency Valve: Opens main valve immediately if closed
            4. Timer Management: Resets shutdown timer if already running
            
        State Handling:
            - IDLE/COOLDOWN: Start pump sequence
            - RUNNING/PRIMING/STARTING: Reset shutdown timer
            - STOPPING: Cancel shutdown if possible
            - REFILLING: Block trigger (safety lockout active)
            - ERROR: Ignore trigger (manual reset required)
            
        Side Effects:
            - Updates _last_trigger_time for telemetry
            - May start pump sequence or modify timers
            - Publishes events for all decisions
            - Opens main valve as emergency measure
            
        Thread Safety:
            Acquires lock for entire operation
        """
        with self._lock:
            self._last_trigger_time = time.time()
            
            # Check if refill is complete
            if not self._refill_complete:
                logger.warning("Fire trigger received but refill in progress")
                self._publish_event('trigger_blocked_refilling')
                return
            
            # Always ensure main valve is open when fire detected
            if not GPIO.input(self.config.main_valve_pin):
                self._set_pin('MAIN_VALVE', True)
                self._publish_event('emergency_valve_open')
            
            # Handle based on current state
            if self._state == PumpState.IDLE:
                self._start_pump_sequence()
            elif self._state == PumpState.COOLDOWN:
                # Restart if in cooldown
                self._state = PumpState.IDLE
                self._start_pump_sequence()
            elif self._state in [PumpState.RUNNING, PumpState.PRIMING, PumpState.STARTING]:
                # Already running or starting, just reset shutdown timer
                self._schedule_timer('fire_off_monitor', self._check_fire_off, self.config.fire_off_delay)
                logger.info("Fire trigger received while pump active, reset shutdown timer")
            elif self._state in [PumpState.REDUCING_RPM, PumpState.STOPPING]:
                # Cancel shutdown if in shutdown sequence
                if not self._shutting_down:
                    self._cancel_shutdown()
                else:
                    logger.warning("Cannot cancel shutdown - already in progress")
            elif self._state == PumpState.REFILLING:
                logger.warning("Fire trigger received during refill - cannot start pump")
                self._publish_event('trigger_blocked_refilling')
            elif self._state == PumpState.ERROR:
                logger.error("System in ERROR state - manual intervention required")
                self._publish_event('error_state_trigger_ignored')
    
    def handle_emergency_command(self, command):
        """Handle emergency bypass commands"""
        with self._lock:
            try:
                cmd_data = json.loads(command) if command.startswith('{') else {'action': command}
                action = cmd_data.get('action', '').lower()
                
                logger.warning(f"Processing emergency command: {action}")
                
                if action == 'start' or action == 'bypass_start':
                    # Emergency start bypass - force pump activation regardless of state
                    logger.warning("EMERGENCY BYPASS: Force starting pump")
                    self._emergency_start()
                    
                elif action == 'stop' or action == 'emergency_stop':
                    # Emergency stop - immediate shutdown
                    logger.warning("EMERGENCY STOP: Immediate pump shutdown")
                    self._emergency_stop()
                    
                elif action == 'valve_open':
                    # Emergency valve open - open main valve only
                    logger.warning("EMERGENCY: Opening main valve")
                    self._set_pin('MAIN_VALVE', True)
                    self._publish_event('emergency_valve_open')
                    
                elif action == 'valve_close':
                    # Emergency valve close
                    logger.warning("EMERGENCY: Closing main valve")
                    self._set_pin('MAIN_VALVE', False)
                    self._publish_event('emergency_valve_close')
                    
                elif action == 'reset':
                    # Emergency reset - clear error state
                    logger.warning("EMERGENCY RESET: Clearing error state")
                    self._emergency_reset()
                    
                else:
                    logger.error(f"Unknown emergency command: {action}")
                    self._publish_event('emergency_command_unknown', {'command': action})
                    
            except Exception as e:
                logger.error(f"Error processing emergency command: {e}")
                self._publish_event('emergency_command_error', {'error': str(e)})
    
    def _emergency_start(self):
        """Emergency start bypass - force pump activation"""
        logger.warning("EMERGENCY BYPASS START - Forcing pump activation")
        
        # Force state to IDLE to allow start sequence
        if self._state == PumpState.ERROR:
            self._state = PumpState.IDLE
            
        # Force start regardless of refill status
        self._refill_complete = True
        self._state = PumpState.IDLE
        
        # Start pump sequence
        self._start_pump_sequence()
        self._publish_event('emergency_bypass_start')
    
    def _emergency_stop(self):
        """Emergency stop - immediate shutdown"""
        logger.warning("EMERGENCY STOP - Immediate pump shutdown")
        
        # Cancel all timers first
        self._cancel_all_timers()
        
        # Use SafeGPIO emergency shutdown if available
        if self.gpio:
            # Build pin configuration for emergency shutdown
            pin_config = {
                'IGN_START': self.config.ign_start_pin,
                'IGN_ON': self.config.ign_on_pin,
                'IGN_OFF': self.config.ign_off_pin,
                'MAIN_VALVE': self.config.main_valve_pin,
                'REFILL_VALVE': self.config.refill_valve_pin,
                'PRIMING_VALVE': self.config.priming_valve_pin,
                'RPM_REDUCE': self.config.rpm_reduce_pin,
            }
            
            # Execute emergency shutdown
            results = self.gpio.emergency_all_off(pin_config)
            
            # Log results
            failed_pins = [pin for pin, success in results.items() if not success]
            if failed_pins:
                logger.critical(f"Emergency stop failed for pins: {failed_pins}")
                self._publish_event('emergency_stop_partial', {'failed_pins': failed_pins})
            else:
                logger.info("Emergency stop successful - all pins controlled")
                
        else:
            # Fallback to original implementation
            # Turn off all control pins immediately
            self._set_pin('IGN_START', False)
            self._set_pin('IGN_ON', False) 
            self._set_pin('IGN_OFF', True)  # Active stop signal
            time.sleep(0.5)
            self._set_pin('IGN_OFF', False)
            
            # Close main valve
            self._set_pin('MAIN_VALVE', False)
            self._set_pin('REFILL_VALVE', False)
            self._set_pin('PRIMING_VALVE', False)
            self._set_pin('RPM_REDUCE', False)
        
        # Set to cooldown state
        self._state = PumpState.COOLDOWN
        self._schedule_timer('cooldown_complete', self._enter_idle, self.config.cooldown_duration)
        
        self._publish_event('emergency_stop')
    
    def _emergency_reset(self):
        """Emergency reset - clear error state"""
        logger.warning("EMERGENCY RESET - Clearing error state")
        
        # Reset all pins to safe state
        self._set_pin('IGN_START', False)
        self._set_pin('IGN_ON', False)
        self._set_pin('IGN_OFF', False)
        self._set_pin('MAIN_VALVE', False)
        self._set_pin('REFILL_VALVE', False)
        self._set_pin('PRIMING_VALVE', False)
        self._set_pin('RPM_REDUCE', False)
        
        # Clear error state
        self._state = PumpState.IDLE
        self._refill_complete = True
        
        # Cancel all timers
        self._cancel_all_timers()
        
        self._publish_event('emergency_reset')
    
    def _start_pump_sequence(self):
        """Start the pump startup sequence with enhanced error handling"""
        with self._lock:
            if self._state != PumpState.IDLE:
                logger.warning(f"Cannot start pump from {self._state.name} state")
                return
            
            self._state = PumpState.PRIMING
            self._publish_event('pump_sequence_start')
            
            # Open main valve first (fail-safe) - CRITICAL for sprinklers
            if not self._set_pin('MAIN_VALVE', True):
                # This is critical - but try emergency valve opening approaches
                logger.critical("Main valve failed - attempting emergency valve procedures")
                
                # Try alternative valve control methods
                if not self._emergency_valve_open():
                    self._enter_error_state("Failed to open main valve - sprinkler system unavailable")
                    return
                else:
                    logger.warning("Emergency valve procedures succeeded - continuing")
            
            # Start priming (important but not critical for immediate sprinkler operation)
            if not self._set_pin('PRIMING_VALVE', True):
                logger.error("Failed to open priming valve - continuing without optimal priming")
                self._publish_event('priming_degraded', {'reason': 'valve_failure'})
            else:
                self._publish_event('priming_started')
            
            # Open refill valve immediately (ensures reservoir refilling)
            if not self._set_pin('REFILL_VALVE', True):
                logger.error("Failed to open refill valve - continuing pump sequence")
                self._publish_event('refill_valve_failed', {'continuing': True})
            else:
                self._publish_event('refill_valve_opened_immediately')
            
            # Schedule engine start after pre-open delay
            self._schedule_timer('start_engine', self._start_engine, self.config.pre_open_delay)
            
            # Schedule fire-off monitor
            self._schedule_timer('fire_off_monitor', self._check_fire_off, self.config.fire_off_delay)
    
    def _emergency_valve_open(self) -> bool:
        """Emergency valve opening procedures when normal methods fail"""
        logger.warning("Attempting emergency valve opening procedures")
        
        # Try direct GPIO manipulation with different timing
        pin = self.config.main_valve_pin
        try:
            # Method 1: Multiple rapid pulses
            for _ in range(5):
                GPIO.output(pin, GPIO.HIGH)
                time.sleep(0.1)
                GPIO.output(pin, GPIO.LOW)
                time.sleep(0.1)
                GPIO.output(pin, GPIO.HIGH)
                time.sleep(0.2)
                
                # Check if valve responded
                if GPIO.input(pin):
                    logger.info("Emergency valve opening successful")
                    self._publish_event('emergency_valve_success')
                    return True
            
            # Method 2: Extended activation time (for sticky valves)
            GPIO.output(pin, GPIO.HIGH)
            time.sleep(1.0)  # Hold longer
            
            if GPIO.input(pin):
                logger.info("Emergency valve opening successful (extended activation)")
                self._publish_event('emergency_valve_success_extended')
                return True
                
        except Exception as e:
            logger.error(f"Emergency valve procedures failed: {e}")
            self._publish_event('emergency_valve_failed', {'error': str(e)})
        
        return False
    
    def _start_engine(self):
        """Start the engine with safety checks and enhanced error handling"""
        with self._lock:
            if self._state != PumpState.PRIMING:
                logger.warning(f"Cannot start engine from {self._state.name} state")
                return
            
            # Safety check: Ensure main valve is open
            if not GPIO.input(self.config.main_valve_pin):
                self._enter_error_state("Main valve not open - aborting engine start")
                return
            
            self._state = PumpState.STARTING
            self._publish_event('engine_start_sequence')
            
            # Allow time for the state to be observed
            time.sleep(0.1)
            
            # Start ignition sequence with retry logic
            if not self._set_pin('IGN_START', True):
                logger.error("Primary ignition start failed - attempting recovery")
                # Continue anyway - engine might start without perfect ignition signal
                self._publish_event('ignition_start_degraded')
            
            # Hold ignition start for configured duration
            time.sleep(self.config.ignition_start_duration)
            
            # Release ignition start
            self._set_pin('IGN_START', False)
            
            # Turn on engine
            if not self._set_pin('IGN_ON', True):
                logger.critical("Failed to turn on ignition - critical for pump operation")
                # This is more critical but try alternative ignition methods
                if not self._emergency_ignition_start():
                    self._enter_error_state("Failed to turn on ignition - engine unavailable")
                    return
                else:
                    logger.warning("Emergency ignition procedures succeeded")
            
            # Engine is now running
            self._state = PumpState.RUNNING
            self._engine_start_time = time.time()
            self._current_runtime = 0
            self._low_pressure_detected = False
            self._publish_event('engine_running')
            
            # Schedule priming valve close after priming duration
            self._schedule_timer(
                'close_priming',
                lambda: self._close_priming_valve(),
                self.config.priming_duration
            )
            
            # Schedule RPM reduction
            rpm_reduction_time = self.config.max_engine_runtime - self.config.rpm_reduction_lead
            if rpm_reduction_time > 0:
                self._schedule_timer('rpm_reduction', self._reduce_rpm, rpm_reduction_time)
            
            # Schedule max runtime shutdown
            self._schedule_timer('max_runtime', self._shutdown_engine, self.config.max_engine_runtime)
    
    def _emergency_ignition_start(self) -> bool:
        """Emergency ignition procedures when normal start fails"""
        logger.warning("Attempting emergency ignition procedures")
        
        try:
            # Method 1: Extended cranking sequence
            for attempt in range(3):
                logger.info(f"Emergency ignition attempt {attempt + 1}")
                
                # Longer crank time
                self._set_pin('IGN_START', True, max_retries=1)
                time.sleep(self.config.ignition_start_duration * 2)
                self._set_pin('IGN_START', False, max_retries=1)
                
                time.sleep(0.5)  # Rest between attempts
                
                # Try to turn on ignition
                if self._set_pin('IGN_ON', True, max_retries=1):
                    # Verify ignition is actually on
                    time.sleep(0.2)
                    if GPIO.input(self.config.ign_on_pin):
                        logger.info("Emergency ignition successful")
                        self._publish_event('emergency_ignition_success')
                        return True
                
                # Reset for next attempt
                self._set_pin('IGN_ON', False, max_retries=1)
                time.sleep(1.0)
            
            logger.error("All emergency ignition attempts failed")
            self._publish_event('emergency_ignition_failed')
            return False
            
        except Exception as e:
            logger.error(f"Emergency ignition procedures failed: {e}")
            return False
    
    def _close_priming_valve(self):
        """Close priming valve and start pressure monitoring"""
        with self._lock:
            self._set_pin('PRIMING_VALVE', False)
            self._publish_event('priming_complete')
            
            # Schedule pressure check if monitoring is enabled
            if self.config.line_pressure_pin:
                self._schedule_timer(
                    'pressure_check',
                    self._check_line_pressure,
                    self.config.pressure_check_delay
                )
    
    def _check_line_pressure(self):
        """Check line pressure after priming period"""
        with self._lock:
            if self._state != PumpState.RUNNING:
                return
            
            if not self._is_line_pressure_ok():
                logger.error("Low line pressure detected!")
                self._low_pressure_detected = True
                self._state = PumpState.LOW_PRESSURE
                self._publish_event('low_pressure_detected')
                self._shutdown_engine()
    
    def _reduce_rpm(self):
        """Reduce engine RPM before shutdown"""
        with self._lock:
            if self._state != PumpState.RUNNING:
                logger.warning(f"Cannot reduce RPM from {self._state.name} state")
                return
            
            self._state = PumpState.REDUCING_RPM
            self._set_pin('RPM_REDUCE', True)
            self._publish_event('rpm_reduced')
    
    def _check_fire_off(self):
        """Check if fire has been off long enough to shutdown"""
        with self._lock:
            time_since_trigger = time.time() - self._last_trigger_time
            
            if time_since_trigger >= self.config.fire_off_delay:
                logger.info(f"No fire trigger for {time_since_trigger:.1f}s, initiating shutdown")
                
                # NEW LOGIC: Check if we need RPM reduction first
                if self._state == PumpState.RUNNING and not self._has_timer('delayed_shutdown_after_rpm'):
                    # Start RPM reduction sequence before shutdown
                    logger.info("Starting RPM reduction before shutdown")
                    self._cancel_timer('rpm_reduction')  # Cancel the scheduled one
                    self._reduce_rpm()
                    
                    # Schedule actual shutdown after RPM reduction period
                    self._schedule_timer(
                        'delayed_shutdown_after_rpm', 
                        self._shutdown_engine, 
                        self.config.rpm_reduction_duration
                    )
                elif self._state == PumpState.REDUCING_RPM:
                    # Already reducing RPM, shutdown will happen via timer
                    logger.debug("RPM reduction in progress, shutdown scheduled")
                else:
                    # Direct shutdown for other states or if already scheduled
                    if not self._has_timer('delayed_shutdown_after_rpm'):
                        self._shutdown_engine()
            else:
                # Re-schedule check
                remaining = self.config.fire_off_delay - time_since_trigger
                self._schedule_timer('fire_off_monitor', self._check_fire_off, remaining)
    
    def _shutdown_engine(self):
        """Shutdown engine with proper sequence"""
        with self._lock:
            if self._state not in [PumpState.RUNNING, PumpState.REDUCING_RPM, PumpState.LOW_PRESSURE]:
                logger.warning(f"Cannot shutdown from {self._state.name} state")
                return
            
            if self._shutting_down:
                logger.warning("Shutdown already in progress")
                return
            
            self._shutting_down = True
            previous_state = self._state
            
            # Ensure RPM is reduced if coming directly from RUNNING state
            if previous_state == PumpState.RUNNING:
                logger.warning("Direct shutdown from RUNNING - applying brief RPM reduction")
                self._set_pin('RPM_REDUCE', True)
                time.sleep(2.0)  # Brief RPM reduction for emergency/direct shutdown cases
            
            self._state = PumpState.STOPPING
            self._publish_event('shutdown_initiated', {'reason': previous_state.name})
            
            # Cancel max runtime timer if still active
            self._cancel_timer('max_runtime')
            self._cancel_timer('pressure_check')
            self._cancel_timer('delayed_shutdown_after_rpm')  # Cancel if exists
            
            # Calculate total runtime
            if self._engine_start_time:
                self._current_runtime = time.time() - self._engine_start_time
                self._total_runtime += self._current_runtime
            else:
                self._current_runtime = 0
            
            # Pulse ignition off
            self._set_pin('IGN_OFF', True)
            self._schedule_timer(
                'ign_off_pulse',
                lambda: self._set_pin('IGN_OFF', False),
                self.config.ignition_off_duration
            )
            
            # Turn off engine
            self._set_pin('IGN_ON', False)
            self._set_pin('RPM_REDUCE', False)
            
            # Schedule valve closures
            self._schedule_timer(
                'close_main_valve',
                lambda: self._set_pin('MAIN_VALVE', False),
                self.config.valve_close_delay
            )
            
            # Calculate refill time based on runtime
            refill_time = self._current_runtime * self.config.refill_multiplier
            
            # Start refill process (unless low pressure detected)
            if not self._low_pressure_detected:
                self._refill_complete = False
                self._state = PumpState.REFILLING
                
                # Refill valve is already open (opened at engine start)
                self._publish_event('refill_continuing', {'duration': refill_time})
                
                # Schedule refill valve closure
                self._schedule_timer(
                    'close_refill_valve',
                    self._complete_refill,
                    refill_time
                )
            else:
                # Close refill valve if low pressure (possible leak)
                self._set_pin('REFILL_VALVE', False)
                self._enter_cooldown()
            
            self._publish_event('shutdown_complete', {'runtime': self._current_runtime})
    
    def _complete_refill(self):
        """Complete the refill process"""
        with self._lock:
            self._set_pin('REFILL_VALVE', False)
            self._refill_complete = True
            self._publish_event('refill_complete_timer')
            self._enter_cooldown()
    
    def _enter_cooldown(self):
        """Enter cooldown state after shutdown"""
        with self._lock:
            self._state = PumpState.COOLDOWN
            self._shutting_down = False
            self._engine_start_time = None
            self._current_runtime = 0
            self._publish_event('cooldown_entered')
            
            # Schedule return to idle
            self._schedule_timer('cooldown_complete', self._enter_idle, self.config.cooldown_duration)
    
    def _enter_idle(self):
        """Return to idle state"""
        with self._lock:
            self._state = PumpState.IDLE
            
            # Ensure all pins are in proper IDLE state
            # This handles cases where pins were left in unexpected states
            self._set_pin('MAIN_VALVE', False)
            self._set_pin('IGN_ON', False)
            self._set_pin('IGN_START', False)
            self._set_pin('IGN_OFF', False)
            self._set_pin('REFILL_VALVE', False)
            self._set_pin('PRIMING_VALVE', False)
            self._set_pin('RPM_REDUCE', False)
            
            self._publish_event('idle_state_entered')
    
    def _cancel_shutdown(self):
        """Cancel ongoing shutdown if possible"""
        with self._lock:
            if self._state not in [PumpState.REDUCING_RPM, PumpState.STOPPING]:
                return
            logger.info("Cancelling shutdown sequence")
            
            # Cancel shutdown timers
            self._cancel_timer('close_main_valve')
            self._cancel_timer('close_refill_valve')
            self._cancel_timer('enter_cooldown')
            self._cancel_timer('ign_off_pulse')
            
            # Restore engine state
            self._set_pin('IGN_OFF', False)
            self._set_pin('IGN_ON', True)
            self._set_pin('RPM_REDUCE', False)
            
            # Return to running state
            self._state = PumpState.RUNNING
            self._shutting_down = False
            
            # Reschedule monitors
            self._schedule_timer('fire_off_monitor', self._check_fire_off, self.config.fire_off_delay)
            
            # Reschedule max runtime if needed
            if self._engine_start_time:
                elapsed = time.time() - self._engine_start_time
                remaining = self.config.max_engine_runtime - elapsed
                if remaining > 0:
                    self._schedule_timer('max_runtime', self._shutdown_engine, remaining)
            
            self._publish_event('shutdown_cancelled')
    
    def _enter_error_state(self, reason: str):
        """Enter error state requiring manual intervention"""
        with self._lock:
            # Prevent recursive error state entry
            if self._state == PumpState.ERROR:
                logger.debug(f"Already in ERROR state, ignoring: {reason}")
                return
                
            logger.error(f"Entering ERROR state: {reason}")
            self._state = PumpState.ERROR
            self._last_error = reason
            
            # Best effort to ensure all pins are LOW for safety (don't recurse on failure)
            try:
                # Temporarily disable error state transitions during cleanup
                original_state = self._state
                
                # Turn off all pins for safety - ERROR state requires all pins LOW
                logger.info("Setting all pins to LOW for ERROR state safety")
                for pin_name in ['IGN_ON', 'IGN_START', 'IGN_OFF', 'MAIN_VALVE', 
                                'REFILL_VALVE', 'PRIMING_VALVE', 'RPM_REDUCE']:
                    logger.debug(f"Setting {pin_name} to LOW")
                    self._set_pin(pin_name, False)
                
                self._state = original_state  # Ensure we stay in ERROR
                logger.info("All pins set to LOW in ERROR state")
            except Exception as e:
                logger.error(f"Failed to turn off pins during error state entry: {e}")
            
            self._publish_event('error_state_entered', {'reason': reason})
            
            # Cancel all timers
            self._cancel_all_timers()
            
            # Keep health monitoring active
            self._schedule_timer('health', self._publish_health, self.config.health_interval)
    
    def _validate_hardware(self):
        """Validate hardware state and detect failures (optional)"""
        if not self.config.hardware_validation_enabled:
            return
        
        current_time = time.time()
        self._last_hardware_check = current_time
        
        try:
            # Check relay feedback pins if configured
            if self.config.relay_feedback_pins:
                for i, pin_str in enumerate(self.config.relay_feedback_pins):
                    if pin_str.strip():
                        pin = int(pin_str.strip())
                        feedback_state = GPIO.input(pin)
                        expected_state = self._get_expected_relay_state(i)
                        
                        if feedback_state != expected_state:
                            self._hardware_failures += 1
                            logger.warning(f"Hardware validation failure: Relay {i+1} feedback mismatch")
                            self._publish_event('hardware_validation_failure', {
                                'relay': i+1,
                                'expected': expected_state,
                                'actual': feedback_state
                            })
                        else:
                            self._hardware_status[f'relay_{i+1}'] = 'OK'
            
            # Check simulation mode status
            if not GPIO_AVAILABLE and self.config.simulation_mode_warnings:
                self._publish_event('simulation_mode_warning', {
                    'message': 'System running in simulation mode - no physical hardware control'
                })
            
        except Exception as e:
            logger.error(f"Hardware validation error: {e}")
            self._hardware_failures += 1
        
        # Reschedule next check
        self._schedule_timer('hardware_check', self._validate_hardware, self.config.hardware_check_interval)
    
    def _get_expected_relay_state(self, relay_index: int) -> bool:
        """Get expected state for relay based on current system state"""
        # This is a simplified example - actual implementation would depend on hardware wiring
        if relay_index == 0:  # Assuming relay 0 is main valve
            return GPIO.input(self.config.main_valve_pin)
        elif relay_index == 1:  # Assuming relay 1 is ignition
            return GPIO.input(self.config.ign_on_pin)
        return False  # Default to off for unknown relays
    
    def _monitor_dry_run_protection(self):
        """Monitor for pump running without water flow (dry run protection)"""
        while not self._shutdown and not getattr(self, '_shutting_down', False):
            try:
                with self._lock:
                    current_time = time.time()
                    
                    # Check if pump is running
                    pump_running = (self._state == PumpState.RUNNING and 
                                  GPIO.input(self.config.ign_on_pin))
                    
                    if pump_running:
                        # Initialize pump start time if not set
                        if self._pump_start_time is None:
                            self._pump_start_time = current_time
                            self._water_flow_detected = False
                            logger.debug(f"Dry run monitor: pump start time set, flow sensor: {self.config.flow_sensor_pin}")
                        
                        # Check for water flow if sensor is available
                        if self.config.flow_sensor_pin:
                            flow_detected = GPIO.input(self.config.flow_sensor_pin)
                            if flow_detected:
                                self._water_flow_detected = True
                        else:
                            # If no flow sensor, check line pressure as proxy
                            if self.config.line_pressure_pin:
                                pressure_ok = self._is_line_pressure_ok()
                                if pressure_ok:
                                    self._water_flow_detected = True
                            else:
                                # No sensors available - assume water flow after priming period
                                if current_time - self._pump_start_time > self.config.priming_duration:
                                    self._water_flow_detected = True
                        
                        # Check dry run time limit
                        dry_run_time = current_time - self._pump_start_time
                        
                        
                        if dry_run_time > self.config.max_dry_run_time and not self._water_flow_detected:
                            logger.critical(f"DRY RUN PROTECTION: Pump running {dry_run_time:.1f}s without water flow!")
                            self._publish_event('dry_run_protection_triggered', {
                                'dry_run_time': dry_run_time,
                                'max_allowed': self.config.max_dry_run_time
                            })
                            self._enter_error_state(f"Dry run protection: {dry_run_time:.1f}s without water flow")
                            # Exit the monitor loop after entering error state
                            return
                        elif dry_run_time > self.config.max_dry_run_time * 0.8 and not self._water_flow_detected:
                            # Warning at 80% of limit
                            self._dry_run_warnings += 1
                            if self._dry_run_warnings % 5 == 1:  # Log every 5th warning to avoid spam
                                logger.warning(f"DRY RUN WARNING: Pump running {dry_run_time:.1f}s without detected water flow")
                    else:
                        # Reset dry run tracking when pump is not running
                        self._pump_start_time = None
                        self._water_flow_detected = False
                        self._dry_run_warnings = 0
            
            except Exception as e:
                logger.error(f"Dry run protection monitoring error: {e}")
            
            # Sleep with shutdown check
            for _ in range(10):
                if self._shutdown:
                    return  # Exit immediately
                time.sleep(0.1)
    
    def _monitor_emergency_button(self):
        """Monitor emergency button for manual fire trigger (optional)"""
        last_state = None
        debounce_time = 0.1  # 100ms debounce
        
        while not self._shutdown:
            try:
                if self.config.emergency_button_pin:
                    current_state = GPIO.input(self.config.emergency_button_pin)
                    active_state = not current_state if self.config.emergency_button_active_low else current_state
                    
                    # Detect button press (transition to active)
                    if last_state is not None and not last_state and active_state:
                        logger.warning("EMERGENCY BUTTON PRESSED - Manual fire trigger activated!")
                        self._publish_event('emergency_button_pressed')
                        self.handle_fire_trigger()
                        time.sleep(1)  # Prevent multiple triggers
                    
                    last_state = active_state
                    time.sleep(debounce_time)
                else:
                    time.sleep(1)
            
            except Exception as e:
                logger.error(f"Emergency button monitoring error: {e}")
                time.sleep(1)

    def _emergency_switch_callback(self, channel):
        """Callback for emergency switch interrupt."""
        logger.warning(f"Emergency switch activated on channel {channel}")
        self.handle_fire_trigger()
    
    def _publish_health(self):
        """Publish periodic health status with enhanced reporting"""
        with self._lock:
            health_data = {
                'uptime': time.time(),
                'total_runtime': self._total_runtime,
                'current_runtime': self._current_runtime,
                'state': self._state.name,
                'last_trigger': self._last_trigger_time,
                'refill_complete': self._refill_complete,
                'low_pressure_detected': self._low_pressure_detected,
            }
            
            # Enhanced status reporting
            if self.config.enhanced_status_enabled:
                # Hardware status
                health_data['hardware'] = {
                    'gpio_available': GPIO_AVAILABLE,
                    'simulation_mode': not GPIO_AVAILABLE,
                    'validation_enabled': self.config.hardware_validation_enabled,
                    'last_hardware_check': self._last_hardware_check,
                    'hardware_failures': self._hardware_failures,
                    'hardware_status': self._hardware_status.copy(),
                }
                
                # Dry run protection status (always enabled)
                health_data['dry_run_protection'] = {
                    'enabled': True,
                    'pump_running': self._pump_start_time is not None,
                    'water_flow_detected': self._water_flow_detected,
                    'dry_run_warnings': self._dry_run_warnings,
                    'max_dry_run_time': self.config.max_dry_run_time,
                }
                if self._pump_start_time:
                    health_data['dry_run_protection']['current_runtime'] = time.time() - self._pump_start_time
                
                # Safety feature status
                health_data['safety_features'] = {
                    'emergency_button_available': self.config.emergency_button_pin is not None,
                    'flow_sensor_available': self.config.flow_sensor_pin is not None,
                    'reservoir_sensor_available': self.config.reservoir_float_pin is not None,
                    'pressure_sensor_available': self.config.line_pressure_pin is not None,
                }
                
                # Critical warnings for simulation mode
                if not GPIO_AVAILABLE and self.config.simulation_mode_warnings:
                    health_data['critical_warnings'] = [
                        'SIMULATION_MODE_ACTIVE',
                        'NO_PHYSICAL_HARDWARE_CONTROL',
                        'PUMP_WILL_NOT_OPERATE_IN_EMERGENCY'
                    ]
            
            # Add sensor states if available
            if self.config.reservoir_float_pin:
                health_data['reservoir_full'] = self._is_reservoir_full()
            
            if self.config.line_pressure_pin:
                health_data['line_pressure_ok'] = self._is_line_pressure_ok()
            
            # Include pin states for diagnostics
            health_data['pin_states'] = self._get_state_snapshot()
            
            self._publish_event('health_report', health_data)
            
            # Reschedule
            self._schedule_timer('health', self._publish_health, self.config.health_interval)
    
    def cleanup(self):
        """Clean shutdown of controller"""
        logger.info("Cleaning up PumpController")
        
        # Set shutdown flag to stop monitoring threads
        self._shutdown = True
        
        # Wait for background threads to exit
        for thread in self._background_threads:
            if thread.is_alive():
                thread.join(timeout=2.0)  # Give threads 2 seconds to exit cleanly
        
        with self._lock:
            # Cancel all timers
            self._cancel_all_timers()
            
            # Ensure pump is off
            if self._state in [PumpState.RUNNING, PumpState.REDUCING_RPM]:
                self._shutdown_engine()
                # Use shorter sleep for tests
                sleep_time = 0.5 if hasattr(self, '_test_mode') else 5
                time.sleep(sleep_time)  # Allow shutdown to complete
            
            # Close all valves
            for pin_name in ['MAIN_VALVE', 'REFILL_VALVE', 'PRIMING_VALVE']:
                self._set_pin(pin_name, False)
            
            # Turn off all control pins
            for pin_name in ['IGN_START', 'IGN_ON', 'IGN_OFF', 'RPM_REDUCE']:
                self._set_pin(pin_name, False)
        
        # Disconnect MQTT
        try:
            self._publish_event('controller_shutdown')
            self.client.loop_stop()
            self.client.disconnect()
        except Exception as e:
            logger.debug(f"Error during MQTT cleanup: {e}")
        
        # Give threads a moment to terminate
        # Use longer timeout for non-test mode to ensure all threads exit
        thread_timeout = 0.5 if hasattr(self, '_test_mode') else 1.0
        time.sleep(thread_timeout)
        
        # Cleanup GPIO
        if GPIO_AVAILABLE:
            GPIO.cleanup()

# ─────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────
controller = None

def main():
    global controller
    
    try:
        # Create configuration from environment
        config = PumpControllerConfig()
        controller = PumpController(config=config)
        logger.info("PumpController started successfully")
        
        # Keep running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        if controller:
            controller.cleanup()
        logger.info("PumpController stopped")

if __name__ == '__main__':
    main()
