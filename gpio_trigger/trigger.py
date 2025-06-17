#!/usr/bin/env python3.12
"""
PumpController for wildfire-watch:
- Fail-safe, thread-safe engine and valve control
- State machine approach for consistent operation
- Comprehensive error recovery and safety checks
- Idempotent operations resilient to concurrent events
- Reservoir level and line pressure monitoring
- Refill valve opens immediately on engine start
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

load_dotenv()

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
CONFIG = {
    'MQTT_BROKER': os.getenv('MQTT_BROKER', 'mqtt_broker'),
    'MQTT_PORT': int(os.getenv('MQTT_PORT', '1883')),
    'MQTT_TLS': os.getenv('MQTT_TLS', 'false').lower() == 'true',
    'TLS_CA_PATH': os.getenv('TLS_CA_PATH', '/mnt/data/certs/ca.crt'),
    'TRIGGER_TOPIC': os.getenv('TRIGGER_TOPIC', 'fire/trigger'),
    'EMERGENCY_TOPIC': os.getenv('EMERGENCY_TOPIC', 'fire/emergency'),
    'TELEMETRY_TOPIC': os.getenv('TELEMETRY_TOPIC', 'system/trigger_telemetry'),
    
    # GPIO Pins - Control
    'MAIN_VALVE_PIN': int(os.getenv('MAIN_VALVE_PIN', '18')),
    'IGN_START_PIN': int(os.getenv('IGNITION_START_PIN', '23')),
    'IGN_ON_PIN': int(os.getenv('IGNITION_ON_PIN', '24')),
    'IGN_OFF_PIN': int(os.getenv('IGNITION_OFF_PIN', '25')),
    'REFILL_VALVE_PIN': int(os.getenv('REFILL_VALVE_PIN', '22')),
    'PRIMING_VALVE_PIN': int(os.getenv('PRIMING_VALVE_PIN', '26')),
    'RPM_REDUCE_PIN': int(os.getenv('RPM_REDUCE_PIN', '27')),
    
    # GPIO Pins - Monitoring (Optional)
    'RESERVOIR_FLOAT_PIN': int(os.getenv('RESERVOIR_FLOAT_PIN', '16')) if os.getenv('RESERVOIR_FLOAT_PIN') else None,
    'LINE_PRESSURE_PIN': int(os.getenv('LINE_PRESSURE_PIN', '20')) if os.getenv('LINE_PRESSURE_PIN') else None,
    
    # Timing Configuration
    'PRE_OPEN_DELAY': float(os.getenv('VALVE_PRE_OPEN_DELAY', '2')),
    'IGNITION_START_DURATION': float(os.getenv('IGNITION_START_DURATION', '5')),
    'FIRE_OFF_DELAY': float(os.getenv('FIRE_OFF_DELAY', '1800')),
    'VALVE_CLOSE_DELAY': float(os.getenv('VALVE_CLOSE_DELAY', '600')),
    'IGNITION_OFF_DURATION': float(os.getenv('IGNITION_OFF_DURATION', '5')),
    'MAX_ENGINE_RUNTIME': float(os.getenv('MAX_ENGINE_RUNTIME', '1800')),  # 30 minutes default
    'REFILL_MULTIPLIER': float(os.getenv('REFILL_MULTIPLIER', '40')),
    'PRIMING_DURATION': float(os.getenv('PRIMING_DURATION', '180')),
    'RPM_REDUCTION_LEAD': float(os.getenv('RPM_REDUCTION_LEAD', '15')),
    'PRESSURE_CHECK_DELAY': float(os.getenv('PRESSURE_CHECK_DELAY', '60')),  # 1 minute after priming
    'HEALTH_INTERVAL': float(os.getenv('TELEMETRY_INTERVAL', '60')),
    'ACTION_RETRY_INTERVAL': float(os.getenv('ACTION_RETRY_INTERVAL', '60')),
    
    # Safety Configuration
    'RESERVOIR_FLOAT_ACTIVE_LOW': os.getenv('RESERVOIR_FLOAT_ACTIVE_LOW', 'true').lower() == 'true',
    'LINE_PRESSURE_ACTIVE_LOW': os.getenv('LINE_PRESSURE_ACTIVE_LOW', 'true').lower() == 'true',
    
    # Hardware Validation (Optional)
    'HARDWARE_VALIDATION_ENABLED': os.getenv('HARDWARE_VALIDATION_ENABLED', 'false').lower() == 'true',
    'RELAY_FEEDBACK_PINS': os.getenv('RELAY_FEEDBACK_PINS', '').split(',') if os.getenv('RELAY_FEEDBACK_PINS') else [],
    'HARDWARE_CHECK_INTERVAL': float(os.getenv('HARDWARE_CHECK_INTERVAL', '30')),
    
    # Dry Run Protection (Always enabled for safety)
    'MAX_DRY_RUN_TIME': float(os.getenv('MAX_DRY_RUN_TIME', '300')),  # 5 minutes default
    'FLOW_SENSOR_PIN': int(os.getenv('FLOW_SENSOR_PIN', '')) if os.getenv('FLOW_SENSOR_PIN') else None,
    
    # Emergency Features (Optional)
    'EMERGENCY_BUTTON_PIN': int(os.getenv('EMERGENCY_BUTTON_PIN', '')) if os.getenv('EMERGENCY_BUTTON_PIN') else None,
    'EMERGENCY_BUTTON_ACTIVE_LOW': os.getenv('EMERGENCY_BUTTON_ACTIVE_LOW', 'true').lower() == 'true',
    
    # Status Reporting
    'ENHANCED_STATUS_ENABLED': os.getenv('ENHANCED_STATUS_ENABLED', 'true').lower() == 'true',
    'SIMULATION_MODE_WARNINGS': os.getenv('SIMULATION_MODE_WARNINGS', 'true').lower() == 'true',
}

# ─────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────
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
    if CONFIG['SIMULATION_MODE_WARNINGS']:
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
    """State machine for pump controller"""
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
class PumpController:
    def __init__(self):
        self.cfg = CONFIG
        self._lock = threading.RLock()
        self._state = PumpState.IDLE
        self._timers: Dict[str, threading.Timer] = {}
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
        
        # Shutdown flag for clean thread termination
        self._shutdown = False
        
        # Initialize GPIO
        self._init_gpio()
        
        # Setup MQTT
        self._setup_mqtt()
        
        # Start monitoring tasks
        self._start_monitoring_tasks()
        
        # Start health monitoring
        self._schedule_timer('health', self._publish_health, self.cfg['HEALTH_INTERVAL'])
        
        logger.info(f"PumpController initialized in {self._state.name} state")
    
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
            pin = self.cfg[pin_name]
            GPIO.setup(pin, GPIO.OUT, initial=initial_state)
            logger.debug(f"Initialized {pin_name} (pin {pin}) to {initial_state}")
        
        # Input pins (optional monitoring)
        if self.cfg['RESERVOIR_FLOAT_PIN']:
            pull = GPIO.PUD_DOWN if self.cfg['RESERVOIR_FLOAT_ACTIVE_LOW'] else GPIO.PUD_UP
            GPIO.setup(self.cfg['RESERVOIR_FLOAT_PIN'], GPIO.IN, pull_up_down=pull)
            logger.info(f"Reservoir float switch on pin {self.cfg['RESERVOIR_FLOAT_PIN']}")
        
        if self.cfg['LINE_PRESSURE_PIN']:
            pull = GPIO.PUD_DOWN if self.cfg['LINE_PRESSURE_ACTIVE_LOW'] else GPIO.PUD_UP
            GPIO.setup(self.cfg['LINE_PRESSURE_PIN'], GPIO.IN, pull_up_down=pull)
            logger.info(f"Line pressure switch on pin {self.cfg['LINE_PRESSURE_PIN']}")
        
        # Optional hardware validation pins
        if self.cfg['RELAY_FEEDBACK_PINS']:
            for i, pin_str in enumerate(self.cfg['RELAY_FEEDBACK_PINS']):
                if pin_str.strip():
                    pin = int(pin_str.strip())
                    GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
                    logger.info(f"Relay feedback pin {i+1} configured on pin {pin}")
        
        # Optional flow sensor pin
        if self.cfg['FLOW_SENSOR_PIN']:
            GPIO.setup(self.cfg['FLOW_SENSOR_PIN'], GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
            logger.info(f"Water flow sensor on pin {self.cfg['FLOW_SENSOR_PIN']}")
        
        # Optional emergency button pin
        if self.cfg['EMERGENCY_BUTTON_PIN']:
            pull = GPIO.PUD_UP if self.cfg['EMERGENCY_BUTTON_ACTIVE_LOW'] else GPIO.PUD_DOWN
            GPIO.setup(self.cfg['EMERGENCY_BUTTON_PIN'], GPIO.IN, pull_up_down=pull)
            logger.info(f"Emergency button on pin {self.cfg['EMERGENCY_BUTTON_PIN']}")
    
    def _setup_mqtt(self):
        """Setup MQTT client with TLS if configured"""
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, clean_session=True)
        
        # Set callbacks
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        
        # Configure TLS if enabled
        if self.cfg['MQTT_TLS']:
            import ssl
            self.client.tls_set(
                ca_certs=self.cfg['TLS_CA_PATH'],
                cert_reqs=ssl.CERT_REQUIRED,
                tls_version=ssl.PROTOCOL_TLS
            )
            logger.info("MQTT TLS enabled")
        
        # Set LWT
        lwt_topic = f"{self.cfg['TELEMETRY_TOPIC']}/{socket.gethostname()}/lwt"
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
        while not self._shutdown:
            try:
                port = 8883 if self.cfg['MQTT_TLS'] else self.cfg['MQTT_PORT']
                self.client.connect(self.cfg['MQTT_BROKER'], port, keepalive=60)
                self.client.loop_start()
                logger.info(f"MQTT client connected to {self.cfg['MQTT_BROKER']}:{port}")
                break
            except Exception as e:
                retry_count += 1
                if max_retries is not None and retry_count >= max_retries:
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
                'engine_on': GPIO.input(self.cfg['IGN_ON_PIN']),
                'main_valve': GPIO.input(self.cfg['MAIN_VALVE_PIN']),
                'refill_valve': GPIO.input(self.cfg['REFILL_VALVE_PIN']),
                'priming_valve': GPIO.input(self.cfg['PRIMING_VALVE_PIN']),
                'rpm_reduced': GPIO.input(self.cfg['RPM_REDUCE_PIN']),
                'total_runtime': self._total_runtime,
                'current_runtime': self._current_runtime,
                'shutting_down': self._shutting_down,
                'refill_complete': self._refill_complete,
                'active_timers': list(self._timers.keys()),
            }
            
            # Add monitoring status if available
            if self.cfg['RESERVOIR_FLOAT_PIN']:
                snapshot['reservoir_full'] = self._is_reservoir_full()
            
            if self.cfg['LINE_PRESSURE_PIN']:
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
                self.cfg['TELEMETRY_TOPIC'],
                json.dumps(payload),
                qos=1
            )
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
    
    def _set_pin(self, pin_name: str, state: bool, max_retries: int = 3) -> bool:
        """Set GPIO pin state with error handling and retry logic"""
        pin = self.cfg[f'{pin_name}_PIN']
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
        if not self.cfg['RESERVOIR_FLOAT_PIN']:
            return True  # Assume full if no sensor
        
        state = GPIO.input(self.cfg['RESERVOIR_FLOAT_PIN'])
        # Active low means LOW = full, HIGH = not full
        if self.cfg['RESERVOIR_FLOAT_ACTIVE_LOW']:
            return not state
        else:
            return state
    
    def _is_line_pressure_ok(self) -> bool:
        """Check if line pressure is adequate"""
        if not self.cfg['LINE_PRESSURE_PIN']:
            return True  # Assume OK if no sensor
        
        state = GPIO.input(self.cfg['LINE_PRESSURE_PIN'])
        # Active low means LOW = pressure OK, HIGH = low pressure
        if self.cfg['LINE_PRESSURE_ACTIVE_LOW']:
            return not state
        else:
            return state
    
    def _schedule_timer(self, name: str, func: Callable, delay: float):
        """Schedule a timer, canceling any existing timer with same name"""
        self._cancel_timer(name)
        
        def wrapped_func():
            with self._lock:
                self._timers.pop(name, None)
                try:
                    func()
                except Exception as e:
                    logger.error(f"Timer {name} failed: {e}")
                    self._publish_event('timer_error', {'timer': name, 'error': str(e)})
        
        timer = threading.Timer(delay, wrapped_func)
        timer.daemon = True
        timer.start()
        self._timers[name] = timer
        logger.debug(f"Scheduled timer '{name}' for {delay}s")
    
    def _cancel_timer(self, name: str):
        """Cancel a timer if it exists"""
        timer = self._timers.pop(name, None)
        if timer and timer.is_alive():
            timer.cancel()
            logger.debug(f"Cancelled timer '{name}'")
    
    def _cancel_all_timers(self):
        """Cancel all active timers"""
        for name in list(self._timers.keys()):
            self._cancel_timer(name)
    
    def _start_monitoring_tasks(self):
        """Start background monitoring tasks"""
        # Monitor reservoir level during refill
        if self.cfg['RESERVOIR_FLOAT_PIN']:
            threading.Thread(target=self._monitor_reservoir_level, daemon=True).start()
        
        # Start hardware validation monitoring if enabled
        if self.cfg['HARDWARE_VALIDATION_ENABLED']:
            self._schedule_timer('hardware_check', self._validate_hardware, self.cfg['HARDWARE_CHECK_INTERVAL'])
        
        # Start dry run protection monitoring (always enabled for safety)
        threading.Thread(target=self._monitor_dry_run_protection, daemon=True).start()
        
        # Start emergency button monitoring if configured
        if self.cfg['EMERGENCY_BUTTON_PIN']:
            threading.Thread(target=self._monitor_emergency_button, daemon=True).start()
    
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
                                self._state = PumpState.IDLE
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
                (self.cfg['TRIGGER_TOPIC'], 0),
                (self.cfg['EMERGENCY_TOPIC'], 0)
            ])
            logger.info(f"Subscribed to {self.cfg['TRIGGER_TOPIC']} and {self.cfg['EMERGENCY_TOPIC']}")
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
            if msg.topic == self.cfg['TRIGGER_TOPIC']:
                logger.info(f"Received fire trigger on {msg.topic}")
                self.handle_fire_trigger()
            elif msg.topic == self.cfg['EMERGENCY_TOPIC']:
                logger.warning(f"Received emergency command on {msg.topic}")
                self.handle_emergency_command(msg.payload.decode())
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    def handle_fire_trigger(self):
        """Handle fire detection trigger"""
        with self._lock:
            self._last_trigger_time = time.time()
            
            # Check if refill is complete
            if not self._refill_complete:
                logger.warning("Fire trigger received but refill in progress")
                self._publish_event('trigger_blocked_refilling')
                return
            
            # Always ensure main valve is open when fire detected
            if not GPIO.input(self.cfg['MAIN_VALVE_PIN']):
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
                self._schedule_timer('fire_off_monitor', self._check_fire_off, self.cfg['FIRE_OFF_DELAY'])
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
        
        # Cancel all timers
        for timer_name in list(self._timers.keys()):
            self._cancel_timer(timer_name)
        
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
        self._schedule_timer('cooldown_complete', self._enter_idle, self.cfg['COOLDOWN_DELAY'])
        
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
        for timer_name in list(self._timers.keys()):
            self._cancel_timer(timer_name)
        
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
            self._schedule_timer('start_engine', self._start_engine, self.cfg['PRE_OPEN_DELAY'])
            
            # Schedule fire-off monitor
            self._schedule_timer('fire_off_monitor', self._check_fire_off, self.cfg['FIRE_OFF_DELAY'])
    
    def _emergency_valve_open(self) -> bool:
        """Emergency valve opening procedures when normal methods fail"""
        logger.warning("Attempting emergency valve opening procedures")
        
        # Try direct GPIO manipulation with different timing
        pin = self.cfg['MAIN_VALVE_PIN']
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
            if not GPIO.input(self.cfg['MAIN_VALVE_PIN']):
                self._enter_error_state("Main valve not open - aborting engine start")
                return
            
            self._state = PumpState.STARTING
            self._publish_event('engine_start_sequence')
            
            # Start ignition sequence with retry logic
            if not self._set_pin('IGN_START', True):
                logger.error("Primary ignition start failed - attempting recovery")
                # Continue anyway - engine might start without perfect ignition signal
                self._publish_event('ignition_start_degraded')
            
            # Hold ignition start for configured duration
            time.sleep(self.cfg['IGNITION_START_DURATION'])
            
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
                self.cfg['PRIMING_DURATION']
            )
            
            # Schedule RPM reduction
            rpm_reduction_time = self.cfg['MAX_ENGINE_RUNTIME'] - self.cfg['RPM_REDUCTION_LEAD']
            if rpm_reduction_time > 0:
                self._schedule_timer('rpm_reduction', self._reduce_rpm, rpm_reduction_time)
            
            # Schedule max runtime shutdown
            self._schedule_timer('max_runtime', self._shutdown_engine, self.cfg['MAX_ENGINE_RUNTIME'])
    
    def _emergency_ignition_start(self) -> bool:
        """Emergency ignition procedures when normal start fails"""
        logger.warning("Attempting emergency ignition procedures")
        
        try:
            # Method 1: Extended cranking sequence
            for attempt in range(3):
                logger.info(f"Emergency ignition attempt {attempt + 1}")
                
                # Longer crank time
                self._set_pin('IGN_START', True, max_retries=1)
                time.sleep(self.cfg['IGNITION_START_DURATION'] * 2)
                self._set_pin('IGN_START', False, max_retries=1)
                
                time.sleep(0.5)  # Rest between attempts
                
                # Try to turn on ignition
                if self._set_pin('IGN_ON', True, max_retries=1):
                    # Verify ignition is actually on
                    time.sleep(0.2)
                    if GPIO.input(self.cfg['IGN_ON_PIN']):
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
            if self.cfg['LINE_PRESSURE_PIN']:
                self._schedule_timer(
                    'pressure_check',
                    self._check_line_pressure,
                    self.cfg['PRESSURE_CHECK_DELAY']
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
            
            if time_since_trigger >= self.cfg['FIRE_OFF_DELAY']:
                logger.info(f"No fire trigger for {time_since_trigger:.1f}s, initiating shutdown")
                # Cancel RPM reduction timer since we're shutting down due to fire off
                self._cancel_timer('rpm_reduction')
                self._shutdown_engine()
            else:
                # Re-schedule check
                remaining = self.cfg['FIRE_OFF_DELAY'] - time_since_trigger
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
            self._state = PumpState.STOPPING
            self._publish_event('shutdown_initiated', {'reason': previous_state.name})
            
            # Cancel max runtime timer if still active
            self._cancel_timer('max_runtime')
            self._cancel_timer('pressure_check')
            
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
                self.cfg['IGNITION_OFF_DURATION']
            )
            
            # Turn off engine
            self._set_pin('IGN_ON', False)
            self._set_pin('RPM_REDUCE', False)
            
            # Schedule valve closures
            self._schedule_timer(
                'close_main_valve',
                lambda: self._set_pin('MAIN_VALVE', False),
                self.cfg['VALVE_CLOSE_DELAY']
            )
            
            # Calculate refill time based on runtime
            refill_time = self._current_runtime * self.cfg['REFILL_MULTIPLIER']
            
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
            self._schedule_timer('cooldown_complete', self._enter_idle, 60)
    
    def _enter_idle(self):
        """Return to idle state"""
        with self._lock:
            self._state = PumpState.IDLE
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
            self._schedule_timer('fire_off_monitor', self._check_fire_off, self.cfg['FIRE_OFF_DELAY'])
            
            # Reschedule max runtime if needed
            if self._engine_start_time:
                elapsed = time.time() - self._engine_start_time
                remaining = self.cfg['MAX_ENGINE_RUNTIME'] - elapsed
                if remaining > 0:
                    self._schedule_timer('max_runtime', self._shutdown_engine, remaining)
            
            self._publish_event('shutdown_cancelled')
    
    def _enter_error_state(self, reason: str):
        """Enter error state requiring manual intervention"""
        with self._lock:
            logger.error(f"Entering ERROR state: {reason}")
            self._state = PumpState.ERROR
            
            # Ensure pump is off for safety
            self._set_pin('IGN_ON', False)
            self._set_pin('IGN_START', False)
            
            # Keep valves in current state
            
            self._publish_event('error_state_entered', {'reason': reason})
            
            # Cancel all timers
            self._cancel_all_timers()
            
            # Keep health monitoring active
            self._schedule_timer('health', self._publish_health, self.cfg['HEALTH_INTERVAL'])
    
    def _validate_hardware(self):
        """Validate hardware state and detect failures (optional)"""
        if not self.cfg['HARDWARE_VALIDATION_ENABLED']:
            return
        
        current_time = time.time()
        self._last_hardware_check = current_time
        
        try:
            # Check relay feedback pins if configured
            if self.cfg['RELAY_FEEDBACK_PINS']:
                for i, pin_str in enumerate(self.cfg['RELAY_FEEDBACK_PINS']):
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
            if not GPIO_AVAILABLE and self.cfg['SIMULATION_MODE_WARNINGS']:
                self._publish_event('simulation_mode_warning', {
                    'message': 'System running in simulation mode - no physical hardware control'
                })
            
        except Exception as e:
            logger.error(f"Hardware validation error: {e}")
            self._hardware_failures += 1
        
        # Reschedule next check
        self._schedule_timer('hardware_check', self._validate_hardware, self.cfg['HARDWARE_CHECK_INTERVAL'])
    
    def _get_expected_relay_state(self, relay_index: int) -> bool:
        """Get expected state for relay based on current system state"""
        # This is a simplified example - actual implementation would depend on hardware wiring
        if relay_index == 0:  # Assuming relay 0 is main valve
            return GPIO.input(self.cfg['MAIN_VALVE_PIN'])
        elif relay_index == 1:  # Assuming relay 1 is ignition
            return GPIO.input(self.cfg['IGN_ON_PIN'])
        return False  # Default to off for unknown relays
    
    def _monitor_dry_run_protection(self):
        """Monitor for pump running without water flow (dry run protection)"""
        while not self._shutdown:
            try:
                with self._lock:
                    current_time = time.time()
                    
                    # Check if pump is running
                    pump_running = (self._state == PumpState.RUNNING and 
                                  GPIO.input(self.cfg['IGN_ON_PIN']))
                    
                    if pump_running:
                        # Initialize pump start time if not set
                        if self._pump_start_time is None:
                            self._pump_start_time = current_time
                            self._water_flow_detected = False
                        
                        # Check for water flow if sensor is available
                        if self.cfg['FLOW_SENSOR_PIN']:
                            flow_detected = GPIO.input(self.cfg['FLOW_SENSOR_PIN'])
                            if flow_detected:
                                self._water_flow_detected = True
                        else:
                            # If no flow sensor, check line pressure as proxy
                            if self.cfg['LINE_PRESSURE_PIN']:
                                pressure_ok = self._check_line_pressure()
                                if pressure_ok:
                                    self._water_flow_detected = True
                            else:
                                # No sensors available - assume water flow after priming period
                                if current_time - self._pump_start_time > self.cfg['PRIMING_DURATION']:
                                    self._water_flow_detected = True
                        
                        # Check dry run time limit
                        dry_run_time = current_time - self._pump_start_time
                        
                        if dry_run_time > self.cfg['MAX_DRY_RUN_TIME'] and not self._water_flow_detected:
                            logger.critical(f"DRY RUN PROTECTION: Pump running {dry_run_time:.1f}s without water flow!")
                            self._publish_event('dry_run_protection_triggered', {
                                'dry_run_time': dry_run_time,
                                'max_allowed': self.cfg['MAX_DRY_RUN_TIME']
                            })
                            self._enter_error_state(f"Dry run protection: {dry_run_time:.1f}s without water flow")
                            break
                        elif dry_run_time > self.cfg['MAX_DRY_RUN_TIME'] * 0.8 and not self._water_flow_detected:
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
                if self.cfg['EMERGENCY_BUTTON_PIN']:
                    current_state = GPIO.input(self.cfg['EMERGENCY_BUTTON_PIN'])
                    active_state = not current_state if self.cfg['EMERGENCY_BUTTON_ACTIVE_LOW'] else current_state
                    
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
            if self.cfg['ENHANCED_STATUS_ENABLED']:
                # Hardware status
                health_data['hardware'] = {
                    'gpio_available': GPIO_AVAILABLE,
                    'simulation_mode': not GPIO_AVAILABLE,
                    'validation_enabled': self.cfg['HARDWARE_VALIDATION_ENABLED'],
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
                    'max_dry_run_time': self.cfg['MAX_DRY_RUN_TIME'],
                }
                if self._pump_start_time:
                    health_data['dry_run_protection']['current_runtime'] = time.time() - self._pump_start_time
                
                # Safety feature status
                health_data['safety_features'] = {
                    'emergency_button_available': self.cfg['EMERGENCY_BUTTON_PIN'] is not None,
                    'flow_sensor_available': self.cfg['FLOW_SENSOR_PIN'] is not None,
                    'reservoir_sensor_available': self.cfg['RESERVOIR_FLOAT_PIN'] is not None,
                    'pressure_sensor_available': self.cfg['LINE_PRESSURE_PIN'] is not None,
                }
                
                # Critical warnings for simulation mode
                if not GPIO_AVAILABLE and self.cfg['SIMULATION_MODE_WARNINGS']:
                    health_data['critical_warnings'] = [
                        'SIMULATION_MODE_ACTIVE',
                        'NO_PHYSICAL_HARDWARE_CONTROL',
                        'PUMP_WILL_NOT_OPERATE_IN_EMERGENCY'
                    ]
            
            # Add sensor states if available
            if self.cfg['RESERVOIR_FLOAT_PIN']:
                health_data['reservoir_full'] = self._is_reservoir_full()
            
            if self.cfg['LINE_PRESSURE_PIN']:
                health_data['line_pressure_ok'] = self._is_line_pressure_ok()
            
            # Include pin states for diagnostics
            health_data['pin_states'] = self._get_state_snapshot()
            
            self._publish_event('health_report', health_data)
            
            # Reschedule
            self._schedule_timer('health', self._publish_health, self.cfg['HEALTH_INTERVAL'])
    
    def cleanup(self):
        """Clean shutdown of controller"""
        logger.info("Cleaning up PumpController")
        
        # Set shutdown flag to stop monitoring threads
        self._shutdown = True
        
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
        controller = PumpController()
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
