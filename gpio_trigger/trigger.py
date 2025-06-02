#!/usr/bin/env python3
"""
PumpController for wildfire-watch:
- Fail-safe, thread-safe engine and valve control
- State machine approach for consistent operation
- Comprehensive error recovery and safety checks
- Idempotent operations resilient to concurrent events
"""
import os
import time
import json
import socket
import threading
import logging
from datetime import datetime
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
    'TELEMETRY_TOPIC': os.getenv('TELEMETRY_TOPIC', 'system/trigger_telemetry'),
    
    # GPIO Pins
    'MAIN_VALVE_PIN': int(os.getenv('MAIN_VALVE_PIN', '18')), # Turns main water valve on
    'IGN_START_PIN': int(os.getenv('IGNITION_START_PIN', '23')), # Pulsed for engine start
    'IGN_ON_PIN': int(os.getenv('IGNITION_ON_PIN', '24')), # On for entire time engine is on
    'IGN_OFF_PIN': int(os.getenv('IGNITION_OFF_PIN', '25')), # Pulsed before engine off
    'REFILL_VALVE_PIN': int(os.getenv('REFILL_VALVE_PIN', '22')), # Open refill valve / run refill pump
    'PRIMING_VALVE_PIN': int(os.getenv('PRIMING_VALVE_PIN', '26')), # Opens priming valve to bleed any air in the pump
    'RPM_REDUCE_PIN': int(os.getenv('RPM_REDUCE_PIN', '27')), # Reduce engine RPM
    
    # Timing Configuration
    'PRE_OPEN_DELAY': float(os.getenv('VALVE_PRE_OPEN_DELAY', '2')), # Open main valve X seconds before ignition start
    'IGNITION_START_DURATION': float(os.getenv('IGNITION_START_DURATION', '5')), # Hold ignition start for X seconds
    'FIRE_OFF_DELAY': float(os.getenv('FIRE_OFF_DELAY', '1800')), # Keep pump running X seconds past last message
    'VALVE_CLOSE_DELAY': float(os.getenv('VALVE_CLOSE_DELAY', '600')), # Close main valve X seconds after engine stop
    'IGNITION_OFF_DURATION': float(os.getenv('IGNITION_OFF_DURATION', '5')), # Pulse X seconds before engine off
    'MAX_ENGINE_RUNTIME': float(os.getenv('MAX_ENGINE_RUNTIME', '600')), # Maximum runtime of fire pump in seconds (initially set low for testing and pump safety)
    'REFILL_MULTIPLIER': float(os.getenv('REFILL_MULTIPLIER', '40')), # Refill water tank for a multiple of run-time
    'PRIMING_DURATION': float(os.getenv('PRIMING_DURATION', '180')), # Bleed off any air in the pump for this long
    'RPM_REDUCTION_LEAD': float(os.getenv('RPM_REDUCTION_LEAD', '15')), # Pulse RPM reduction pin X seconds before shutdown
    'HEALTH_INTERVAL': float(os.getenv('TELEMETRY_INTERVAL', '60')),
    'ACTION_RETRY_INTERVAL': float(os.getenv('ACTION_RETRY_INTERVAL', '60')),
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
    logger.warning("RPi.GPIO unavailable; using simulation mode")
    
    class GPIO:
        BCM = "BCM"
        OUT = "OUT"
        HIGH = True
        LOW = False
        _state = {}
        
        @classmethod
        def setmode(cls, mode):
            pass
        
        @classmethod
        def setwarnings(cls, warnings):
            pass
        
        @classmethod
        def setup(cls, pin, mode, initial=None):
            cls._state[pin] = initial if initial is not None else cls.LOW
        
        @classmethod
        def output(cls, pin, value):
            cls._state[pin] = bool(value)
        
        @classmethod
        def input(cls, pin):
            return cls._state.get(pin, cls.LOW)
        
        @classmethod
        def cleanup(cls):
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
    ERROR = auto()         # Error state requiring manual intervention

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
        
        # Initialize GPIO
        self._init_gpio()
        
        # Setup MQTT
        self._setup_mqtt()
        
        # Start health monitoring
        self._schedule_timer('health', self._publish_health, self.cfg['HEALTH_INTERVAL'])
        
        logger.info(f"PumpController initialized in {self._state.name} state")
    
    def _init_gpio(self):
        """Initialize all GPIO pins to safe state"""
        pins = {
            'MAIN_VALVE_PIN': GPIO.LOW,
            'IGN_START_PIN': GPIO.LOW,
            'IGN_ON_PIN': GPIO.LOW,
            'IGN_OFF_PIN': GPIO.LOW,
            'REFILL_VALVE_PIN': GPIO.LOW,
            'PRIMING_VALVE_PIN': GPIO.LOW,
            'RPM_REDUCE_PIN': GPIO.LOW,
        }
        
        for pin_name, initial_state in pins.items():
            pin = self.cfg[pin_name]
            GPIO.setup(pin, GPIO.OUT, initial=initial_state)
            logger.debug(f"Initialized {pin_name} (pin {pin}) to {initial_state}")
    
    def _setup_mqtt(self):
        """Setup MQTT client with TLS if configured"""
        self.client = mqtt.Client(clean_session=True)
        
        # Set LWT
        lwt_topic = f"{self.cfg['TELEMETRY_TOPIC']}/{socket.gethostname()}/lwt"
        lwt_payload = json.dumps({
            'host': socket.gethostname(),
            'status': 'offline',
            'timestamp': self._now_iso()
        })
        self.client.will_set(lwt_topic, payload=lwt_payload, qos=1, retain=True)
        
        # Configure TLS if enabled
        if self.cfg['MQTT_TLS']:
            import ssl
            self.client.tls_set(
                ca_certs=self.cfg['TLS_CA_PATH'],
                cert_reqs=ssl.CERT_REQUIRED,
                tls_version=ssl.PROTOCOL_TLS
            )
            logger.info("MQTT TLS enabled")
        
        # Set callbacks
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        
        # Connect
        try:
            port = 8883 if self.cfg['MQTT_TLS'] else 1883
            self.client.connect(self.cfg['MQTT_BROKER'], port, keepalive=60)
            self.client.loop_start()
            logger.info(f"MQTT client connected to {self.cfg['MQTT_BROKER']}:{port}")
        except Exception as e:
            logger.error(f"MQTT connection failed: {e}")
            self._state = PumpState.ERROR
    
    def _now_iso(self) -> str:
        """Get current UTC timestamp in ISO format"""
        return datetime.utcnow().isoformat() + "Z"
    
    def _get_state_snapshot(self) -> Dict[str, Any]:
        """Get current state of all pins and system"""
        with self._lock:
            return {
                'state': self._state.name,
                'engine_on': GPIO.input(self.cfg['IGN_ON_PIN']),
                'main_valve': GPIO.input(self.cfg['MAIN_VALVE_PIN']),
                'refill_valve': GPIO.input(self.cfg['REFILL_VALVE_PIN']),
                'priming_valve': GPIO.input(self.cfg['PRIMING_VALVE_PIN']),
                'rpm_reduced': GPIO.input(self.cfg['RPM_REDUCE_PIN']),
                'total_runtime': self._total_runtime,
                'shutting_down': self._shutting_down,
                'active_timers': list(self._timers.keys()),
            }
    
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
    
    def _set_pin(self, pin_name: str, state: bool) -> bool:
        """Set GPIO pin state with error handling"""
        pin = self.cfg[f'{pin_name}_PIN']
        try:
            GPIO.output(pin, GPIO.HIGH if state else GPIO.LOW)
            logger.debug(f"Set {pin_name} (pin {pin}) to {'HIGH' if state else 'LOW'}")
            return True
        except Exception as e:
            logger.error(f"Failed to set {pin_name}: {e}")
            self._publish_event('gpio_error', {'pin': pin_name, 'error': str(e)})
            return False
    
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
    
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            client.subscribe([(self.cfg['TRIGGER_TOPIC'], 0)])
            logger.info(f"Subscribed to {self.cfg['TRIGGER_TOPIC']}")
            self._publish_event('mqtt_connected')
        else:
            logger.error(f"MQTT connection failed with code {rc}")
            self._state = PumpState.ERROR
    
    def _on_disconnect(self, client, userdata, rc):
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
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    def handle_fire_trigger(self):
        """Handle fire detection trigger"""
        with self._lock:
            self._last_trigger_time = time.time()
            
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
            elif self._state == PumpState.ERROR:
                logger.error("System in ERROR state - manual intervention required")
                self._publish_event('error_state_trigger_ignored')
    
    def _start_pump_sequence(self):
        """Start the pump startup sequence"""
        with self._lock:
            if self._state != PumpState.IDLE:
                logger.warning(f"Cannot start pump from {self._state.name} state")
                return
            
            self._state = PumpState.PRIMING
            self._publish_event('pump_sequence_start')
            
            # Open main valve first (fail-safe)
            if not self._set_pin('MAIN_VALVE', True):
                self._enter_error_state("Failed to open main valve")
                return
            
            # Start priming
            if not self._set_pin('PRIMING_VALVE', True):
                self._enter_error_state("Failed to open priming valve")
                return
            
            self._publish_event('priming_started')
            
            # Schedule engine start after pre-open delay
            self._schedule_timer('start_engine', self._start_engine, self.cfg['PRE_OPEN_DELAY'])
            
            # Schedule fire-off monitor
            self._schedule_timer('fire_off_monitor', self._check_fire_off, self.cfg['FIRE_OFF_DELAY'])
    
    def _start_engine(self):
        """Start the engine with safety checks"""
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
            
            # Start ignition sequence
            if not self._set_pin('IGN_START', True):
                self._enter_error_state("Failed to activate ignition start")
                return
            
            # Hold ignition start for configured duration
            time.sleep(self.cfg['IGNITION_START_DURATION'])
            
            # Release ignition start
            self._set_pin('IGN_START', False)
            
            # Turn on engine
            if not self._set_pin('IGN_ON', True):
                self._enter_error_state("Failed to turn on ignition")
                return
            
            # Engine is now running
            self._state = PumpState.RUNNING
            self._engine_start_time = time.time()
            self._publish_event('engine_running')
            
            # Start refill valve
            if not self._set_pin('REFILL_VALVE', True):
                logger.warning("Failed to open refill valve")
            
            # Schedule priming valve close
            self._schedule_timer(
                'close_priming',
                lambda: self._set_pin('PRIMING_VALVE', False),
                self.cfg['PRIMING_DURATION']
            )
            
            # Schedule RPM reduction
            rpm_reduction_time = self.cfg['MAX_ENGINE_RUNTIME'] - self.cfg['RPM_REDUCTION_LEAD']
            if rpm_reduction_time > 0:
                self._schedule_timer('rpm_reduction', self._reduce_rpm, rpm_reduction_time)
            
            # Schedule max runtime shutdown
            self._schedule_timer('max_runtime', self._shutdown_engine, self.cfg['MAX_ENGINE_RUNTIME'])
    
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
                self._shutdown_engine()
            else:
                # Re-schedule check
                remaining = self.cfg['FIRE_OFF_DELAY'] - time_since_trigger
                self._schedule_timer('fire_off_monitor', self._check_fire_off, remaining)
    
    def _shutdown_engine(self):
        """Shutdown engine with proper sequence"""
        with self._lock:
            if self._state not in [PumpState.RUNNING, PumpState.REDUCING_RPM]:
                logger.warning(f"Cannot shutdown from {self._state.name} state")
                return
            
            if self._shutting_down:
                logger.warning("Shutdown already in progress")
                return
            
            self._shutting_down = True
            self._state = PumpState.STOPPING
            self._publish_event('shutdown_initiated')
            
            # Cancel max runtime timer if still active
            self._cancel_timer('max_runtime')
            
            # Calculate total runtime
            if self._engine_start_time:
                runtime = time.time() - self._engine_start_time
                self._total_runtime += runtime
            else:
                runtime = 0
            
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
            refill_time = runtime * self.cfg['REFILL_MULTIPLIER']
            self._schedule_timer(
                'close_refill_valve',
                lambda: self._set_pin('REFILL_VALVE', False),
                refill_time
            )
            
            # Enter cooldown state
            self._schedule_timer(
                'enter_cooldown',
                self._enter_cooldown,
                max(self.cfg['VALVE_CLOSE_DELAY'], refill_time) + 5
            )
            
            self._publish_event('shutdown_complete', {'runtime': runtime})
    
    def _enter_cooldown(self):
        """Enter cooldown state after shutdown"""
        with self._lock:
            self._state = PumpState.COOLDOWN
            self._shutting_down = False
            self._engine_start_time = None
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
    
    def _publish_health(self):
        """Publish periodic health status"""
        with self._lock:
            health_data = {
                'uptime': time.time(),
                'total_runtime': self._total_runtime,
                'state': self._state.name,
                'last_trigger': self._last_trigger_time,
            }
            
            self._publish_event('health_report', health_data)
            
            # Reschedule
            self._schedule_timer('health', self._publish_health, self.cfg['HEALTH_INTERVAL'])
    
    def cleanup(self):
        """Clean shutdown of controller"""
        logger.info("Cleaning up PumpController")
        
        with self._lock:
            # Cancel all timers
            self._cancel_all_timers()
            
            # Ensure pump is off
            if self._state in [PumpState.RUNNING, PumpState.REDUCING_RPM]:
                self._shutdown_engine()
                time.sleep(5)  # Allow shutdown to complete
            
            # Close all valves
            for pin_name in ['MAIN_VALVE', 'REFILL_VALVE', 'PRIMING_VALVE']:
                self._set_pin(pin_name, False)
            
            # Turn off all control pins
            for pin_name in ['IGN_START', 'IGN_ON', 'IGN_OFF', 'RPM_REDUCE']:
                self._set_pin(pin_name, False)
        
        # Disconnect MQTT
        self._publish_event('controller_shutdown')
        self.client.loop_stop()
        self.client.disconnect()
        
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
