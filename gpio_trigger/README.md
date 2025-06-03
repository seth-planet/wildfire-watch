# 🚿 GPIO Trigger Service (Pump Controller)

## What Does This Do?

The GPIO Trigger Service controls your water pump and sprinkler system. It's the "muscle" of your wildfire defense that:

- 💧 **Controls water valves** - Opens/closes water flow
- 🔥 **Starts fire pump** - Activates gas/diesel pump engine  
- ⏱️ **Manages timing** - Runs pump for appropriate duration
- 🔄 **Handles refill** - Automatically refills water reservoir
- 🛡️ **Prevents damage** - Protects pump from running dry
- 📊 **Monitors safety** - Checks reservoir level and line pressure

## Why This Matters

This service ensures your sprinkler system activates reliably when fire is detected, while preventing costly damage from:
- Running pumps without water
- Engine damage from improper shutdown
- Reservoir depletion or overflow
- Dry pump operation from leaks
- False activations during refill

## Safety First! ⚠️

**IMPORTANT**: This service controls physical equipment that can cause injury or damage. Before using:

1. **Test without engine** - Disconnect ignition wires first
2. **Verify valve operation** - Ensure valves open/close properly
3. **Check water levels** - Never run pump dry
4. **Install emergency stop** - Have manual override available
5. **Follow local codes** - Comply with fire safety regulations

## Quick Start

### Default Operation

When fire is detected, the system:
1. Opens main water valve
2. Opens priming valve (bleeds air)
3. Starts pump engine
4. **Opens refill valve immediately** (ensures reservoir refilling)
5. Runs with priming valve open for 3 minutes
6. Closes priming valve for full pressure
7. Monitors for continued fire
8. Checks line pressure (if sensor installed)
9. Shuts down safely when fire is gone
10. Continues refilling for (runtime × multiplier)
11. Monitors reservoir level (if sensor installed)

### What You'll See

Normal operation logs:
```
[INFO] Event: pump_sequence_start
[INFO] Event: valve_opened, State: {'MAIN_VALVE': True}
[INFO] Event: priming_started, State: {'PRIMING_VALVE': True}
[INFO] Event: engine_start_sequence
[INFO] Event: engine_running
[INFO] Event: refill_valve_opened_on_start, State: {'REFILL_VALVE': True}
[INFO] Event: priming_complete, State: {'PRIMING_VALVE': False}
[INFO] Event: shutdown_initiated
[INFO] Event: refill_continuing, duration: 12000 seconds
[INFO] Event: refill_complete_float_switch (or refill_complete_timer)
```

## Hardware Connections

### GPIO Pin Assignments

| Function | Default Pin | Wire Color (Suggested) | Description |
|----------|------------|------------------------|-------------|
| **Control Outputs** |
| Main Valve | GPIO 18 | Blue | Opens main water valve to sprinklers |
| Ignition Start | GPIO 23 | Yellow | Pulse to start engine (like turning key) |
| Ignition On | GPIO 24 | Green | Stays on while engine runs |
| Ignition Off | GPIO 25 | Red | Pulse to stop engine |
| Refill Valve | GPIO 22 | Purple | Opens valve to refill reservoir |
| Priming Valve | GPIO 26 | Orange | Bleeds air from pump |
| RPM Reduce | GPIO 27 | Brown | Reduces engine speed before shutdown |
| **Monitoring Inputs** |
| Reservoir Float | GPIO 16 | White | Tank level sensor (optional) |
| Line Pressure | GPIO 20 | Gray | Pressure switch (optional) |

### Safety Sensor Wiring

**Reservoir Float Switch**:
- Normally Open (NO) switch recommended
- Connect between GPIO 16 and Ground
- Closes when tank is full
- Set `RESERVOIR_FLOAT_ACTIVE_LOW=true` for NO switches

**Line Pressure Switch**:
- Normally Closed (NC) switch recommended
- Connect between GPIO 20 and 3.3V
- Opens when pressure is low
- Set `LINE_PRESSURE_ACTIVE_LOW=true` for NC switches

### Wiring Diagram Example

```
Raspberry Pi                    Relay Board
GPIO 18 -----------------> Relay 1 -> Main Valve Solenoid
GPIO 23 -----------------> Relay 2 -> Ignition Start Circuit
GPIO 24 -----------------> Relay 3 -> Ignition Run Circuit
GPIO 25 -----------------> Relay 4 -> Ignition Stop Circuit
GPIO 22 -----------------> Relay 5 -> Refill Valve
GPIO 26 -----------------> Relay 6 -> Priming Valve
GPIO 27 -----------------> Relay 7 -> RPM Reduce Circuit

GPIO 16 <-- Float Switch <-- GND (when full)
GPIO 20 <-- Pressure Switch <-- 3.3V (when OK)
```

**💡 Tip**: Use a relay board rated for your valve/ignition voltage (typically 12V or 24V)

## Configuration Options

### Timing Configuration

```bash
# Valve and startup timings
VALVE_PRE_OPEN_DELAY=2      # Open valve 2 seconds before starting
IGNITION_START_DURATION=5   # Hold start for 5 seconds
PRIMING_DURATION=180        # Run with bleed valve open for 3 minutes

# Shutdown timings  
FIRE_OFF_DELAY=1800         # Run 30 minutes after fire gone
VALVE_CLOSE_DELAY=600       # Close valve 10 min after engine stops
IGNITION_OFF_DURATION=5     # Pulse stop for 5 seconds
RPM_REDUCTION_LEAD=15       # Reduce RPM 15 seconds before stop

# Safety limits
MAX_ENGINE_RUNTIME=1800     # Maximum 30 minute run time (adjust for your tank!)
REFILL_MULTIPLIER=40        # Refill 40x longer than runtime
PRESSURE_CHECK_DELAY=60     # Check pressure 1 minute after priming
```

**⚠️ IMPORTANT**: The `MAX_ENGINE_RUNTIME` default of 30 minutes is intentionally conservative. Calculate your actual limit:
- Tank capacity (gallons) ÷ Pump flow rate (gpm) = Maximum runtime (minutes)
- Example: 10,000 gallons ÷ 100 gpm = 100 minutes maximum
- Set this value LOWER than your calculation to prevent pump damage!

### Pin Customization

```bash
# Control pins (required)
MAIN_VALVE_PIN=18
IGNITION_START_PIN=23
IGNITION_ON_PIN=24
IGNITION_OFF_PIN=25
REFILL_VALVE_PIN=22
PRIMING_VALVE_PIN=26
RPM_REDUCE_PIN=27

# Monitoring pins (optional - leave blank if not used)
RESERVOIR_FLOAT_PIN=16      # Tank level sensor
LINE_PRESSURE_PIN=20        # Pressure switch

# Sensor configuration
RESERVOIR_FLOAT_ACTIVE_LOW=true   # true = NO switch, false = NC switch
LINE_PRESSURE_ACTIVE_LOW=true     # true = NC switch, false = NO switch
```

### System Settings

```bash
# Health monitoring
TELEMETRY_INTERVAL=60       # Report status every minute
ACTION_RETRY_INTERVAL=60    # Retry failed actions

# Logging
LOG_LEVEL=INFO              # Set to DEBUG for troubleshooting
```

## Understanding the Pump Sequence

### Startup Sequence

1. **Pre-Open Valve** (2 sec)
   - Main valve opens before engine start
   - Prevents pressure damage

2. **Priming Valve Opens** (immediate)
   - Opens air bleed valve
   - Prepares to remove air from pump

3. **Engine Start** (5 sec pulse)
   - Ignition start pulse
   - Like turning a key

4. **Engine Running**
   - Engine starts and runs
   - **Refill valve opens immediately**
   - Ensures reservoir stays topped off

5. **Running with Priming** (3 min)
   - Engine runs with priming valve OPEN
   - Air bleeds out continuously
   - Pump fills with water
   - Refill valve remains open
   
6. **Priming Complete**
   - Priming valve closes after 3 minutes
   - Pump now at full pressure/flow
   - Line pressure check starts (if sensor installed)
   - Refill valve stays open

7. **Pressure Monitoring** (optional)
   - 1 minute after priming complete
   - Shuts down if low pressure detected
   - Prevents damage from leaks or dry pump

### Shutdown Sequence

1. **RPM Reduction** (15 sec before)
   - Reduces engine speed
   - Gentler shutdown

2. **Engine Stop** (5 sec pulse)
   - Ignition off signal
   - Engine stops

3. **Valve Timing**
   - Main valve stays open 10 min
   - Allows pressure relief

4. **Refill Process**
   - Refill valve remains open
   - Continues for runtime × multiplier
   - Stops early if float switch triggers
   - Pump cannot restart during refill

### Safety Features

1. **Refill Lockout**
   - Pump will NOT start during refill
   - Prevents dry running
   - Must complete refill first

2. **Reservoir Monitoring**
   - Optional float switch input
   - Stops refill when tank full
   - Prevents overflow

3. **Line Pressure Monitoring**
   - Optional pressure switch input
   - Detects pump problems
   - Shuts down if pressure low

### Pump Operation Timeline

```
Time    Main    Priming    Engine    Refill    Pressure    Action
----    ----    -------    ------    ------    --------    ------
0s      CLOSED  CLOSED     OFF       CLOSED    N/A         Fire detected
2s      OPEN    CLOSED     OFF       CLOSED    N/A         Main valve opens
2s      OPEN    OPEN       OFF       CLOSED    N/A         Priming valve opens
4s      OPEN    OPEN       START     CLOSED    N/A         Engine cranking
9s      OPEN    OPEN       RUN       OPEN      N/A         Engine running, refill starts
...     OPEN    OPEN       RUN       OPEN      N/A         Priming continues
180s    OPEN    CLOSED     RUN       OPEN      N/A         Priming complete
240s    OPEN    CLOSED     RUN       OPEN      CHECK       Pressure check
...     OPEN    CLOSED     RUN       OPEN      OK          Normal pumping
1800s   OPEN    CLOSED     STOP      OPEN      OK          Fire off, shutdown
1810s   OPEN    CLOSED     OFF       OPEN      N/A         Engine stopped
2410s   CLOSED  CLOSED     OFF       OPEN      N/A         Main valve closed
...     CLOSED  CLOSED     OFF       OPEN      N/A         Refill continues
73800s  CLOSED  CLOSED     OFF       CLOSED    N/A         Refill complete (or float triggered)
```

**Critical Points**:
- Refill valve opens immediately when engine starts
- Refill continues throughout entire runtime
- Refill extends well beyond engine shutdown
- Float switch can terminate refill early

## Common Issues and Solutions

### Problem: Engine Won't Start

**Symptoms**: Ignition clicks but engine doesn't run

**Solutions**:
1. **Check fuel**: Ensure tank has gas/diesel
2. **Check battery**: Engine battery may be dead
3. **Increase start time**:
   ```bash
   IGNITION_START_DURATION=10
   ```
4. **Verify wiring**: Test relays with multimeter
5. **Manual test**: Try starting engine manually

### Problem: Pump Runs Dry

**Symptoms**: Pump makes grinding noise, no water flow

**Solutions**:
1. **Check water level**: Refill reservoir manually
2. **Increase refill time**:
   ```bash
   REFILL_MULTIPLIER=60
   ```
3. **Check refill valve**: May be clogged or stuck
4. **Add float switch**: Wire to GPIO 16 for automatic detection
5. **Verify runtime limit**: Ensure MAX_ENGINE_RUNTIME matches your tank

### Problem: Engine Won't Stop

**Symptoms**: Engine continues running after trigger

**Solutions**:
1. **Emergency stop**: Use manual kill switch
2. **Check GPIO 25**: Ignition off signal
3. **Increase stop duration**:
   ```bash
   IGNITION_OFF_DURATION=10
   ```
4. **Check max runtime**:
   ```bash
   MAX_ENGINE_RUNTIME=1800  # 30 minutes max
   ```

### Problem: Valves Not Operating

**Symptoms**: No click from valve solenoids

**Solutions**:
1. **Check relay board**: LEDs should light when activated
2. **Test manually**: 
   ```python
   import RPi.GPIO as GPIO
   GPIO.setmode(GPIO.BCM)
   GPIO.setup(18, GPIO.OUT)
   GPIO.output(18, GPIO.HIGH)  # Test valve
   ```
3. **Check power**: Solenoids need 12V/24V supply
4. **Verify wiring**: Common ground between Pi and relays

### Problem: Tank Overflows During Refill

**Symptoms**: Water spills from reservoir

**Solutions**:
1. **Install float switch**: Connect to GPIO 16
2. **Reduce refill multiplier**:
   ```bash
   REFILL_MULTIPLIER=30
   ```
3. **Calculate proper timing**: 
   - Measure actual pump runtime
   - Calculate refill rate
   - Set multiplier accordingly

### Problem: Low Pressure Shutdown

**Symptoms**: Pump shuts down after priming

**Solutions**:
1. **Check for leaks**: Inspect all connections
2. **Verify pump prime**: May need longer priming
3. **Adjust pressure switch**: May be too sensitive
4. **Increase check delay**:
   ```bash
   PRESSURE_CHECK_DELAY=120  # 2 minutes
   ```

## Testing and Validation

### Safe Testing Procedure

1. **Disconnect ignition** - Remove engine start wires
2. **Test valve sequence**:
   ```bash
   # Publish test trigger
   mosquitto_pub -t "fire/trigger" -m "{}"
   ```
3. **Verify timing** - Watch LED indicators
4. **Check logs** - Ensure proper sequence
5. **Test safety sensors**:
   - Trigger float switch during refill
   - Trigger pressure switch during run
6. **Reconnect gradually** - Test each component

### GPIO Testing Script

```python
#!/usr/bin/env python3
import RPi.GPIO as GPIO
import time

# Test each pin
pins = {
    18: "Main Valve",
    23: "Ignition Start", 
    24: "Ignition On",
    25: "Ignition Off",
    22: "Refill Valve",
    26: "Priming Valve",
    27: "RPM Reduce",
    16: "Float Switch (input)",
    20: "Pressure Switch (input)"
}

GPIO.setmode(GPIO.BCM)

# Setup outputs
for pin in [18, 23, 24, 25, 22, 26, 27]:
    GPIO.setup(pin, GPIO.OUT)

# Setup inputs
GPIO.setup(16, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(20, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

# Test each output
for pin in [18, 23, 24, 25, 22, 26, 27]:
    print(f"Testing {pins[pin]} on GPIO {pin}")
    GPIO.output(pin, GPIO.HIGH)
    time.sleep(2)
    GPIO.output(pin, GPIO.LOW)
    time.sleep(1)

# Read inputs
print(f"\nFloat Switch (GPIO 16): {'FULL' if GPIO.input(16) else 'NOT FULL'}")
print(f"Pressure Switch (GPIO 20): {'OK' if GPIO.input(20) else 'LOW'}")

GPIO.cleanup()
```

## State Machine Explained

The controller uses a state machine for safety:

```
IDLE → PRIMING → STARTING → RUNNING → REDUCING_RPM → STOPPING → REFILLING → COOLDOWN → IDLE
         ↓          ↓          ↓           ↓            ↓           ↓
        ERROR ←──-──┴──────────┴───────────┴────────────┴───────────┘
                               ↓
                         LOW_PRESSURE
```

**States**:
- **IDLE**: Ready, waiting for fire
- **PRIMING**: Valves open, preparing to start engine
- **STARTING**: Engine cranking/starting up
- **RUNNING**: Engine running, pump operating, refill active
- **REDUCING_RPM**: Slowing engine before stop
- **STOPPING**: Shutting down engine
- **REFILLING**: Continuing refill after shutdown (pump cannot start)
- **COOLDOWN**: Post-shutdown wait period
- **LOW_PRESSURE**: Low pressure detected, shutting down
- **ERROR**: Manual intervention needed

**Key Safety Features**:
- Cannot start during refill
- Refill valve opens with engine start
- Pressure monitoring after priming
- Float switch stops refill early

## Safety Features

### Automatic Protections

1. **Valve-First Start**: Valve must be open before engine
2. **Maximum Runtime**: Prevents tank depletion
3. **Refill Lockout**: No starts during refill
4. **Overflow Prevention**: Float switch stops refill
5. **Dry Pump Protection**: Pressure monitoring
6. **RPM Reduction**: Gentle shutdown sequence
7. **Cooldown Period**: Prevents rapid cycling
8. **State Validation**: Can't start from invalid state

### Manual Overrides

Always install these physical controls:
1. **Emergency Stop Button** - Kills engine immediately
2. **Manual Valve Control** - Bypass GPIO control
3. **Fuel Shutoff** - Stops engine without electrical
4. **Main Power Switch** - Disconnects all systems
5. **Refill Bypass** - Manual refill control

## Monitoring and Debugging

### View System Status
```bash
docker logs gpio_trigger
```

### Monitor MQTT Events
```bash
mosquitto_sub -t "system/trigger_telemetry" -v
```

### Enable Debug Mode
```bash
LOG_LEVEL=DEBUG
```

### Common Log Messages

**Normal Operation**:
```
[INFO] MQTT connected successfully
[INFO] Event: pump_sequence_start
[INFO] Event: priming_started
[INFO] Event: engine_running
[INFO] Event: refill_valve_opened_on_start
[INFO] Event: priming_complete
[INFO] Event: shutdown_complete, runtime: 300
[INFO] Event: refill_continuing, duration: 12000
[INFO] Event: refill_complete_float_switch
```

**Safety Events**:
```
[WARNING] Fire trigger received but refill in progress
[INFO] Reservoir full detected, stopping refill
[ERROR] Low line pressure detected!
[INFO] Event: low_pressure_detected
```

**Issues**:
```
[ERROR] Main valve not open - aborting engine start
[WARNING] Cannot start pump from REFILLING state
[ERROR] Failed to set MAIN_VALVE: [Errno 13] Permission denied
```

## Advanced Customization

### Tank Size Calculation

```python
# Calculate your settings
tank_gallons = 10000
pump_gpm = 100
refill_gpm = 25

max_runtime_minutes = tank_gallons / pump_gpm  # 100 minutes
safe_runtime = max_runtime_minutes * 0.8       # 80 minutes (20% safety margin)
refill_multiplier = pump_gpm / refill_gpm      # 4x

print(f"Set MAX_ENGINE_RUNTIME={int(safe_runtime * 60)}")  # 4800 seconds
print(f"Set REFILL_MULTIPLIER={refill_multiplier}")        # 4
```

### Multiple Pump Support

```python
# Define additional pumps
PUMP2_VALVE_PIN = 19
PUMP2_START_PIN = 20

# Modify startup to handle multiple pumps
def _start_all_pumps(self):
    self._start_engine()  # Pump 1
    self._start_pump_2()  # Pump 2
```

### Advanced Sensor Integration

```python
# Add pressure transducer instead of switch
import spidev

def read_pressure_psi(self):
    spi = spidev.SpiDev()
    spi.open(0, 0)
    raw = spi.readbytes(2)
    voltage = (raw[0] << 8 | raw[1]) * 3.3 / 1024
    psi = (voltage - 0.5) * 100 / 4.0  # 0.5-4.5V = 0-100 PSI
    return psi
```

### SMS/Email Alerts

```python
# Add notification on events
def _notify_event(self, event_type):
    if event_type in ['low_pressure_detected', 'error_state_entered']:
        message = f"ALERT: {event_type} at {datetime.now()}"
        send_sms(message)
        send_email(message)
```

## Integration Examples

### Home Assistant

```yaml
sensor:
  - platform: mqtt
    name: "Pump State"
    state_topic: "system/trigger_telemetry"
    value_template: "{{ value_json.system_state.state }}"
    
  - platform: mqtt
    name: "Reservoir Level"
    state_topic: "system/trigger_telemetry"
    value_template: "{{ 'Full' if value_json.system_state.reservoir_full else 'Not Full' }}"

switch:
  - platform: mqtt
    name: "Fire Pump Trigger"
    command_topic: "fire/trigger"
    payload_on: "{}"
    payload_off: "stop"
    state_topic: "system/trigger_telemetry"
    value_template: "{{ value_json.system_state.engine_on }}"
```

### Node-RED Flow

```json
[{
  "type": "mqtt in",
  "topic": "system/trigger_telemetry",
  "name": "Pump Status"
},{
  "type": "function",
  "func": "msg.payload = JSON.parse(msg.payload);\nif(msg.payload.action === 'low_pressure_detected'){\n  msg.alert = true;\n}\nreturn msg;",
  "name": "Check Alerts"
},{
  "type": "notification",
  "name": "Send Alert"
}]
```

## Hardware Recommendations

### Relay Boards
- **SainSmart 8-Channel**: Good for complete system
- **Elegoo 4-Channel**: Budget option
- **DIN Rail Relays**: Professional installation

### Valves
- **Rain Bird**: Reliable irrigation valves
- **Orbit**: Budget-friendly option  
- **Industrial Solenoids**: For high pressure

### Engine Control
- **Universal Ignition Module**: Works with most engines
- **Diesel Controller**: For diesel generators
- **Remote Start Kit**: Automotive option

### Safety Sensors
- **Float Switches**: Madison M8000 series
- **Pressure Switches**: Honeywell or Square D
- **Pressure Transducers**: For precise monitoring

## Maintenance

### Regular Checks

1. **Weekly**: 
   - Check water levels
   - Verify float switch operation
   - Monitor pressure readings

2. **Monthly**: 
   - Test full sequence (dry run)
   - Check valve operations
   - Inspect wiring connections

3. **Quarterly**: 
   - Clean valve filters
   - Test emergency stops
   - Calibrate sensors

4. **Annually**: 
   - Replace engine oil
   - Test with water flow
   - Verify flow rates
   - Update software
   - Replace valve diaphragms

### Troubleshooting Checklist

- [ ] Power to relay board?
- [ ] GPIO pins connected correctly?
- [ ] Relays clicking when activated?
- [ ] Valve solenoids getting power?
- [ ] Engine battery charged?
- [ ] Fuel in tank?
- [ ] Water in reservoir?
- [ ] Float switch working?
- [ ] Pressure switch calibrated?
- [ ] MQTT broker running?
- [ ] Refill complete?

## Learn More

### GPIO Resources
- [Raspberry Pi GPIO](https://www.raspberrypi.org/documentation/usage/gpio/)
- [RPi.GPIO Documentation](https://pypi.org/project/RPi.GPIO/)
- [GPIO Zero Library](https://gpiozero.readthedocs.io/)

### Pump/Engine Control
- [Small Engine Repair](https://www.briggsandstratton.com/)
- [Irrigation Tutorials](https://www.irrigationtutorials.com/)
- [Relay Logic](https://en.wikipedia.org/wiki/Relay_logic)

### Safety Standards
- [NFPA Fire Pump Standards](https://www.nfpa.org/codes-and-standards)
- [Electrical Safety](https://www.osha.gov/electrical)
- [Water System Design](https://www.epa.gov/watersense)

## Getting Help

If the pump system isn't working:
1. Check GPIO permissions: `groups` should include `gpio`
2. Verify physical connections with multimeter
3. Test each component separately
4. Review logs for error states
5. Check refill status and safety sensors
6. Use manual overrides to verify hardware

Remember: Safety first! The system includes multiple protections, but always have manual override controls. When in doubt, use manual controls and consult a professional for installation of high-voltage or engine control systems.
