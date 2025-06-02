# 🚿 GPIO Trigger Service (Pump Controller)

## What Does This Do?

The GPIO Trigger Service controls your water pump and sprinkler system. It's the "muscle" of your wildfire defense that:

- 💧 **Controls water valves** - Opens/closes water flow
- 🔥 **Starts fire pump** - Activates gas/diesel pump engine  
- ⏱️ **Manages timing** - Runs pump for appropriate duration
- 🔄 **Handles refill** - Automatically refills water reservoir
- 🛡️ **Prevents damage** - Protects pump from running dry

## Why This Matters

This service ensures your sprinkler system activates reliably when fire is detected, while preventing costly damage from:
- Running pumps without water
- Engine damage from improper shutdown
- Reservoir depletion
- False activations

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
4. Monitors for continued fire
5. Shuts down safely when fire is gone
6. Refills reservoir from well/source

### What You'll See

Normal operation logs:
```
[INFO] Event: valve_opened, State: {'MAIN_VALVE': True}
[INFO] Event: priming_started, State: {'PRIMING_VALVE': True}
[INFO] Event: engine_start_sequence
[INFO] Event: ignition_on, State: {'IGN_ON': True}
[INFO] Event: engine_running
[INFO] Engine running with priming valve open for air removal
[INFO] Priming duration complete - closing bleed valve
[INFO] Event: priming_complete, State: {'PRIMING_VALVE': False}
[INFO] Event: rpm_reduced
[INFO] Event: shutdown_initiated
```

## Hardware Connections

### GPIO Pin Assignments

| Function | Default Pin | Wire Color (Suggested) | Description |
|----------|------------|------------------------|-------------|
| Main Valve | GPIO 18 | Blue | Opens main water valve to sprinklers |
| Ignition Start | GPIO 23 | Yellow | Pulse to start engine (like turning key) |
| Ignition On | GPIO 24 | Green | Stays on while engine runs |
| Ignition Off | GPIO 25 | Red | Pulse to stop engine |
| Refill Valve | GPIO 22 | Purple | Opens valve to refill reservoir |
| Priming Valve | GPIO 26 | Orange | Bleeds air from pump |
| RPM Reduce | GPIO 27 | Brown | Reduces engine speed before shutdown |

### Wiring Diagram Example

```
Raspberry Pi                    Relay Board
GPIO 18 -----------------> Relay 1 -> Main Valve Solenoid
GPIO 23 -----------------> Relay 2 -> Ignition Start Circuit
GPIO 24 -----------------> Relay 3 -> Ignition Run Circuit
GPIO 25 -----------------> Relay 4 -> Ignition Stop Circuit
GND ---------------------> Relay Board Ground
```

**💡 Tip**: Use a relay board rated for your valve/ignition voltage (typically 12V or 24V)

## Configuration Options

### Timing Configuration

```bash
# Valve and startup timings
VALVE_PRE_OPEN_DELAY=2      # Open valve 2 seconds before starting
IGNITION_START_DURATION=5   # Hold start for 5 seconds
PRIMING_DURATION=180        # Run engine with bleed valve open for 3 minutes

# Shutdown timings  
FIRE_OFF_DELAY=1800         # Run 30 minutes after fire gone
VALVE_CLOSE_DELAY=600       # Close valve 10 min after engine stops
IGNITION_OFF_DURATION=5     # Pulse stop for 5 seconds
RPM_REDUCTION_LEAD=15       # Reduce RPM 15 seconds before stop

# Safety limits
MAX_ENGINE_RUNTIME=600      # Maximum 10 minute run time
REFILL_MULTIPLIER=40        # Refill 40x longer than runtime
```

### Pin Customization

```bash
# Change pin assignments if needed
MAIN_VALVE_PIN=18
IGNITION_START_PIN=23
IGNITION_ON_PIN=24
IGNITION_OFF_PIN=25
REFILL_VALVE_PIN=22
PRIMING_VALVE_PIN=26
RPM_REDUCE_PIN=27
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

2. **Priming Valve Opens** (before start)
   - Opens air bleed valve
   - Prepares to remove air from pump

3. **Engine Start** (5 sec)
   - Ignition start pulse
   - Like turning a key

4. **Running State**
   - Engine runs with priming valve OPEN
   - Air bleeds out while pump runs
   - Refill valve opens
   
5. **Priming Complete** (after 3 min)
   - Priming valve closes
   - Pump now at full pressure/flow
   - No air in pump housing

### Shutdown Sequence

1. **RPM Reduction** (15 sec before)
   - Reduces engine speed
   - Gentler shutdown

2. **Engine Stop** (5 sec pulse)
   - Ignition off signal
   - Engine stops

3. **Valve Timing**
   - Main valve stays open 10 min
   - Refill continues based on runtime
   - All valves eventually close

### Priming Sequence Timeline

```
Time    Main Valve    Priming Valve    Engine    Action
----    ----------    -------------    ------    ------
0s      CLOSED        CLOSED           OFF       Fire detected
2s      OPEN          CLOSED           OFF       Main valve opens
2s      OPEN          OPEN             OFF       Priming valve opens
4s      OPEN          OPEN             START     Engine cranking
9s      OPEN          OPEN             RUN       Engine running (air bleeding)
...     OPEN          OPEN             RUN       Air continues bleeding out
180s    OPEN          OPEN             RUN       3 minutes elapsed
180s    OPEN          CLOSED           RUN       Priming complete, full pressure
...     OPEN          CLOSED           RUN       Normal pumping operation
```

**Critical Points**:
- Engine MUST run with priming valve open
- Air removal takes ~3 minutes of operation
- Closing priming valve too early = poor performance
- Opening priming valve during operation = pressure loss

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
4. **Add float switch**: Wire to GPIO for water level sensing

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
   MAX_ENGINE_RUNTIME=300  # 5 minutes max
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
5. **Reconnect gradually** - Test each component

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
    27: "RPM Reduce"
}

GPIO.setmode(GPIO.BCM)
for pin in pins:
    GPIO.setup(pin, GPIO.OUT)

# Test each output
for pin, name in pins.items():
    print(f"Testing {name} on GPIO {pin}")
    GPIO.output(pin, GPIO.HIGH)
    time.sleep(2)
    GPIO.output(pin, GPIO.LOW)
    time.sleep(1)

GPIO.cleanup()
```

## State Machine Explained

The controller uses a state machine for safety:

```
IDLE → PRIMING → STARTING → RUNNING → REDUCING_RPM → STOPPING → COOLDOWN → IDLE
         ↓                      ↓           ↓            ↓
        ERROR ←────────────────┴───────────┴────────────┘
```

**States**:
- **IDLE**: Ready, waiting for fire
- **PRIMING**: Valves open, preparing to start engine
- **STARTING**: Engine cranking/starting up
- **RUNNING**: Engine running, initially with bleed valve open for air removal, then closed for full pressure
- **REDUCING_RPM**: Slowing engine before stop
- **STOPPING**: Shutting down engine
- **COOLDOWN**: Post-shutdown wait period
- **ERROR**: Manual intervention needed

**Key Timing**:
- Priming valve opens BEFORE engine starts
- Engine runs WITH priming valve open for 3 minutes
- Only after air is bled does priming valve close
- This ensures maximum pump performance

## Safety Features

### Automatic Protections

1. **Valve-First Start**: Valve must be open before engine
2. **Maximum Runtime**: Prevents endless running
3. **RPM Reduction**: Gentle shutdown sequence
4. **Cooldown Period**: Prevents rapid cycling
5. **State Validation**: Can't start from invalid state

### Manual Overrides

Always install these physical controls:
1. **Emergency Stop Button** - Kills engine immediately
2. **Manual Valve Control** - Bypass GPIO control
3. **Fuel Shutoff** - Stops engine without electrical
4. **Main Power Switch** - Disconnects all systems

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
[INFO] Event: valve_opened
[INFO] Event: engine_running
[INFO] Event: shutdown_complete, runtime: 300
```

**Issues**:
```
[ERROR] Main valve not open - aborting engine start
[WARNING] Cannot reduce RPM from IDLE state
[ERROR] Failed to set MAIN_VALVE: [Errno 13] Permission denied
```

## Advanced Customization

### Adding Water Level Sensor

```python
# Add to trigger.py
WATER_LEVEL_PIN = 17
GPIO.setup(WATER_LEVEL_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def check_water_level(self):
    return GPIO.input(WATER_LEVEL_PIN) == GPIO.LOW

# In start sequence
if not self.check_water_level():
    self._enter_error_state("Low water level")
    return
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

### SMS/Email Alerts

```python
# Add notification on trigger
def _notify_trigger(self):
    message = f"Fire detected! Pump activated at {datetime.now()}"
    # Send via your preferred method
    send_sms(message)
    send_email(message)
```

## Integration Examples

### Home Assistant

```yaml
switch:
  - platform: mqtt
    name: "Fire Pump"
    command_topic: "fire/trigger"
    state_topic: "system/trigger_telemetry"
    value_template: "{{ value_json.system_state.engine_on }}"
    payload_on: "{}"
    payload_off: "stop"
```

### Node-RED Flow

```json
[{
  "type": "mqtt in",
  "topic": "system/trigger_telemetry",
  "name": "Pump Status"
},{
  "type": "function",
  "func": "msg.payload = JSON.parse(msg.payload);\nreturn msg;",
  "name": "Parse JSON"
},{
  "type": "ui_text",
  "label": "Pump State",
  "name": "Display State"
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

## Maintenance

### Regular Checks

1. **Monthly**: Test full sequence (dry run)
2. **Quarterly**: Clean valve filters
3. **Annually**: 
   - Replace engine oil
   - Test emergency stops
   - Verify water flow rates
   - Update software

### Troubleshooting Checklist

- [ ] Power to relay board?
- [ ] GPIO pins connected correctly?
- [ ] Relays clicking when activated?
- [ ] Valve solenoids getting power?
- [ ] Engine battery charged?
- [ ] Fuel in tank?
- [ ] Water in reservoir?
- [ ] MQTT broker running?

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

## Getting Help

If the pump system isn't working:
1. Check GPIO permissions: `groups` should include `gpio`
2. Verify physical connections with multimeter
3. Test each component separately
4. Review logs for error states
5. Use manual overrides to verify hardware

Remember: Safety first! When in doubt, use manual controls and consult a professional for installation of high-voltage or engine control systems.
