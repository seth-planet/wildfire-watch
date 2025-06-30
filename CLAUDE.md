# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Wildfire Watch is an automated fire detection and suppression system that runs on edge devices. It uses AI-powered cameras with multi-camera consensus to detect fires and automatically activates sprinkler systems via GPIO control.

## Tool Usage Guidelines

### Parallel Tool Execution
When performing multiple independent operations, use parallel tool calls in a single message for optimal performance:

```python
# ✅ Good: Parallel tool execution
# Use multiple tool calls in one message
bash_tool.run("git status")
bash_tool.run("git diff") 
read_tool.read("file1.py")
read_tool.read("file2.py")

# ❌ Bad: Sequential messages
# Multiple separate messages with single tool calls
```

**Use parallel tools for:**
- Running multiple bash commands simultaneously
- Reading multiple files for analysis
- Performing independent grep/glob searches
- Creating multiple test scripts or output files

**Benefits:**
- Faster execution and analysis
- More efficient workflow
- Better performance for complex tasks

## Development Commands

### Build and Deployment
```bash
# Generate secure certificates (required for production)
./scripts/generate_certs.sh custom

# Development with hot reload
docker-compose --env-file .env.dev up

# Full deployment
docker-compose up -d

# Multi-platform builds
./scripts/build_multiplatform.sh

# Balena deployment
balena push wildfire-watch
```

### Testing
```bash
# AUTOMATIC PYTHON VERSION SELECTION (Recommended)
# Tests automatically run with correct Python version based on dependencies

# Run all tests with automatic Python version selection
./scripts/run_tests_by_python_version.sh --all

# Run specific Python version tests
./scripts/run_tests_by_python_version.sh --python312  # Most tests
./scripts/run_tests_by_python_version.sh --python310  # YOLO-NAS/super-gradients  
./scripts/run_tests_by_python_version.sh --python38   # Coral TPU/TensorFlow Lite

# Run specific test with auto-detection
./scripts/run_tests_by_python_version.sh --test tests/test_detect.py

# Validate Python environment
./scripts/run_tests_by_python_version.sh --validate

# MANUAL PYTHON VERSION SELECTION (Advanced)
# Python 3.12 tests (most tests) - timeout configuration handles long infrastructure setup
python3.12 -m pytest -c pytest-python312.ini

# Python 3.10 tests (YOLO-NAS/super-gradients)
python3.10 -m pytest -c pytest-python310.ini

# Python 3.8 tests (Coral TPU/tflite_runtime)
python3.8 -m pytest -c pytest-python38.ini

# SPECIFIC TEST CATEGORIES
# Quick tests (skip slow infrastructure setup)
python3.12 -m pytest tests/ -v -m "not slow and not infrastructure_dependent"

# MQTT-specific tests (expected ~15s infrastructure setup)
python3.12 -m pytest tests/ -v -m "mqtt"

# Technology-specific tests
python3.10 -m pytest tests/ -v -m "yolo_nas"      # YOLO-NAS training
python3.8 -m pytest tests/ -v -m "coral_tpu"     # Coral TPU hardware

# Disable timeouts for debugging
python3.12 -m pytest tests/ -v --timeout=0

# See docs/python_version_testing.md for automatic version selection details
# See docs/timeout_configuration.md for timeout handling details
```

### Test Timeout Configuration
All pytest configuration files are pre-configured with appropriate timeouts:
- **pytest.ini** (general): 1 hour per test, 2 hour session timeout
- **pytest-python312.ini**: 1 hour per test, 2 hour session timeout  
- **pytest-python310.ini**: 2 hours per test, 4 hour session timeout (for training tests)
- **pytest-python38.ini**: 2 hours per test, 4 hour session timeout (for model conversion)

These timeouts ensure:
- Infrastructure-heavy tests (MQTT setup, Docker containers) complete successfully
- Model conversion and training tests have sufficient time
- Tests don't hang indefinitely on failures
- Proper cleanup after test completion

For individual test runs that may need extended timeouts:
```bash
# Run with explicit timeout override
python3.12 -m pytest tests/test_trigger.py --timeout=600

# Run all tests with extended timeout
python3.12 -m pytest tests/ --timeout=3600
```

### Python Version
This project requires Python 3.12. All commands should use `python3.12` and `pip3.12`.

**Exceptions**:

1. **Coral TPU**: Requires Python 3.8 for `tflite_runtime` compatibility. When running Coral-specific code:
   - Use `python3.8` instead of `python3.12`
   - Install tflite_runtime with: `python3.8 -m pip install tflite-runtime`
   - See `docs/coral_python38_requirements.md` for details
   - Run `./scripts/check_coral_python.py` to verify Python 3.8 setup

2. **YOLO-NAS Training**: Requires Python 3.10 for `super-gradients` compatibility. When training YOLO-NAS models:
   - Use `python3.10` instead of `python3.12`
   - Install super-gradients with: `python3.10 -m pip install super-gradients`
   - Training scripts in `converted_models/` should be run with Python 3.10:
     ```bash
     python3.10 converted_models/train_yolo_nas.py
     python3.10 converted_models/complete_yolo_nas_pipeline.py
     ```

### Service Management
```bash
# Individual services
docker-compose up mqtt-broker camera-detector
docker-compose logs -f fire-consensus

# Service shell access
docker exec -it camera-detector /bin/bash

# Enable debug logging
LOG_LEVEL=DEBUG docker-compose up
```

## Architecture Overview

### Microservices Communication
All services communicate via MQTT broker with the following data flow:

1. **Camera Detector** → Discovers IP cameras, publishes to `cameras/discovered`
2. **Security NVR (Frigate)** → AI detection, publishes to `frigate/*/fire` and `frigate/*/smoke`
3. **Fire Consensus** → Validates detections, publishes to `trigger/fire_detected`
4. **GPIO Trigger** → Controls pump hardware, publishes to `gpio/status`
5. **Telemetry** → Health monitoring, publishes to `telemetry/*`

### Service Dependencies
```
mqtt_broker (core)
├── camera_detector (needs MQTT)
├── security_nvr (needs MQTT + camera config)
├── fire_consensus (needs MQTT + camera data)
├── gpio_trigger (needs MQTT + consensus)
└── cam_telemetry (needs MQTT)
```

### Key Configuration Files
- `docker-compose.yml` - Service orchestration with healthchecks and dependencies
- `.env` - Environment variables for all services
- `certs/` - TLS certificates (generated by `./scripts/generate_certs.sh`)
- `frigate_config/config.yml` - Dynamically generated by camera_detector

## Development Patterns

### Adding New Camera Support
1. Modify `camera_detector/detect.py` discovery methods
2. Update camera credential handling in environment variables
3. Test RTSP stream validation
4. Ensure MAC address tracking works for persistent identification

### AI Model Integration
- Models stored in `converted_models/` with conversion script
- Supports Coral TPU (.tflite), Hailo (.hef), ONNX, TensorRT formats
- Update `security_nvr/nvr_base_config.yml` for new models
- **Model sizes**: Export in multiple sizes - 640x640 (optimal accuracy), 416x416 (balanced), 320x320 (edge devices)
- **Default model size: 640x640** for optimal fire detection accuracy
- **Fallback to 320x320** for hardware-limited devices (Raspberry Pi, Coral TPU)
- Performance benchmarking required for each accelerator type
- **Note:** Coral TPU requires Python 3.8 for tflite_runtime compatibility
- **Always use QAT (Quantization-Aware Training) for INT8 formats when available**
- Model accuracy validation ensures <2% degradation for production deployment
- **Calibration data**: Download from https://huggingface.co/datasets/mailseth/wildfire-watch/resolve/main/wildfire_calibration_data.tar.gz?download=true

#### Model Conversion Timeouts
Due to the complexity of neural network compilation and optimization, model conversions can take significant time:
- **ONNX**: ~2-5 minutes (includes simplification)
- **TFLite**: ~15-30 minutes (includes INT8 quantization with calibration data)
- **TensorRT**: ~30-60 minutes (engine optimization is compute-intensive)
- **OpenVINO**: ~10-30 minutes (includes IR generation and optimization)
- **Hailo**: ~20-40 minutes (requires Docker and specialized compilation)

All conversion scripts have been configured with appropriate timeouts:
- Standard conversions: 30 minutes
- TensorRT engine building: 60 minutes
- Quantization with large calibration datasets: 30 minutes

If conversions timeout, consider:
1. Using smaller model sizes (320x320 instead of 640x640)
2. Reducing calibration dataset size for quantization
3. Running conversions on more powerful hardware
4. Using pre-converted models when available

### GPIO Safety Systems
- All pump control in `gpio_trigger/trigger.py` uses state machine pattern
- Maximum runtime protection (default 30 minutes)
- Automatic refill calculation based on runtime
- Hardware simulation mode for development on non-Pi systems

### Multi-Camera Consensus Logic
- Located in `fire_consensus/consensus.py`
- Configurable threshold (CONSENSUS_THRESHOLD environment variable)
- Time-based confidence weighting
- Cooldown periods to prevent rapid re-triggering

## Hardware Abstraction

### AI Accelerator Support
- Auto-detection via `FRIGATE_DETECTOR=auto`
- Platform-specific device mappings in docker-compose.yml
- Performance targets: Coral (15-20ms), Hailo (10-25ms), GPU (8-12ms)
- Coral TPU requires Python 3.8 runtime environment

### GPIO Simulation
- Automatically enabled on non-Raspberry Pi systems
- Override with `GPIO_SIMULATION=true/false`
- All pin assignments configurable via environment variables

## Security Architecture

### Certificate Management
- Production requires running `./scripts/generate_certs.sh custom`
- Default certificates are intentionally insecure for development
- TLS enabled via `MQTT_TLS=true` environment variable

### Camera Credentials
- Supports multiple credential sets: `CAMERA_CREDENTIALS=username:password,username2:password2`
- Automatic credential testing during discovery
- MAC address tracking prevents IP-based spoofing

## Testing Strategy

### Unit Tests (`tests/`)
- `test_consensus.py` - Multi-camera validation logic
- `test_detect.py` - Camera discovery and RTSP validation
- `test_trigger.py` - GPIO state machine and safety systems
- `test_telemetry.py` - Health monitoring

### Integration Tests
- `test_integration_e2e.py` - Full system workflow
- `test_hardware_integration.py` - Requires physical hardware
- `test_model_converter.py` - AI model format conversion

### Development Testing
- Use `GPIO_SIMULATION=true` for pump control testing
- Mock MQTT broker available for unit tests
- Camera discovery can be tested with fake RTSP streams

### Test Fixing Guidelines
When fixing failing tests, follow these principles:
1. **Test the actual code, not a mock** - Ensure tests exercise the real implementation
   - If testing `mqtt_connect()`, call the actual function, not a simplified version
   - Only mock external dependencies (network, hardware, time.sleep)
2. **Fix the code, not just the test** - If a test reveals a bug, fix the source code
   - Example: Add retry limits to prevent infinite recursion in network code
   - Make functions testable by adding optional parameters (e.g., `max_retries`)
3. **Preserve test intent** - Understand what the test is trying to verify
   - Don't remove assertions to make tests pass
   - Don't mock away the functionality being tested
4. **Minimal mocking** - Only mock what's necessary
   - Mock external I/O (files, network, hardware)
   - Don't mock the module or functions under test
5. **Test real behavior** - Tests should reflect actual usage
   - If a function is called with certain parameters in production, test those
   - Include edge cases and error conditions

### Integration Testing Philosophy
**Avoid mocking functions and files within wildfire-watch**. We want to perform integration tests that actually test our system without mocking out important functionality:

1. **Never mock internal modules** - Don't mock wildfire-watch modules like `consensus`, `trigger`, `detect`, etc.
   - ❌ Bad: `@patch('consensus.FireConsensus')`
   - ✅ Good: Actually instantiate and use `FireConsensus` class

2. **Only mock external dependencies**:
   - ✅ Mock: `RPi.GPIO`, `docker`, `requests`
   - ✅ Mock: File I/O, network calls, hardware interfaces
   - ✅ Mock: Time delays (`time.sleep`) for faster tests
   - ❌ **DO NOT Mock: `paho.mqtt.client`** - Use real MQTT broker for testing

3. **MQTT Integration Testing Requirements**:
   - **Always use real MQTT broker** - Start and teardown actual MQTT server for proper integration testing
   - Use `TestMQTTBroker` class from `tests/mqtt_test_broker.py` for broker lifecycle management
   - Test actual MQTT message flow between components with real pub/sub
   - Verify real MQTT connection handling, reconnection logic, and message delivery
   - **Never mock MQTT client** - This prevents testing the actual communication layer

4. **Test real interactions**:
   - Test actual MQTT message flow between components
   - Test real state transitions and validation logic
   - Test actual configuration loading and parsing

5. **Use test fixtures properly**:
   - Create real instances of classes under test
   - Set up proper test environments (temp dirs, real MQTT broker)
   - Use `test_mqtt_broker` fixture to provide actual MQTT server
   - Clean up resources properly after tests

6. **Integration test examples**:
   ```python
   # Good: Testing real consensus logic
   consensus = FireConsensus()  # Real instance
   consensus.process_detection(detection)  # Real method call
   
   # Bad: Mocking internal functionality
   @patch('consensus.FireConsensus.process_detection')  # Don't do this
   ```

## Common Environment Variables

### Core Settings
- `CONSENSUS_THRESHOLD=2` - Cameras required for fire trigger
- `MIN_CONFIDENCE=0.7` - AI detection confidence threshold
- `MAX_ENGINE_RUNTIME=1800` - Safety limit in seconds
- `FRIGATE_DETECTOR=auto` - AI accelerator selection

### Development Settings
- `LOG_LEVEL=DEBUG` - Verbose logging
- `GPIO_SIMULATION=true` - Safe testing without hardware
- `DISCOVERY_INTERVAL=300` - Camera discovery frequency

## Deployment Considerations

### Balena Cloud
- Fleet management for multiple edge devices
- Environment variables managed via Balena dashboard
- Automatic updates and rollback support

### Resource Requirements
- Minimum 4GB RAM for Frigate + AI detection
- 32GB+ storage for video retention
- USB 3.0 for Coral TPU, PCIe for Hailo
- GPIO access required for pump control

### Network Architecture
- Cameras discovered via ONVIF, mDNS, port scanning
- MQTT broker creates isolated network (192.168.100.0/24)
- Frigate UI accessible on port 5000
- TLS encryption for production MQTT (port 8883)

## File Organization Guidelines

### Directory Structure for Claude-Generated Files
- **tmp/** - All temporary files including test scripts, debugging files, and intermediate outputs
- **output/** - Final output files, test results, generated reports, and converted models
- **scripts/** - Permanent utility scripts that should be kept in the repository
- **docs/** - Documentation files and guides

### Examples:
```bash
# Temporary test scripts
tmp/test_tensorrt_fix.py
tmp/debug_yolov9.py
tmp/test_conversion.py

# Output files
output/test_results.log
output/model_conversion_report.md
output/converted_models/
output/comprehensive_test_results.log

# Permanent scripts (kept in repo)
scripts/validate_models.py
scripts/run_all_tests.py
scripts/demo_accuracy_validation.py
```

### Scripts (`scripts/`)
- Utility and demo scripts belong in the `scripts/` directory
- Executable scripts should have proper shebang (e.g., `#!/usr/bin/env python3.12`)
- Examples:
  - `scripts/generate_certs.sh` - Certificate generation
  - `scripts/build_multiplatform.sh` - Docker build utilities
  - `scripts/demo_accuracy_validation.py` - Model accuracy demo
  - `scripts/startup_coordinator.py` - Service coordination

### Temporary Files (`tmp/`)
- All testing scripts, debugging files, and temporary utilities belong in `tmp/`
- This includes validation scripts, quick tests, and experimental code
- Examples:
  - `tmp/test_tensorrt_fix.py` - Temporary test scripts
  - `tmp/debug_yolov9.py` - Debugging utilities
  - `tmp/quick_validation.py` - Quick validation tests

### Documentation (`docs/`)
- Technical documentation and guides belong in the `docs/` directory
- Use descriptive filenames with lowercase and underscores
- Examples:
  - `docs/configuration.md` - Configuration guide
  - `docs/hardware.md` - Hardware requirements
  - `docs/accuracy_validation.md` - Model accuracy validation guide
  - `docs/troubleshooting.md` - Common issues and solutions

### Models (`converted_models/`)
- Model conversion scripts and utilities stay in `converted_models/`
- Converted model outputs organized by model name and size
- Calibration data in subdirectories
- Model conversion logs and reports generated here

### Tests (`tests/`)
- All test files must start with `test_` prefix
- Integration tests should be clearly named (e.g., `test_integration_e2e.py`)
- Hardware-specific tests marked appropriately

## Development Workflow for Non-Trivial Work

### Planning Methodology
For any non-trivial work (>30 minutes or involving multiple files), follow this structured approach:

1. **Create a Plan File**
   - Name: `[feature_name]_plan.md` in the appropriate directory
   - Include: Overview, phases, timeline, technical requirements
   - Structure with clear phases and deliverables

2. **Execute Plan with Progress Updates**
   - Mark each phase as: `## Phase X: [Name] - ⏳ IN PROGRESS` when starting
   - Update to: `## Phase X: [Name] - ✅ COMPLETE` when finished
   - Add progress notes with what was accomplished
   - Document any deviations or issues encountered

3. **Testing Requirements**
   - **At the end of each plan**: Run all tests related to the changed code
   - **Test failure priority**: Fix the program's code first, not the test
   - **Change tests only if**: The test itself is incorrect or outdated
   - **Skip tests only if**: They cannot reasonably be made to pass
   - **Document skipped tests**: Note at end of plan with specific reasons

### Plan File Template
```markdown
# [Feature Name] Implementation Plan

## Overview
Brief description of what will be accomplished

## Phases
### Phase 1: [Name] - ⏳ PENDING
- Task 1
- Task 2

### Phase 2: [Name] - ⏳ PENDING  
- Task 1
- Task 2

## Testing
- List of test files that will be affected
- Expected test changes

## Timeline
- Estimated completion time per phase

## Progress Notes
[Add progress updates here as work proceeds]

## Test Results
[Add test results at completion]
- Tests run: X
- Tests passed: Y
- Tests failed: Z
- Tests skipped: N (with reasons)
```

### Examples of Non-Trivial Work
- Adding new model architectures
- Implementing new conversion formats
- Multi-service integrations
- Complex refactoring across multiple files
- New testing frameworks or validation systems

## Documentation Best Practices

### Sphinx-Compatible Documentation
All Python code in this repository should follow Sphinx documentation standards for automatic documentation generation. Documentation should be insightful and helpful for debugging and understanding the system architecture.

Reference: https://www.sphinx-doc.org/en/master/usage/quickstart.html

#### Documentation Principles
1. **Component Connectivity**: Clearly document how each component connects to other services via MQTT topics
2. **Parameter Implications**: Document non-obvious effects of parameter values
3. **Side Effects**: Document any side effects of function calls, especially MQTT publishes
4. **Error Handling**: Document what exceptions can be raised and under what conditions
5. **Thread Safety**: Document if functions/classes are thread-safe or require synchronization
6. **MQTT Topics**: Document all MQTT topics published to or subscribed from

#### Docstring Format
Use Google-style docstrings for consistency:

```python
def process_detection(self, detection: Dict[str, Any]) -> bool:
    """Process a fire detection from a camera and update consensus state.
    
    This method handles incoming fire/smoke detections from the security NVR
    (Frigate) and maintains a sliding window of detections for multi-camera
    consensus. When consensus is reached, it publishes a trigger command.
    
    Args:
        detection: Detection data containing:
            - camera_id (str): Unique camera identifier (MAC address preferred)
            - confidence (float): Detection confidence score (0.0-1.0)
            - object_type (str): 'fire' or 'smoke'
            - timestamp (float): Unix timestamp of detection
            - bbox (dict, optional): Bounding box coordinates
    
    Returns:
        bool: True if consensus threshold is met, False otherwise
        
    Raises:
        ValueError: If detection data is missing required fields
        
    Side Effects:
        - Updates internal detection history
        - May publish to 'trigger/fire_detected' if consensus reached
        - Logs detection details to configured logger
        
    MQTT Topics:
        - Subscribes to: frigate/+/fire, frigate/+/smoke
        - Publishes to: trigger/fire_detected (on consensus)
        
    Thread Safety:
        This method is thread-safe due to internal locking on detection_history
    """
```

#### Class Documentation
```python
class FireConsensus:
    """Multi-camera fire detection consensus manager.
    
    This service subscribes to fire/smoke detections from multiple cameras
    and implements a voting mechanism to reduce false positives. It requires
    a configurable number of cameras to agree before triggering the suppression
    system.
    
    The consensus algorithm uses a sliding time window and confidence weighting
    to evaluate detections. Recent detections are weighted more heavily than
    older ones within the window.
    
    Attributes:
        consensus_threshold (int): Number of cameras required for consensus
        time_window (float): Time window in seconds for valid detections
        detection_history (Dict[str, List[Detection]]): Per-camera detection history
        mqtt_client (mqtt.Client): MQTT client for pub/sub operations
        
    MQTT Integration:
        - Broker: Connects to MQTT_BROKER:MQTT_PORT (default: localhost:1883)
        - Client ID: 'fire_consensus_service'
        - Topics:
            - Subscribes: cameras/discovered, frigate/+/fire, frigate/+/smoke
            - Publishes: trigger/fire_detected, consensus/status
            
    Thread Model:
        - Main thread: MQTT message handling
        - Background thread: Cleanup of old detections (runs every 60s)
        
    Configuration:
        Environment variables:
        - CONSENSUS_THRESHOLD: Number of cameras for consensus (default: 2)
        - TIME_WINDOW: Detection validity window in seconds (default: 30)
        - MIN_CONFIDENCE: Minimum confidence score (default: 0.7)
    """
```

#### Module Documentation
Add module-level docstrings at the top of each Python file:

```python
"""Fire detection consensus service for multi-camera validation.

This module implements the consensus logic for the Wildfire Watch system.
It aggregates fire/smoke detections from multiple cameras running through
the Frigate NVR and determines when to trigger the suppression system.

The consensus mechanism helps reduce false positives by requiring multiple
cameras to detect fire before activation. This is critical for preventing
unnecessary water discharge.

Communication Flow:
    1. Camera Detector publishes discovered cameras to 'cameras/discovered'
    2. Frigate processes video streams and publishes to 'frigate/{camera}/fire'
    3. This service aggregates detections and evaluates consensus
    4. On consensus, publishes to 'trigger/fire_detected'
    5. GPIO Trigger receives command and activates sprinkler system

Integration Points:
    - Upstream: camera_detector (camera discovery), security_nvr (AI detection)
    - Downstream: gpio_trigger (pump control), cam_telemetry (monitoring)
    - Lateral: mqtt_broker (message bus)

Example:
    Run standalone::
    
        $ python3.12 consensus.py
        
    Run in Docker::
    
        $ docker-compose up fire-consensus
"""
```

## AI Assistant Guidelines

### Debugging and Code Analysis
For complex debugging and code analysis tasks, use specialized AI models to leverage their unique strengths:

**Use Gemini (Google) for:**
- **Large context analysis** - Gemini can handle extensive codebases and multiple files simultaneously
- **Complex debugging** - Deep investigation of multi-file issues and complex system interactions
- **Architecture analysis** - Understanding large-scale system relationships and dependencies
- **Code review** - Comprehensive analysis of extensive code changes
- **Performance analysis** - Analyzing performance across multiple components
- **Large context code reviews** - When reviewing multiple files or entire modules

**Use ChatGPT o3 for:**
- **Tricky logic problems** - Complex algorithmic and mathematical reasoning
- **Small context debugging** - Focused analysis of specific functions or modules
- **Logic flow analysis** - Understanding complex control flow and state management
- **Algorithm optimization** - Improving specific algorithmic implementations
- **Edge case identification** - Finding subtle bugs in focused code sections
- **Small context code reviews** - When reviewing specific functions or algorithms
- **Logic evaluation** - Verifying correctness of specific logic implementations

**Model Selection Guidelines:**
- **Context size**: >10 files or >5000 lines → Gemini
- **Logic complexity**: Algorithmic puzzles, mathematical problems → o3
- **System scope**: Multi-service interactions → Gemini
- **Function scope**: Single function debugging → o3
- **Code review scope**: Multiple files → Gemini, Single function → o3
- **Unknown complexity**: Start with Gemini for broader analysis, then o3 for specific issues

### Information Accuracy Guidelines

#### Web Search for Technical Details
When encountering uncertainty about facts, current information, or technical details, always use web search to verify and provide accurate information rather than speculating or admitting uncertainty without investigation.

**Required for web search:**
1. **API Documentation**: When working with specific APIs or libraries, search for current documentation rather than assuming knowledge
2. **Library Features**: Verify available methods, parameters, and usage patterns
3. **Version Compatibility**: Check compatibility between different library versions
4. **Error Messages**: Search for specific error messages to find solutions
5. **Best Practices**: Look up current best practices for frameworks and tools
6. **Breaking Changes**: Verify if APIs have changed between versions

**Examples requiring web search:**
- "What parameters does super-gradients trainer.train() accept?"
- "How to configure YOLO-NAS dataloader in super-gradients 3.6+"
- "What's the correct import path for DetectionMetrics_050?"
- "How to enable QAT in super-gradients?"
- "What are the supported transforms in super-gradients detection?"

**Process:**
1. Use WebSearch tool to find official documentation
2. Verify information from multiple authoritative sources
3. Provide accurate, up-to-date information with source references
4. Update code examples based on current API documentation