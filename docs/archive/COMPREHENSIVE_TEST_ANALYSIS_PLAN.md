# Comprehensive Test Analysis Plan

## Objective
Identify and fix all failing tests in the wildfire-watch system, ensuring proper error handling, thread safety, and hardware compatibility.

## Test Execution Strategy

### 1. Initial Test Discovery
- Run all tests with Python version compatibility
- Use 30-minute timeouts to catch hanging tests
- Document all failures, timeouts, and skipped tests

### 2. Test Categories to Analyze
1. **Unit Tests** - Core logic without external dependencies
2. **Integration Tests** - Service interactions via MQTT
3. **Hardware Tests** - Coral TPU, GPU, camera tests
4. **Docker Tests** - Container-specific behavior
5. **End-to-End Tests** - Full system workflows

### 3. Known Issues to Address
- Thread safety in shared state access
- Docker vs bare metal path differences
- Hardware enumeration assumptions
- Internal mocking that should be removed
- Missing error handling for external commands

## Phase 5: Thread Safety Analysis

### Services to Review:
1. **Camera Detector** (`camera_detector/detect.py`)
   - Shared camera dictionary
   - MAC tracker state
   - Discovery thread coordination

2. **Fire Consensus** (`fire_consensus/consensus.py`)
   - Detection history management
   - Object tracker updates
   - Camera state tracking

3. **Telemetry Service** (`cam_telemetry/telemetry.py`)
   - Global state variables
   - Metric collection

4. **GPIO Trigger** (`gpio_trigger/trigger.py`)
   - Timer management (already addressed)
   - State transitions (already addressed)

### Thread Safety Checklist:
- [ ] Identify all shared mutable state
- [ ] Add appropriate locking mechanisms
- [ ] Ensure atomic operations
- [ ] Prevent race conditions
- [ ] Test concurrent access patterns

## Phase 6: Docker vs Bare Metal

### Path Handling:
- [ ] Certificate paths
- [ ] Model file paths
- [ ] Configuration file paths
- [ ] Log file paths
- [ ] Data directory paths

### Permission Issues:
- [ ] Device access permissions
- [ ] File system permissions
- [ ] Network capabilities

### Service Discovery:
- [ ] Container networking
- [ ] Host networking mode
- [ ] Service name resolution

## Phase 7: Test Fixes

### Mocking Review:
- [ ] Remove internal service mocks
- [ ] Keep only external dependency mocks
- [ ] Ensure integration tests use real services
- [ ] Fix tests that rely on mocked behavior

### Timeout Issues:
- [ ] Infrastructure setup timeouts
- [ ] MQTT broker startup times
- [ ] Camera discovery delays
- [ ] Model loading times

### Hardware Test Categories:
- [ ] Coral TPU detection and usage
- [ ] GPU enumeration (NVIDIA, AMD)
- [ ] Camera RTSP validation
- [ ] Hardware accelerator fallback

## Phase 8: Hardware Test Coverage

### Coral TPU Tests:
- [ ] USB Coral detection
- [ ] PCIe Coral detection
- [ ] Model inference on Coral
- [ ] Python 3.8 compatibility

### GPU Tests:
- [ ] NVIDIA GPU with TensorRT
- [ ] AMD GPU with ROCm
- [ ] Intel GPU with VA-API
- [ ] Multi-GPU systems

### Camera Tests:
- [ ] ONVIF discovery
- [ ] RTSP stream validation
- [ ] Multiple camera support
- [ ] Resolution detection

### Raspberry Pi 5:
- [ ] GPIO functionality
- [ ] Balena deployment
- [ ] Resource constraints
- [ ] Hardware acceleration

## Test Execution Plan

### Step 1: Baseline Test Run
```bash
# Run all tests with automatic Python version selection
./scripts/run_tests_by_python_version.sh --all --timeout=1800 > baseline_test_results.log 2>&1
```

### Step 2: Categorize Failures
1. Actual test failures (assertion errors)
2. Timeout failures (hanging tests)
3. Import/dependency errors
4. Mock-related failures
5. Hardware-specific failures

### Step 3: Fix Priority Order
1. **Critical**: Thread safety issues (data corruption risk)
2. **High**: Test infrastructure (mocking, timeouts)
3. **Medium**: Hardware compatibility
4. **Low**: Code style, warnings

### Step 4: Validation Strategy
After each fix:
1. Run affected test file
2. Run related integration tests
3. Check for regressions in other tests
4. Verify on target hardware if applicable

## Success Criteria

1. **All tests pass** without timeouts or skips
2. **No internal mocking** of wildfire-watch components
3. **Hardware tests** validate actual hardware when available
4. **Thread safety** verified under concurrent load
5. **Docker and bare metal** both work correctly
6. **Raspberry Pi 5** deployment successful

## Gemini Consultation Points

1. **Before Phase 5**: Thread safety patterns and best practices
2. **Before Phase 6**: Docker vs bare metal abstraction strategies
3. **After major refactoring**: Code review for correctness
4. **For complex bugs**: Root cause analysis
5. **For test design**: Hardware test strategies

## Documentation Requirements

1. Update README files with new error handling
2. Document thread safety guarantees
3. Add hardware compatibility matrix
4. Create troubleshooting guide
5. Update API documentation