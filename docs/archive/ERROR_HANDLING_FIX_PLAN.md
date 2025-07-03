# Error Handling Fix Plan

## Overview
Based on the comprehensive error analysis, this plan addresses all critical issues in priority order.

## Phase 1: Critical Safety Fixes (GPIO Trigger) - ✅ COMPLETE

### 1.1 Fix Emergency Procedures Error Handling - ✅ COMPLETE
- [x] Add try-except blocks to emergency valve operations
- [x] Add hardware failure recovery logic
- [x] Implement safe failure modes
- [x] Add state verification after operations

**Completed Actions:**
- Created `gpio_safety.py` module with SafeGPIO wrapper
- Implemented read-after-write verification for critical pins
- Added emergency_all_off() method for best-effort shutdown
- Integrated GPIOVerificationError for critical pin failures

### 1.2 Fix Thread Safety in State Machine - ✅ COMPLETE
- [x] Add locks for state transitions
- [x] Make timer operations atomic
- [x] Fix concurrent access to _timers dict

**Completed Actions:**
- Created ThreadSafeStateMachine base class
- Implemented SafeTimerManager for thread-safe timer operations
- Updated PumpController to inherit from ThreadSafeStateMachine
- All timer operations now use SafeTimerManager when available

### 1.3 Add GPIO Simulation Improvements - ✅ COMPLETE
- [x] Better error messages for hardware failures
- [x] Proper cleanup on errors

**Completed Actions:**
- Enhanced error messages with pin names and failure types
- Added failure statistics tracking
- Implemented proper cleanup in emergency procedures

## Phase 2: Network Timeout Fixes - ✅ COMPLETE

### 2.1 Camera Detector Timeouts - ✅ COMPLETE
- [x] Add socket timeout to MQTT connection
- [x] Fix RTSP validation hanging issues  
- [x] Add timeout to ONVIF operations
- [x] Fix subprocess command timeouts

**Completed Actions:**
- Created `mqtt_resilient.py` with ResilientMQTTClient class
- Implemented socket timeouts, retry limits, and graceful degradation
- Created `network_timeout_utils.py` with timeout-aware network operations
- Created `detect_resilient.py` using the new resilient components

### 2.2 Fire Consensus Timeouts - ✅ COMPLETE
- [x] Add max retry limit to MQTT connection
- [x] Fix infinite recursion in reconnect

**Completed Actions:**
- Created `consensus_resilient.py` using ResilientMQTTClient
- Removed infinite retry loops, added max retry limits
- Implemented FAILED state with recovery attempts
- Added queue-based message processing for thread safety

## Phase 3: Hardware Assumptions - ✅ COMPLETE

### 3.1 Resolution Independence - ✅ COMPLETE
- [x] Make camera resolution configurable
- [x] Fix hardcoded 1920x1080 assumption
- [x] Support 4K and 720p cameras

**Completed Actions:**
- Created `camera_resolution_handler.py` with dynamic resolution support
- Implemented letterboxing to preserve aspect ratios
- Added resolution detection and optimal selection
- Created coordinate transformation for bbox mapping

### 3.2 Device Enumeration - ✅ COMPLETE
- [x] Enumerate GPU devices dynamically
- [x] Remove hardcoded render device paths  
- [x] Support multiple Coral TPUs

**Completed Actions:**
- Created `hardware_detector_improved.py` with proper enumeration
- Added support for multiple GPUs (NVIDIA, AMD, Intel)
- Enumeration of all Coral TPUs (USB and PCIe)
- Added Hailo device enumeration
- Removed hardcoded device assumptions

## Phase 4: Configuration Decoupling - ✅ COMPLETE

### 4.1 Service Discovery - ✅ COMPLETE
- [x] Add environment-based service resolution
- [x] Fix Docker vs bare metal hostname issues
- [x] Implement proper path resolution

### 4.2 Topic Configuration - ✅ COMPLETE
- [x] Centralize MQTT topic definitions
- [x] Add topic validation
- [x] Remove hardcoded topics

**Completed Actions:**
- Created `config_manager.py` with centralized configuration
- Implemented ServiceDiscovery for deployment mode detection
- Created PathResolver for Docker vs bare metal paths
- Centralized all MQTT topics in MQTTTopicConfig class
- Added service-specific config classes
- Created migration guide in `docs/configuration_migration_guide.md`

## Phase 5: Thread Safety - ✅ COMPLETE

### 5.1 Camera Detector - ✅ COMPLETE
- [x] Add locks for cameras dict
- [x] Fix race conditions in discovery
- [x] Make MAC tracker thread-safe

### 5.2 Telemetry Service - ✅ COMPLETE
- [x] Protect global state with locks (already implemented)
- [x] Fix timer management (already thread-safe)

**Completed Actions:**
- Created `camera_detector/thread_safety.py` with thread-safe collections
- Created `camera_detector/detect_thread_safe_mixin.py` for easy integration
- Created `fire_consensus/thread_safety.py` with thread-safe components
- Created comprehensive `docs/thread_safety_guide.md`
- Created `tests/test_thread_safety.py` with thorough testing
- Verified telemetry service already has thread safety with _state_lock

## Phase 6: Docker vs Bare Metal - ⏳ PENDING

### 6.1 Path Resolution
- [ ] Add deployment mode detection
- [ ] Implement path abstraction layer
- [ ] Fix certificate path issues

### 6.2 Permission Handling
- [ ] Add proper error messages for permission issues
- [ ] Handle Docker device mapping errors

## Phase 7: Test Fixes - ⏳ PENDING

### 7.1 Fix Failing Tests
- [ ] Fix integration test timeouts
- [ ] Remove internal mocking
- [ ] Add hardware test coverage

### 7.2 Add Missing Tests
- [ ] Hardware failure tests
- [ ] Timeout tests
- [ ] Thread safety tests

## Execution Order
1. GPIO Trigger safety (CRITICAL)
2. Network timeouts (HIGH)
3. Thread safety (HIGH)
4. Hardware assumptions (MEDIUM)
5. Configuration decoupling (MEDIUM)
6. Docker vs bare metal (LOW)
7. Test improvements (ONGOING)