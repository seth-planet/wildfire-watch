# Configuration Dependencies Analysis Report - Wildfire Watch

## Executive Summary

This report analyzes configuration dependencies across the Wildfire Watch core services to identify coupling issues and create a decoupling strategy. The analysis reveals significant configuration coupling, hardcoded values, and opportunities for improvement.

## Core Services Analyzed

1. **camera_detector/detect.py** - Camera discovery and management
2. **fire_consensus/consensus.py** - Multi-camera fire consensus 
3. **gpio_trigger/trigger.py** - Physical pump control
4. **cam_telemetry/telemetry.py** - System health monitoring
5. **security_nvr/camera_manager.py** - Frigate configuration sync
6. **security_nvr/hardware_detector.py** - Hardware capability detection

## Key Findings

### 1. Environment Variable Usage Patterns

#### Common Configuration Approach
All services use a similar pattern:
```python
class Config:
    def __init__(self):
        self.SETTING = os.getenv("ENV_VAR", "default_value")
```

**Issues:**
- No validation of environment variable values
- Type conversion errors not handled gracefully
- Missing values silently use defaults
- No configuration schema or documentation

#### Service-Specific Findings

**Camera Detector (detect.py)**
- 35+ configuration parameters
- Complex validation logic (e.g., port ranges, time limits)
- Some validation but inconsistent application
- Hardcoded resolution assumption (1920x1080)

**Fire Consensus (consensus.py)**
- 20+ configuration parameters  
- Minimal validation
- Critical safety parameters (thresholds) not validated
- Zone mapping uses raw JSON parsing without validation

**GPIO Trigger (trigger.py)**
- 40+ configuration parameters
- Mix of direct dict access and Config class
- Critical safety timings not validated
- No range checking on timing parameters

**Telemetry (telemetry.py)**
- Minimal configuration (6 parameters)
- No validation at all
- Direct environment access without Config class

**Camera Manager**
- Direct os.environ.get() throughout code
- No centralized configuration
- String-based substitution risks YAML injection

**Hardware Detector**
- No configuration class
- Hardcoded paths throughout
- No configurability for detection methods

### 2. Hardcoded Values

#### Network Configuration
- MQTT default ports: 1883 (plain), 8883 (TLS)
- Network discovery: 192.168.100.0/24 hardcoded
- Timeouts: Various hardcoded timeouts (5s, 10s, 30s, 60s)

#### Hardware Assumptions
- GPIO pin assignments have defaults but no validation
- Camera resolution: 1920x1080 assumed for area calculations
- Detection thresholds: 0.7 confidence, area ratios

#### File Paths
- Certificate paths: /mnt/data/certs/*
- Config paths: /config/*
- Model paths: /models/wildfire/*

### 3. Cross-Service Dependencies

#### MQTT Topic Coupling
Services are tightly coupled through MQTT topics:
- `camera/discovery/+` - Camera detector → Camera manager
- `fire/detection/+` - Cameras → Fire consensus
- `fire/trigger` - Fire consensus → GPIO trigger
- `system/telemetry` - All services → Monitoring

**Issues:**
- Topic names hardcoded in multiple places
- No topic validation or schema
- Services assume specific message formats

#### Timing Dependencies
- Camera discovery interval affects consensus
- Consensus window must align with detection rate
- GPIO refill timing depends on runtime calculations
- No coordination of timing parameters

### 4. Configuration Validation Issues

#### Missing Validation
- String values not checked for valid content
- Numeric ranges not enforced
- Boolean parsing inconsistent
- No schema validation

#### Type Safety
- Manual int() conversions without error handling
- Float parsing can fail silently
- JSON parsing without validation
- No type hints on config values

#### Error Handling
- Most services continue with defaults on errors
- No configuration error reporting
- Silent failures make debugging difficult

### 5. Configuration Coupling Anti-Patterns

#### Tight Coupling Examples
1. **Camera detector hardcodes Frigate paths**
   ```python
   self.FRIGATE_CONFIG_PATH = os.getenv("FRIGATE_CONFIG_PATH", "/config/frigate/cameras.yml")
   ```

2. **Fire consensus assumes camera detector message format**
   ```python
   camera_id = data.get('camera_id')  # Assumes specific JSON structure
   ```

3. **GPIO trigger has embedded MQTT broker config**
   ```python
   'MQTT_BROKER': os.getenv('MQTT_BROKER', 'mqtt_broker')
   ```

#### Configuration Sprawl
- Each service defines its own MQTT settings
- Duplicated timeout configurations
- No shared configuration validation

## Recommendations

### 1. Centralized Configuration Management

Create a shared configuration module:
```python
# config/base.py
from dataclasses import dataclass
from typing import Optional
import os
from config.validators import validate_port, validate_timeout

@dataclass
class MQTTConfig:
    broker: str
    port: int
    tls_enabled: bool
    ca_path: Optional[str]
    
    @classmethod
    def from_env(cls):
        return cls(
            broker=os.getenv("MQTT_BROKER", "mqtt_broker"),
            port=validate_port(os.getenv("MQTT_PORT", "1883")),
            tls_enabled=os.getenv("MQTT_TLS", "false").lower() == "true",
            ca_path=os.getenv("TLS_CA_PATH") if os.getenv("MQTT_TLS", "false").lower() == "true" else None
        )
```

### 2. Configuration Schema and Validation

Implement JSON Schema validation:
```python
# config/schemas/camera_detector.json
{
  "type": "object",
  "properties": {
    "discovery_interval": {
      "type": "integer",
      "minimum": 30,
      "maximum": 3600
    },
    "rtsp_timeout": {
      "type": "integer", 
      "minimum": 1,
      "maximum": 60
    }
  },
  "required": ["discovery_interval"]
}
```

### 3. Decouple Services Through Interfaces

Define clear service interfaces:
```python
# interfaces/camera_discovery.py
from abc import ABC, abstractmethod
from typing import Dict, List

class CameraDiscoveryInterface(ABC):
    @abstractmethod
    def discover_cameras(self) -> List[Dict]:
        """Return list of discovered cameras"""
        pass
```

### 4. Configuration Service

Create a configuration service that:
- Loads configuration from multiple sources (env, files, defaults)
- Validates all configuration at startup
- Provides typed configuration objects
- Supports hot-reloading for non-critical settings

### 5. Specific Improvements by Service

#### Camera Detector
- Move network ranges to configuration
- Make resolution configurable per camera
- Externalize discovery methods configuration
- Add discovery strategy patterns

#### Fire Consensus  
- Validate all threshold parameters
- Make consensus algorithm pluggable
- Add configuration for different fire types
- Support dynamic threshold adjustment

#### GPIO Trigger
- Validate all timing parameters at startup
- Add safety limit configurations
- Support hardware profiles
- Make pin assignments dynamic

#### Camera Manager
- Use proper YAML manipulation instead of string replacement
- Add configuration validation before writing
- Support configuration templates
- Add rollback capability

#### Hardware Detector
- Make detection methods configurable
- Add hardware profile definitions
- Support custom detection scripts
- Cache detection results

### 6. Migration Strategy

1. **Phase 1: Extract Common Configuration**
   - Create shared MQTT configuration
   - Extract common validation functions
   - Add configuration documentation

2. **Phase 2: Add Validation Layer**
   - Implement schema validation
   - Add startup configuration checks
   - Create configuration test suite

3. **Phase 3: Decouple Services**
   - Define service interfaces
   - Implement message schemas
   - Add integration tests

4. **Phase 4: Advanced Features**
   - Add configuration service
   - Implement hot-reloading
   - Add configuration UI

## Configuration Debt Items

### High Priority
1. Camera resolution hardcoded to 1920x1080 in consensus calculations
2. No validation of safety-critical GPIO timing parameters
3. MQTT topics hardcoded across services
4. Network discovery ranges hardcoded

### Medium Priority
1. Certificate paths hardcoded
2. No configuration schema documentation
3. Inconsistent boolean parsing
4. Missing timeout validations

### Low Priority
1. Default values scattered across codebase
2. No configuration versioning
3. Missing configuration metrics
4. No configuration change auditing

## Conclusion

The Wildfire Watch system has significant configuration coupling that impacts maintainability, testability, and reliability. The current approach of direct environment variable access with minimal validation creates risks, especially for safety-critical parameters in the GPIO trigger service.

Implementing the recommended configuration management strategy will:
- Improve system reliability through validation
- Enhance maintainability through centralization
- Enable better testing through decoupling
- Support future scaling through proper abstractions

Priority should be given to validating safety-critical configurations in the GPIO trigger service and fixing the hardcoded camera resolution assumption in the fire consensus service.