# Documentation Structure

This document describes the organization of documentation in the Wildfire Watch project.

## Active Documentation

### Root Directory
- `README.md` - Main project overview and quick start
- `CONTRIBUTING.md` - Contribution guidelines
- `CLAUDE.md` - Claude AI assistant instructions
- `GEMINI.md` - Gemini AI assistant instructions (if applicable)
- `DOCUMENTATION_INDEX.md` - Master index of all documentation

### `/docs` - Current Documentation
- `configuration.md` - System configuration guide
- `hardware.md` - Hardware requirements and setup
- `security.md` - Security architecture and practices
- `troubleshooting.md` - Common issues and solutions
- `multi-node.md` - Multi-node deployment guide
- `QUICK_START_pc.md` - PC quick start guide
- `QUICK_START_pi5.md` - Raspberry Pi 5 quick start guide
- `OPERATIONAL_RUNBOOK.md` - Operations guide
- `EMERGENCY_PROCEDURES_CARD.md` - Emergency procedures
- `TEAM_TRAINING_GUIDE.md` - Team training materials

### `/docs/adr` - Architecture Decision Records
- `ADR-001-service-refactoring.md` - Service refactoring decisions
- `ADR-002-mqtt-topic-naming.md` - MQTT topic naming conventions

### Service Documentation
- `/camera_detector/README.md` - Camera detection service
- `/cam_telemetry/README.md` - Camera telemetry service
- `/fire_consensus/README.md` - Fire consensus service
- `/gpio_trigger/README.md` - GPIO trigger service
- `/mqtt_broker/README.md` - MQTT broker service
- `/security_nvr/README.md` - Security NVR service
- `/certs/README.md` - TLS certificate management

### Model and Conversion Documentation
- `/converted_models/README.md` - Model conversion overview
- `/converted_models/YOLO_NAS_TRAINING_README.md` - YOLO-NAS training guide
- Various accuracy reports in model output directories

### Test Documentation
- `/tests/README.md` - Test suite overview
- `/tests/E2E_TEST_README.md` - End-to-end test guide
- `/tests/HARDWARE_TEST_README.md` - Hardware test guide
- `/tests/README_security_nvr_tests.md` - Security NVR test guide
- `/tests/README_hailo_tests.md` - Hailo test guide

## Archived Documentation

### `/docs/archive` - Historical Technical Documentation
Contains completed implementation plans, test results, and technical summaries from development phases.

### `/docs/archived` - Historical Project Documentation
Contains project status reports, completion summaries, and milestone documentation.

### `/output` - Build and Test Output Reports
Contains final test reports, deployment guides, and validation summaries.

## Documentation Guidelines

1. **Active Documentation**: Keep in `/docs` or service directories
2. **Historical Documentation**: Move to `/docs/archive` or `/docs/archived`
3. **Build Outputs**: Keep in `/output`
4. **Service-Specific**: Keep in service directories
5. **Test Documentation**: Keep in `/tests`

## Maintenance

- Review quarterly to archive completed project documentation
- Keep active documentation up-to-date with code changes
- Use ADRs for significant architectural decisions
- Archive old plans and summaries to maintain clarity