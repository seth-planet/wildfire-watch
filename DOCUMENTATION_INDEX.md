# Wildfire Watch Documentation Index

## 📋 Project Overview
- [PROJECT_COMPLETION_SUMMARY.md](./PROJECT_COMPLETION_SUMMARY.md) - Overall project status and achievements
- [FINAL_PROJECT_SUMMARY.md](./FINAL_PROJECT_SUMMARY.md) - Comprehensive project summary
- [EXECUTIVE_SUMMARY_PRESENTATION.md](./EXECUTIVE_SUMMARY_PRESENTATION.md) - Executive briefing

## 🚀 Deployment Documentation
- [PRODUCTION_DEPLOYMENT_GUIDE.md](./PRODUCTION_DEPLOYMENT_GUIDE.md) - Step-by-step deployment instructions
- [PRODUCTION_READINESS_CHECKLIST.md](./PRODUCTION_READINESS_CHECKLIST.md) - Pre-deployment verification
- [DEPLOYMENT_TESTING_CHECKLIST.md](./DEPLOYMENT_TESTING_CHECKLIST.md) - Testing requirements

## 🛠️ Technical Documentation
- [REFACTORING_STATUS_SUMMARY.md](./REFACTORING_STATUS_SUMMARY.md) - Service refactoring status
- [E2E_TEST_FINAL_SUMMARY.md](./E2E_TEST_FINAL_SUMMARY.md) - Test results and fixes
- [docs/adr/ADR-001-service-refactoring.md](./docs/adr/ADR-001-service-refactoring.md) - Architecture decision
- [docs/adr/ADR-002-mqtt-topic-naming.md](./docs/adr/ADR-002-mqtt-topic-naming.md) - Topic naming convention

## 📚 Operational Guides
- [docs/OPERATIONAL_RUNBOOK.md](./docs/OPERATIONAL_RUNBOOK.md) - Day-to-day operations
- [docs/EMERGENCY_PROCEDURES_CARD.md](./docs/EMERGENCY_PROCEDURES_CARD.md) - Quick reference card
- [docs/TEAM_TRAINING_GUIDE.md](./docs/TEAM_TRAINING_GUIDE.md) - Team training modules

## 🔧 Scripts and Tools

### Deployment Scripts
- `scripts/deploy_refactored_services.sh` - Automated deployment with rollback
- `scripts/post_deployment_validation.py` - Post-deployment health checks
- `scripts/monitor_topic_migration.py` - Topic migration tracking

### Test Scripts
- `test_gpio_trigger_simple.py` - Simple integration test
- `scripts/run_tests_by_python_version.sh` - Python version-specific testing

## 📊 Monitoring Configuration
- `monitoring/grafana_dashboard.json` - Grafana dashboard for refactored services

## 🔄 Migration Documentation

### Topic Migration
- Legacy: `system/trigger_telemetry`
- New: `system/gpio_trigger/health`
- Status: Dual publishing enabled

### Service Migration Status
| Service | Refactored | Health Topic |
|---------|------------|--------------|
| Camera Detector | ✅ | `system/camera_detector/health` |
| Fire Consensus | ✅ | `system/fire_consensus/health` |
| GPIO Trigger | ✅ | `system/gpio_trigger/health` |
| Security NVR | ❌ | Pending |

## 📝 Key Configuration Files

### Environment Variables
```bash
# GPIO Pin Updates
RESERVOIR_FLOAT_PIN=13  # Changed from 16
LINE_PRESSURE_PIN=19    # Changed from 20

# Reconnection Settings
MQTT_RECONNECT_MIN_DELAY=1.0
MQTT_RECONNECT_MAX_DELAY=60.0
MQTT_RECONNECT_MULTIPLIER=2.0
MQTT_RECONNECT_JITTER=0.3
```

### Docker Compose
- Production: `docker-compose.yml`
- Staging: `docker-compose.staging.yml`
- Rollback: `docker-compose.rollback.yml`

## 🎯 Quick Start Guides

### For Developers
1. Read [TEAM_TRAINING_GUIDE.md](./docs/TEAM_TRAINING_GUIDE.md)
2. Review [ADR-001](./docs/adr/ADR-001-service-refactoring.md) and [ADR-002](./docs/adr/ADR-002-mqtt-topic-naming.md)
3. Run `./scripts/run_tests_by_python_version.sh --all`

### For Operations
1. Review [OPERATIONAL_RUNBOOK.md](./docs/OPERATIONAL_RUNBOOK.md)
2. Print [EMERGENCY_PROCEDURES_CARD.md](./docs/EMERGENCY_PROCEDURES_CARD.md)
3. Configure monitoring with `grafana_dashboard.json`

### For Deployment
1. Complete [PRODUCTION_READINESS_CHECKLIST.md](./PRODUCTION_READINESS_CHECKLIST.md)
2. Follow [PRODUCTION_DEPLOYMENT_GUIDE.md](./PRODUCTION_DEPLOYMENT_GUIDE.md)
3. Run `./scripts/deploy_refactored_services.sh staging`

## 📞 Support Contacts

| Role | Contact | Area |
|------|---------|------|
| Development Lead | dev-lead@wildfire-watch.com | Technical questions |
| DevOps Engineer | devops@wildfire-watch.com | Deployment support |
| On-Call | +1-555-0911 | Emergency support |

## 🔍 Search Index

### By Topic
- **Deployment**: PRODUCTION_DEPLOYMENT_GUIDE, deploy_refactored_services.sh
- **Emergency**: EMERGENCY_PROCEDURES_CARD, OPERATIONAL_RUNBOOK
- **Monitoring**: grafana_dashboard.json, monitor_topic_migration.py
- **Testing**: DEPLOYMENT_TESTING_CHECKLIST, post_deployment_validation.py
- **Training**: TEAM_TRAINING_GUIDE, architecture decisions

### By Service
- **GPIO Trigger**: trigger_refactored.py, test_gpio_trigger_simple.py
- **Camera Detector**: Already refactored, see health topics
- **Fire Consensus**: Already refactored, see health topics
- **Security NVR**: Pending refactoring

---

**Last Updated**: July 4, 2025  
**Version**: 1.0  
**Next Review**: After production deployment