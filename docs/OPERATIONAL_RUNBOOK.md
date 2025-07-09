# Wildfire Watch - Operational Runbook

## Overview
This runbook provides step-by-step procedures for common operational scenarios with the refactored Wildfire Watch system.

## Quick Reference

| Scenario | Page |
|----------|------|
| Service Not Reporting Health | [Link](#service-not-reporting-health) |
| MQTT Connection Issues | [Link](#mqtt-connection-issues) |
| GPIO Pin Conflicts | [Link](#gpio-pin-conflicts) |
| Fire Detection Not Working | [Link](#fire-detection-not-working) |
| Emergency Pump Shutdown | [Link](#emergency-pump-shutdown) |
| Topic Migration Issues | [Link](#topic-migration-issues) |

---

## Service Not Reporting Health

### Symptoms
- Missing health messages in monitoring
- Service shows as OFFLINE in dashboard
- No messages on `system/{service}/health` topic

### Diagnosis Steps

1. **Check Service Status**
   ```bash
   docker-compose ps
   ```
   Expected: Service should show as "Up"

2. **Check Recent Logs**
   ```bash
   docker-compose logs --tail=100 <service_name>
   ```
   Look for: Connection errors, crashes, configuration issues

3. **Verify MQTT Connection**
   ```bash
   docker-compose logs <service_name> | grep -i "mqtt"
   ```
   Look for: "Connected to MQTT broker" message

4. **Check Health Topic**
   ```bash
   mosquitto_sub -h localhost -t "system/+/health" -v
   ```
   Wait 60 seconds for health messages

### Resolution Steps

1. **Restart Service**
   ```bash
   docker-compose restart <service_name>
   ```

2. **Check Environment Variables**
   ```bash
   docker-compose config | grep -A 20 <service_name>
   ```
   Verify: MQTT_BROKER, MQTT_PORT, HEALTH_REPORT_INTERVAL

3. **Force Recreation**
   ```bash
   docker-compose up -d --force-recreate <service_name>
   ```

4. **Check Network Connectivity**
   ```bash
   docker exec <service_name> ping mqtt_broker
   ```

---

## MQTT Connection Issues

### Symptoms
- Multiple reconnection attempts in logs
- "Connection refused" errors
- Services cycling between online/offline

### Diagnosis Steps

1. **Check Broker Status**
   ```bash
   docker-compose ps mqtt_broker
   nc -zv localhost 1883
   ```

2. **Monitor Connection Attempts**
   ```bash
   mosquitto_sub -h localhost -t "system/+/lwt" -v
   ```

3. **Check Broker Logs**
   ```bash
   docker-compose logs --tail=200 mqtt_broker | grep -i "error\|disconnect"
   ```

### Resolution Steps

1. **Restart MQTT Broker** (WARNING: Causes temporary outage)
   ```bash
   docker-compose restart mqtt_broker
   ```

2. **Check Broker Configuration**
   ```bash
   cat mqtt_broker/mosquitto.conf
   # Verify listener and authentication settings
   ```

3. **Test Manual Connection**
   ```bash
   mosquitto_pub -h localhost -t test -m "test"
   mosquitto_sub -h localhost -t test -C 1
   ```

4. **Scale Issues - Increase Connection Limits**
   ```conf
   # mosquitto.conf
   max_connections 1000
   ```

---

## GPIO Pin Conflicts

### Symptoms
- "Duplicate GPIO pin assignments detected" error
- GPIO operations failing
- Incorrect hardware behavior

### Diagnosis Steps

1. **Check Pin Assignments**
   ```bash
   grep "_PIN=" .env* | sort -t= -k2 -n
   ```

2. **Identify Duplicates**
   ```bash
   grep "_PIN=" .env | awk -F= '{print $2}' | sort | uniq -d
   ```

3. **Verify Running Configuration**
   ```bash
   docker exec gpio_trigger env | grep "_PIN"
   ```

### Resolution Steps

1. **Update Pin Assignments**
   ```bash
   # Edit .env file
   vim .env
   
   # Common safe pins:
   # RESERVOIR_FLOAT_PIN=13  (was 16)
   # LINE_PRESSURE_PIN=19    (was 20)
   ```

2. **Restart GPIO Trigger**
   ```bash
   docker-compose restart gpio_trigger
   ```

3. **Verify No Conflicts**
   ```bash
   docker-compose logs gpio_trigger | grep -i "gpio\|pin"
   ```

---

## Fire Detection Not Working

### Symptoms
- No fire triggers despite actual fire
- Consensus not reached
- Detection confidence too low

### Diagnosis Steps

1. **Check Camera Detection**
   ```bash
   mosquitto_sub -h localhost -t "frigate/+/fire" -t "frigate/+/smoke" -v
   ```

2. **Verify Consensus Service**
   ```bash
   mosquitto_sub -h localhost -t "fire/trigger" -v
   docker-compose logs fire_consensus | tail -50
   ```

3. **Check Thresholds**
   ```bash
   docker exec fire_consensus env | grep -E "CONSENSUS_THRESHOLD|MIN_CONFIDENCE"
   ```

### Resolution Steps

1. **Lower Detection Thresholds** (Temporary)
   ```bash
   # For testing only!
   docker-compose exec fire_consensus sh -c 'export MIN_CONFIDENCE=0.5'
   ```

2. **Check Camera Status**
   ```bash
   mosquitto_sub -h localhost -t "cameras/discovered" -C 1
   ```

3. **Verify AI Model Loading**
   ```bash
   docker-compose logs security_nvr | grep -i "model\|detector"
   ```

4. **Manual Fire Trigger Test**
   ```bash
   mosquitto_pub -h localhost -t "fire/trigger" \
     -m '{"action": "activate"}' -q 1
   ```

---

## Emergency Pump Shutdown

### Symptoms
- Pump running beyond safe limits
- Low water pressure detected
- System malfunction requiring immediate stop

### Emergency Procedures

1. **Immediate Shutdown**
   ```bash
   # Send emergency stop command
   mosquitto_pub -h localhost -t "fire/emergency" \
     -m '{"command": "stop"}' -q 2
   ```

2. **Verify Pump Stopped**
   ```bash
   mosquitto_sub -h localhost -t "system/gpio_trigger/health" -C 1 | jq .state
   # Should show "IDLE" or "ERROR"
   ```

3. **Force System Reset**
   ```bash
   mosquitto_pub -h localhost -t "fire/emergency" \
     -m '{"command": "reset"}' -q 2
   ```

4. **Physical Emergency Stop**
   ```bash
   # If software fails, restart service
   docker-compose restart gpio_trigger
   
   # Last resort - stop container
   docker-compose stop gpio_trigger
   ```

### Post-Emergency Checklist

- [ ] Verify pump is physically stopped
- [ ] Check water levels
- [ ] Inspect for leaks or damage
- [ ] Review logs for root cause
- [ ] Clear error states before restart
- [ ] Test safety systems

---

## Topic Migration Issues

### Symptoms
- Some services using old topics
- Missing data in monitoring
- Duplicate messages on multiple topics

### Diagnosis Steps

1. **Identify Topic Usage**
   ```bash
   # Monitor all health topics
   mosquitto_sub -h localhost \
     -t "system/+/health" \
     -t "system/trigger_telemetry" -v
   ```

2. **Check Service Versions**
   ```bash
   docker-compose images
   ```

3. **Verify Topic Configuration**
   ```bash
   docker exec <service> env | grep TOPIC
   ```

### Migration Steps

1. **Enable Dual Publishing** (GPIO Trigger)
   ```bash
   # Already enabled in refactored version
   # Publishes to both topics during migration
   ```

2. **Update Consumers**
   ```python
   # Update monitoring scripts
   topics = [
       'system/gpio_trigger/health',    # New
       'system/trigger_telemetry'       # Legacy
   ]
   ```

3. **Verify Both Topics Active**
   ```bash
   # Should see messages on both
   mosquitto_sub -h localhost -t "system/trigger_telemetry" -C 1 &
   mosquitto_sub -h localhost -t "system/gpio_trigger/health" -C 1
   ```

4. **Deprecate Legacy Topic** (After 30 days)
   ```bash
   # Remove legacy topic from monitoring
   # Update all consumers to use new topic only
   ```

---

## Performance Troubleshooting

### High CPU Usage

1. **Identify High CPU Service**
   ```bash
   docker stats --no-stream
   ```

2. **Check Message Rates**
   ```bash
   mosquitto_sub -h localhost -t "#" -v | pv -l -i 10
   ```

3. **Review Logs for Loops**
   ```bash
   docker-compose logs <service> | grep -c "ERROR\|WARNING"
   ```

### Memory Leaks

1. **Monitor Memory Growth**
   ```bash
   docker stats <service> --no-stream --format "table {{.MemUsage}}"
   ```

2. **Check Queue Sizes**
   ```bash
   docker exec <service> sh -c 'ps aux | grep python'
   ```

3. **Force Garbage Collection**
   ```bash
   docker-compose restart <service>
   ```

---

## Monitoring and Alerting

### Key Metrics to Monitor

1. **Service Health**
   - Health message frequency (should be < 2 minutes)
   - LWT status (should be "online")
   - Error count in health payload

2. **MQTT Broker**
   - Connection count
   - Message rate
   - Subscription count

3. **GPIO Trigger**
   - Pump state
   - Total runtime
   - Refill status

### Alert Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| Health Message Gap | > 3 min | > 5 min |
| MQTT Reconnections | > 5/hour | > 20/hour |
| Error Rate | > 1% | > 5% |
| Pump Runtime | > 25 min | > 30 min |

### Log Aggregation Queries

```bash
# Find all errors
docker-compose logs --no-color | grep -i "error\|exception" | tail -50

# Connection issues
docker-compose logs --no-color | grep -i "disconnect\|reconnect" | tail -50

# State changes
docker-compose logs --no-color | grep -i "state.*change\|transition" | tail -50
```

---

## Contact Information

### Escalation Matrix

| Severity | Contact | Response Time |
|----------|---------|--------------|
| P1 - Fire Active, System Down | On-Call Engineer | < 5 minutes |
| P2 - Service Degraded | Team Lead | < 30 minutes |
| P3 - Non-Critical Issue | Dev Team | < 4 hours |
| P4 - Enhancement | Product Owner | Next Sprint |

### Support Channels
- Slack: `#wildfire-watch-ops`
- PagerDuty: `wildfire-watch-oncall`
- Email: `wildfire-ops@example.com`

---

**Last Updated**: July 4, 2025  
**Version**: 1.0  
**Next Review**: August 4, 2025