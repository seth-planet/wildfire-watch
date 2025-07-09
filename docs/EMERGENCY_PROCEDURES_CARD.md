# 🚨 WILDFIRE WATCH - EMERGENCY PROCEDURES 🚨

## IMMEDIATE ACTIONS

### 🔴 PUMP RUNNING TOO LONG
```bash
mosquitto_pub -h localhost -t "fire/emergency" \
  -m '{"command": "stop"}' -q 2
```

### 🔴 SYSTEM MALFUNCTION
```bash
mosquitto_pub -h localhost -t "fire/emergency" \
  -m '{"command": "reset"}' -q 2
```

### 🔴 SERVICE CRASH LOOP
```bash
docker-compose stop <service_name>
# Then investigate logs before restart
```

---

## VERIFICATION COMMANDS

### Check Pump State
```bash
mosquitto_sub -h localhost \
  -t "system/gpio_trigger/health" -C 1 | jq .state
```
Expected: `"IDLE"` after emergency stop

### Check All Services
```bash
docker-compose ps
```
All should show "Up" status

### Monitor Errors
```bash
docker-compose logs --tail=50 | grep -i error
```

---

## CRITICAL CONTACTS

| Role | Contact | When to Call |
|------|---------|--------------|
| On-Call Engineer | +1-555-0911 | System down, pump won't stop |
| Team Lead | +1-555-0112 | Major service failure |
| Facilities | +1-555-0113 | Water leak, electrical issue |

---

## POST-EMERGENCY CHECKLIST

1. ✓ Verify pump stopped
2. ✓ Check water pressure
3. ✓ Review error logs
4. ✓ Document incident
5. ✓ Clear error states
6. ✓ Test before restart

---

## MANUAL OVERRIDE (LAST RESORT)

### Software Override Failed
```bash
# 1. Stop GPIO service completely
docker-compose stop gpio_trigger

# 2. Force container removal
docker rm -f gpio_trigger

# 3. Physical pump shutoff required
```

### Network Issues
```bash
# Direct container access
docker exec -it gpio_trigger /bin/bash
# Then use internal commands
```

---

**KEEP THIS CARD ACCESSIBLE AT ALL TIMES**

Print copies for:
- Control room
- Server room  
- Maintenance shop
- On-call binder

Version: 1.0 | Updated: 2025-07-04