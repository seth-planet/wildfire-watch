# Multi-Node Deployment Guide

Scale Wildfire Watch across large properties with multiple detection nodes, centralized monitoring, and redundant systems.

## Architecture Overview

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│   Edge Node 1   │       │   Edge Node 2   │       │   Edge Node 3   │
│  (North Field)  │       │  (East Ridge)   │       │  (South Valley) │
│  Pi 5 + Hailo   │       │  Pi 5 + Hailo   │       │  Pi 5 + Hailo   │
│  4 Cameras      │       │  4 Cameras      │       │  4 Cameras      │
└────────┬────────┘       └────────┬────────┘       └────────┬────────┘
         │                         │                         │
         └─────────────────────────┴─────────────────────────┘
                                   │
                          ┌────────┴────────┐
                          │  Central Node   │
                          │  Intel NUC      │
                          │  Coordination   │
                          │  Monitoring     │
                          └─────────────────┘
```

## Deployment Patterns

### 1. Distributed Edge (Recommended)
- **Use Case:** Large properties, multiple buildings
- **Nodes:** 3-10 edge devices
- **Benefits:** Low latency, fault tolerance
- **Challenges:** Complex coordination

### 2. Centralized Processing
- **Use Case:** High camera density, single location
- **Nodes:** 1 powerful server, multiple camera nodes
- **Benefits:** Easier management, shared resources
- **Challenges:** Single point of failure

### 3. Hybrid Hierarchical
- **Use Case:** Campus or industrial sites
- **Nodes:** Zone controllers + central coordinator
- **Benefits:** Balanced approach
- **Challenges:** Higher complexity

## Node Configuration

### Edge Node Setup

```bash
# /etc/wildfire/node.conf
NODE_ID=edge-north-01
NODE_ROLE=detector
CENTRAL_BROKER=central.local:8883
LOCAL_BROKER=localhost:1883
HEARTBEAT_INTERVAL=10
```

### Central Node Setup

```bash
# /etc/wildfire/central.conf
NODE_ID=central-01
NODE_ROLE=coordinator
ENABLE_BRIDGING=true
CONSENSUS_MODE=global
MONITORING_ENABLED=true
```

## MQTT Bridging

### Edge to Central Bridge

```conf
# Edge node mosquitto.conf
connection central
address central.local:8883
topic frigate/fire/# out 2
topic consensus/vote/# in 2
topic gpio/pump/# in 2
bridge_cafile /mosquitto/certs/ca.crt
bridge_certfile /mosquitto/certs/edge-north.crt
bridge_keyfile /mosquitto/certs/edge-north.key
```

### Central Broker Configuration

```conf
# Central mosquitto.conf
listener 8883
require_certificate true
use_identity_as_username true

# Per-node isolation
listener 8884
protocol websockets
```

## Network Architecture

### VLAN Design
```
VLAN 10: Management (Nodes, SSH)
VLAN 20: Camera North
VLAN 21: Camera East  
VLAN 22: Camera South
VLAN 30: Pump Control
VLAN 99: Central Coordination
```

### Routing Configuration

```bash
# Enable routing between VLANs
sudo sysctl net.ipv4.ip_forward=1

# Zone-based firewall rules
sudo iptables -A FORWARD -i vlan20 -o vlan99 -p tcp --dport 8883 -j ACCEPT
sudo iptables -A FORWARD -i vlan99 -o vlan30 -p tcp --dport 1883 -j ACCEPT
```

### Bandwidth Requirements

| Link Type | Bandwidth | Latency |
|-----------|-----------|---------|
| Camera → Edge | 5 Mbps per camera | <10ms |
| Edge → Central | 1 Mbps average | <50ms |
| Central → Pump | 100 Kbps | <100ms |
| Total WAN | 10 Mbps | <100ms |

## Consensus Across Nodes

### Global Fire Consensus

```python
# Modified consensus logic for multi-node
class GlobalConsensus:
    def __init__(self):
        self.node_votes = {}
        self.camera_locations = self.load_camera_map()
    
    def process_detection(self, node_id, camera_id, confidence):
        # Weight by node reliability and camera overlap
        weight = self.calculate_weight(node_id, camera_id)
        
        # Require detections from multiple nodes
        if self.cross_node_validation(camera_id):
            self.trigger_pumps(self.affected_zones())
```

### Zone-Based Activation

```yaml
# Zone configuration
zones:
  north:
    nodes: [edge-north-01, edge-north-02]
    pumps: [pump-01, pump-02]
    overlap: [east]
  
  east:
    nodes: [edge-east-01]
    pumps: [pump-03]
    overlap: [north, south]
```

## Redundancy & Failover

### Node Health Monitoring

```bash
# Health check script
#!/bin/bash
mosquitto_pub -t "health/$NODE_ID/status" -m "$(date +%s)"

# Central monitoring
mosquitto_sub -t "health/+/status" | while read msg; do
  check_node_timeout "$msg"
done
```

### Automatic Failover

```yaml
# Docker Swarm deployment
version: '3.8'
services:
  fire_consensus:
    image: wildfire/consensus:latest
    deploy:
      replicas: 2
      placement:
        constraints:
          - node.labels.role == edge
      restart_policy:
        condition: any
```

### Split-Brain Prevention

```python
# Quorum-based consensus
MIN_NODES_FOR_CONSENSUS = 2
QUORUM_PERCENTAGE = 0.51

def has_quorum(active_nodes, total_nodes):
    return active_nodes >= max(MIN_NODES_FOR_CONSENSUS, 
                               int(total_nodes * QUORUM_PERCENTAGE))
```

## Deployment Tools

### Ansible Playbook

```yaml
---
- name: Deploy Wildfire Watch
  hosts: all
  roles:
    - common
    - docker
    - certificates
  
- name: Configure Edge Nodes
  hosts: edge
  roles:
    - camera_detector
    - frigate
    - mqtt_bridge
  
- name: Configure Central
  hosts: central
  roles:
    - mqtt_broker
    - consensus_coordinator
    - monitoring
```

### Balena Fleet Configuration

```yaml
# balena.yml
name: Wildfire Watch Fleet
type: sw.application
data:
  applicationEnvironmentVariables:
    - CONSENSUS_MODE: distributed
    - FLEET_COORDINATION: enabled
  applicationConfigVariables:
    - BALENA_HOST_CONFIG_gpu_mem: 128
```

## Monitoring & Observability

### Metrics Collection

```yaml
# Prometheus configuration
scrape_configs:
  - job_name: 'wildfire-nodes'
    static_configs:
      - targets:
        - edge-north-01:9090
        - edge-east-01:9090
        - central-01:9090
```

### Grafana Dashboard

Key metrics to monitor:
- Detections per node/camera
- Consensus participation rate
- Node latency and health (via [Camera Telemetry](../cam_telemetry/README.md))
- Pump activation history
- Network bandwidth usage
- System metrics (CPU, memory, disk) from telemetry service

### Alerting Rules

```yaml
# Alertmanager rules
groups:
  - name: wildfire
    rules:
      - alert: NodeDown
        expr: up{job="wildfire-nodes"} == 0
        for: 2m
        
      - alert: ConsensusTimeout
        expr: consensus_pending_duration > 60
        
      - alert: PumpRuntime
        expr: pump_runtime_seconds > 1200
```

## Health Monitoring

The [Camera Telemetry](../cam_telemetry/README.md) service provides real-time health monitoring for all nodes:

```bash
# Monitor all node telemetry
mosquitto_sub -h central-broker -t "system/telemetry/+" -v

# Watch for offline nodes (Last Will Testament)
mosquitto_sub -h central-broker -t "system/telemetry/+/lwt" -v
```

Telemetry data includes:
- System metrics (CPU, memory, disk usage)
- Camera status and configuration
- Node uptime and availability
- Detection backend in use

## Scaling Considerations

### Horizontal Scaling

```bash
# Add new edge node
./scripts/provision_node.sh edge-west-01 192.168.40.10

# Register with central
mosquitto_pub -t "nodes/register" -m '{"id":"edge-west-01","role":"detector"}'
```

### Vertical Scaling

Upgrade paths:
1. **More cameras per node:** Add Coral/Hailo accelerator
2. **Faster consensus:** Upgrade central node CPU
3. **More history:** Add NAS for recordings

### Geographic Distribution

For very large deployments:
- Regional coordinators
- Satellite/cellular backhaul
- Edge caching of models
- Autonomous zone operation

## Best Practices

### Network Design
1. Dedicate bandwidth for critical paths
2. Use multicast for camera discovery
3. Implement QoS for fire events
4. Plan for 50% growth

### Reliability
1. Test failover monthly
2. Stagger updates across nodes
3. Maintain spare hardware
4. Document cable runs

### Security
1. Unique certificates per node
2. Rotate keys quarterly
3. Isolate node networks
4. Central audit logging

## Example Deployments

### 100-Acre Ranch
- 3 edge nodes (Pi 5 + Hailo)
- 12 cameras total
- 3 pump zones
- Cellular backhaul

### Industrial Complex
- 8 edge nodes (Intel NUC)
- 32 cameras
- Integrated with existing systems
- Redundant central nodes

### Municipality
- 20+ edge nodes
- 100+ cameras
- Integration with dispatch
- Public safety compliance

## Troubleshooting Multi-Node

### Common Issues

**Nodes not syncing:**
```bash
# Check bridge connection
mosquitto_sub -v -t '$SYS/broker/connection/+/state'

# Verify certificates
openssl s_client -connect central.local:8883 -cert edge.crt -key edge.key
```

**Consensus delays:**
```bash
# Check network latency
ping -c 100 central.local | grep avg

# Monitor consensus timing
mosquitto_sub -t 'consensus/timing/#' -v
```

**Split brain scenario:**
```bash
# Force leader election
mosquitto_pub -t 'consensus/election/force' -m "$(date +%s)"

# Check quorum status
curl http://central:8080/api/consensus/quorum
```
