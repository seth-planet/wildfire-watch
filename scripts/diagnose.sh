#!/bin/bash
#===============================================================================
# Wildfire Watch Diagnostic Script
# Checks system health and reports issues
#===============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_PROJECT=${COMPOSE_PROJECT_NAME:-wildfire-watch}
REQUIRED_SERVICES=("mqtt_broker" "camera_detector" "fire_consensus" "gpio_trigger" "cam_telemetry")

echo "======================================"
echo "Wildfire Watch System Diagnostics"
echo "======================================"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print status
print_status() {
    local status=$1
    local message=$2
    if [ "$status" -eq 0 ]; then
        echo -e "${GREEN}[OK]${NC} $message"
    else
        echo -e "${RED}[FAIL]${NC} $message"
    fi
}

# Function to print warning
print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Function to print info
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# 1. Check Docker
echo "1. Docker Status"
echo "----------------"
if command_exists docker; then
    if docker info >/dev/null 2>&1; then
        print_status 0 "Docker daemon is running"
        docker_version=$(docker --version | awk '{print $3}' | sed 's/,$//')
        print_info "Docker version: $docker_version"
    else
        print_status 1 "Docker daemon is not running"
        exit 1
    fi
else
    print_status 1 "Docker is not installed"
    exit 1
fi

# 2. Check Docker Compose
echo ""
echo "2. Docker Compose Status"
echo "------------------------"
if command_exists docker-compose; then
    compose_version=$(docker-compose --version | awk '{print $3}' | sed 's/,$//')
    print_status 0 "Docker Compose installed"
    print_info "Version: $compose_version"
else
    print_status 1 "Docker Compose not installed"
fi

# 3. Check Services
echo ""
echo "3. Service Status"
echo "-----------------"
for service in "${REQUIRED_SERVICES[@]}"; do
    container_name="${COMPOSE_PROJECT}-${service}-1"
    if docker ps --format '{{.Names}}' | grep -q "^${container_name}$"; then
        # Get container status
        status=$(docker inspect -f '{{.State.Status}}' "$container_name" 2>/dev/null || echo "unknown")
        if [ "$status" = "running" ]; then
            print_status 0 "$service is running"
            # Check health if available
            health=$(docker inspect -f '{{.State.Health.Status}}' "$container_name" 2>/dev/null || echo "none")
            if [ "$health" != "none" ] && [ "$health" != "" ]; then
                if [ "$health" = "healthy" ]; then
                    print_info "  Health: $health"
                else
                    print_warning "  Health: $health"
                fi
            fi
        else
            print_status 1 "$service status: $status"
        fi
    else
        print_status 1 "$service not found"
    fi
done

# 4. Check MQTT Connectivity
echo ""
echo "4. MQTT Broker Connectivity"
echo "---------------------------"
if docker ps --format '{{.Names}}' | grep -q "mqtt_broker"; then
    # Try to connect to MQTT
    if nc -zv localhost 1883 >/dev/null 2>&1; then
        print_status 0 "MQTT port 1883 is accessible"
    else
        print_status 1 "Cannot connect to MQTT on port 1883"
    fi
    
    if [ "${MQTT_TLS:-false}" = "true" ]; then
        if nc -zv localhost 8883 >/dev/null 2>&1; then
            print_status 0 "MQTT TLS port 8883 is accessible"
        else
            print_status 1 "Cannot connect to MQTT TLS on port 8883"
        fi
    fi
else
    print_warning "MQTT broker not running, skipping connectivity test"
fi

# 5. Check Network
echo ""
echo "5. Network Configuration"
echo "------------------------"
# Check if wildfire network exists
if docker network ls --format '{{.Name}}' | grep -q "wildfire_net"; then
    print_status 0 "Docker network 'wildfire_net' exists"
    # Get network details
    subnet=$(docker network inspect wildfire_net -f '{{range .IPAM.Config}}{{.Subnet}}{{end}}' 2>/dev/null || echo "unknown")
    print_info "  Subnet: $subnet"
else
    print_status 1 "Docker network 'wildfire_net' not found"
fi

# 6. Check Volumes
echo ""
echo "6. Docker Volumes"
echo "-----------------"
volumes=("mqtt_data" "mqtt_logs" "frigate_data" "camera_data")
for volume in "${volumes[@]}"; do
    volume_name="${COMPOSE_PROJECT}_${volume}"
    if docker volume ls --format '{{.Name}}' | grep -q "^${volume_name}$"; then
        print_status 0 "Volume $volume exists"
        # Get volume size if possible
        mount_point=$(docker volume inspect "$volume_name" -f '{{.Mountpoint}}' 2>/dev/null || echo "")
        if [ -n "$mount_point" ] && [ -d "$mount_point" ]; then
            size=$(du -sh "$mount_point" 2>/dev/null | cut -f1 || echo "unknown")
            print_info "  Size: $size"
        fi
    else
        print_warning "Volume $volume not found"
    fi
done

# 7. Check Disk Space
echo ""
echo "7. Disk Space"
echo "-------------"
df -h / | tail -n +2 | while read -r line; do
    usage=$(echo "$line" | awk '{print $5}' | sed 's/%//')
    mount=$(echo "$line" | awk '{print $6}')
    if [ "$usage" -gt 90 ]; then
        print_status 1 "Root filesystem usage: ${usage}%"
    elif [ "$usage" -gt 80 ]; then
        print_warning "Root filesystem usage: ${usage}%"
    else
        print_status 0 "Root filesystem usage: ${usage}%"
    fi
done

# 8. Check Memory
echo ""
echo "8. Memory Usage"
echo "---------------"
if command_exists free; then
    mem_total=$(free -m | awk 'NR==2{print $2}')
    mem_used=$(free -m | awk 'NR==2{print $3}')
    mem_percent=$((mem_used * 100 / mem_total))
    
    if [ "$mem_percent" -gt 90 ]; then
        print_status 1 "Memory usage: ${mem_percent}% (${mem_used}MB/${mem_total}MB)"
    elif [ "$mem_percent" -gt 80 ]; then
        print_warning "Memory usage: ${mem_percent}% (${mem_used}MB/${mem_total}MB)"
    else
        print_status 0 "Memory usage: ${mem_percent}% (${mem_used}MB/${mem_total}MB)"
    fi
fi

# 9. Check GPIO (if on Raspberry Pi)
echo ""
echo "9. GPIO Status"
echo "--------------"
if [ -e /sys/class/gpio ]; then
    print_status 0 "GPIO subsystem available"
    # Check if we can access GPIO
    if [ -w /sys/class/gpio/export ]; then
        print_info "GPIO access: Available"
    else
        print_warning "GPIO access: Requires root/gpio group membership"
    fi
else
    print_info "GPIO not available (not on Raspberry Pi or GPIO not enabled)"
fi

# 10. Check Cameras
echo ""
echo "10. Camera Detection"
echo "--------------------"
# Check if camera detector found any cameras
if docker ps --format '{{.Names}}' | grep -q "camera_detector"; then
    # Try to get camera count from logs
    camera_count=$(docker logs "${COMPOSE_PROJECT}-camera_detector-1" 2>&1 | grep -c "Camera discovered" || echo "0")
    if [ "$camera_count" -gt 0 ]; then
        print_status 0 "Cameras discovered: $camera_count"
    else
        print_warning "No cameras discovered yet"
    fi
else
    print_warning "Camera detector not running"
fi

# 11. Check AI Accelerators
echo ""
echo "11. AI Accelerators"
echo "-------------------"
# Check for Coral TPU
if [ -e /dev/bus/usb ] && lsusb 2>/dev/null | grep -q "Google.*Coral"; then
    print_status 0 "Coral TPU detected"
elif [ -e /dev/apex_0 ]; then
    print_status 0 "Coral PCIe TPU detected"
else
    print_info "No Coral TPU detected"
fi

# Check for Hailo
if [ -e /dev/hailo0 ]; then
    print_status 0 "Hailo-8 detected"
else
    print_info "No Hailo accelerator detected"
fi

# Check for NVIDIA GPU
if command_exists nvidia-smi && nvidia-smi >/dev/null 2>&1; then
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    print_status 0 "NVIDIA GPU detected: $gpu_name"
else
    print_info "No NVIDIA GPU detected"
fi

# 12. Recent Errors
echo ""
echo "12. Recent Errors (last 10)"
echo "---------------------------"
for service in "${REQUIRED_SERVICES[@]}"; do
    container_name="${COMPOSE_PROJECT}-${service}-1"
    if docker ps -a --format '{{.Names}}' | grep -q "^${container_name}$"; then
        error_count=$(docker logs "$container_name" 2>&1 | grep -iE "(error|exception|failed)" | tail -10 | wc -l)
        if [ "$error_count" -gt 0 ]; then
            print_warning "$service has $error_count recent errors"
            docker logs "$container_name" 2>&1 | grep -iE "(error|exception|failed)" | tail -3 | sed 's/^/  /'
        fi
    fi
done

echo ""
echo "======================================"
echo "Diagnostic complete"
echo "======================================"

# Exit with error if any critical issues found
if docker ps --format '{{.Names}}' | grep -q "mqtt_broker"; then
    exit 0
else
    echo ""
    echo "Critical issue: Core services not running"
    echo "Run 'docker-compose up -d' to start services"
    exit 1
fi