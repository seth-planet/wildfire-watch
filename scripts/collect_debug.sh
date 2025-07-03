#!/bin/bash
#===============================================================================
# Wildfire Watch Debug Collection Script
# Collects logs, configuration, and system information for troubleshooting
#===============================================================================

set -euo pipefail

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DEBUG_DIR="/tmp/wildfire_debug_${TIMESTAMP}"
ARCHIVE_NAME="wildfire_debug_${TIMESTAMP}.tar.gz"
COMPOSE_PROJECT=${COMPOSE_PROJECT_NAME:-wildfire-watch}

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "======================================"
echo "Wildfire Watch Debug Collection"
echo "======================================"
echo "Collecting debug information..."
echo ""

# Create debug directory
mkdir -p "$DEBUG_DIR"/{logs,config,system}

# Function to safely copy files
safe_copy() {
    local src=$1
    local dst=$2
    if [ -e "$src" ]; then
        cp -r "$src" "$dst" 2>/dev/null || echo "Could not copy $src"
    fi
}

# 1. Collect Docker logs
echo -e "${BLUE}[1/7]${NC} Collecting Docker logs..."
for service in mqtt_broker camera_detector fire_consensus gpio_trigger cam_telemetry security_nvr; do
    container="${COMPOSE_PROJECT}-${service}-1"
    if docker ps -a --format '{{.Names}}' | grep -q "^${container}$"; then
        echo "  - $service logs"
        docker logs "$container" >"$DEBUG_DIR/logs/${service}.log" 2>&1
        docker inspect "$container" >"$DEBUG_DIR/logs/${service}_inspect.json" 2>&1
    fi
done

# 2. Collect configuration files
echo -e "${BLUE}[2/7]${NC} Collecting configuration files..."
safe_copy ".env" "$DEBUG_DIR/config/"
safe_copy ".env.example" "$DEBUG_DIR/config/"
safe_copy "docker-compose.yml" "$DEBUG_DIR/config/"
safe_copy "docker-compose.override.yml" "$DEBUG_DIR/config/"
safe_copy "mqtt_broker/mosquitto.conf" "$DEBUG_DIR/config/"
safe_copy "security_nvr/nvr_base_config.yml" "$DEBUG_DIR/config/"

# Sanitize sensitive information
if [ -f "$DEBUG_DIR/config/.env" ]; then
    echo "  - Sanitizing .env file"
    sed -i 's/\(PASSWORD=\).*/\1<REDACTED>/' "$DEBUG_DIR/config/.env"
    sed -i 's/\(CREDENTIALS=\).*/\1<REDACTED>/' "$DEBUG_DIR/config/.env"
fi

# 3. Collect system information
echo -e "${BLUE}[3/7]${NC} Collecting system information..."
{
    echo "=== System Information ==="
    echo "Date: $(date)"
    echo "Hostname: $(hostname)"
    echo "Kernel: $(uname -a)"
    echo ""
    
    echo "=== OS Information ==="
    if [ -f /etc/os-release ]; then
        cat /etc/os-release
    fi
    echo ""
    
    echo "=== CPU Information ==="
    if [ -f /proc/cpuinfo ]; then
        grep -E "(model name|processor)" /proc/cpuinfo | sort -u
        echo "CPU Count: $(nproc)"
    fi
    echo ""
    
    echo "=== Memory Information ==="
    free -h
    echo ""
    
    echo "=== Disk Usage ==="
    df -h
    echo ""
    
    echo "=== Docker Information ==="
    docker --version
    docker-compose --version
    docker info
    echo ""
    
    echo "=== Python Versions ==="
    if command -v python3.12 >/dev/null 2>&1; then
        echo "Python 3.12: $(python3.12 --version)"
    fi
    if command -v python3.8 >/dev/null 2>&1; then
        echo "Python 3.8: $(python3.8 --version)"
    fi
    echo ""
    
} >"$DEBUG_DIR/system/system_info.txt" 2>&1

# 4. Collect Docker information
echo -e "${BLUE}[4/7]${NC} Collecting Docker information..."
{
    echo "=== Docker Containers ==="
    docker ps -a
    echo ""
    
    echo "=== Docker Images ==="
    docker images
    echo ""
    
    echo "=== Docker Networks ==="
    docker network ls
    echo ""
    
    echo "=== Docker Volumes ==="
    docker volume ls
    echo ""
    
    echo "=== Docker Compose Status ==="
    docker-compose ps
    echo ""
    
} >"$DEBUG_DIR/system/docker_info.txt" 2>&1

# 5. Collect network information
echo -e "${BLUE}[5/7]${NC} Collecting network information..."
{
    echo "=== Network Interfaces ==="
    ip addr show
    echo ""
    
    echo "=== Routing Table ==="
    ip route show
    echo ""
    
    echo "=== Listening Ports ==="
    if command -v ss >/dev/null 2>&1; then
        ss -tlnp 2>/dev/null | grep -E "(1883|8883|5000|8554)" || true
    elif command -v netstat >/dev/null 2>&1; then
        netstat -tlnp 2>/dev/null | grep -E "(1883|8883|5000|8554)" || true
    fi
    echo ""
    
    echo "=== MQTT Connectivity Test ==="
    if nc -zv localhost 1883 >/dev/null 2>&1; then
        echo "MQTT port 1883: OPEN"
    else
        echo "MQTT port 1883: CLOSED"
    fi
    echo ""
    
} >"$DEBUG_DIR/system/network_info.txt" 2>&1

# 6. Collect GPIO information (if available)
echo -e "${BLUE}[6/7]${NC} Collecting GPIO information..."
if [ -e /sys/class/gpio ]; then
    {
        echo "=== GPIO Exports ==="
        ls -la /sys/class/gpio/ 2>/dev/null || echo "No GPIO exports"
        echo ""
        
        echo "=== GPIO Chip Info ==="
        if command -v gpioinfo >/dev/null 2>&1; then
            gpioinfo
        else
            echo "gpioinfo not available"
        fi
        echo ""
        
    } >"$DEBUG_DIR/system/gpio_info.txt" 2>&1
else
    echo "GPIO not available" >"$DEBUG_DIR/system/gpio_info.txt"
fi

# 7. Check for AI accelerators
echo -e "${BLUE}[7/7]${NC} Checking AI accelerators..."
{
    echo "=== USB Devices ==="
    if command -v lsusb >/dev/null 2>&1; then
        lsusb | grep -E "(Coral|Google|Hailo)" || echo "No AI accelerators found on USB"
    fi
    echo ""
    
    echo "=== PCIe Devices ==="
    if command -v lspci >/dev/null 2>&1; then
        lspci | grep -E "(Coral|Hailo|NVIDIA)" || echo "No AI accelerators found on PCIe"
    fi
    echo ""
    
    echo "=== Device Files ==="
    ls -la /dev/{apex_0,hailo0,nvidia*} 2>/dev/null || echo "No AI device files found"
    echo ""
    
} >"$DEBUG_DIR/system/ai_accelerators.txt" 2>&1

# Run diagnostics
echo ""
echo "Running diagnostics..."
"$(dirname "$0")/diagnose.sh" >"$DEBUG_DIR/diagnose_output.txt" 2>&1 || true

# Create archive
echo ""
echo "Creating archive..."
cd /tmp
tar -czf "$ARCHIVE_NAME" "wildfire_debug_${TIMESTAMP}"

# Cleanup
rm -rf "$DEBUG_DIR"

# Final output
echo ""
echo -e "${GREEN}Debug collection complete!${NC}"
echo "Archive created: /tmp/$ARCHIVE_NAME"
echo ""
echo "This file contains:"
echo "  - Docker container logs"
echo "  - Configuration files (sanitized)"
echo "  - System information"
echo "  - Network configuration"
echo "  - Diagnostic output"
echo ""
echo "Share this file when reporting issues."
echo "Note: Passwords and credentials have been redacted."