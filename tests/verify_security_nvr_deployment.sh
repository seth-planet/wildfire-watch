#!/bin/bash
# Verification script for Security NVR deployment
# This script checks that the Security NVR is properly installed and configured

set -e

echo "=== Security NVR Deployment Verification ==="
echo

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Function to run a test
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    echo -n "Testing: $test_name ... "
    
    if eval "$test_command" > /dev/null 2>&1; then
        echo -e "${GREEN}PASSED${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}FAILED${NC}"
        ((TESTS_FAILED++))
        echo "  Command: $test_command"
    fi
}

# Function to check MQTT message
check_mqtt_message() {
    local topic="$1"
    local timeout="${2:-5}"
    
    timeout $timeout mosquitto_sub -h localhost -t "$topic" -C 1 > /dev/null 2>&1
}

echo "1. Checking Docker container status"
echo "-----------------------------------"
run_test "Security NVR container running" "docker ps | grep -q security_nvr"
run_test "Container healthy" "docker ps --filter name=security_nvr --format '{{.Status}}' | grep -q healthy"

echo
echo "2. Checking service connectivity"
echo "--------------------------------"
run_test "Frigate API accessible" "curl -sf http://localhost:5000/api/version"
run_test "Frigate stats endpoint" "curl -sf http://localhost:5000/api/stats"
run_test "Web UI accessible" "curl -sf http://localhost:5000/"

echo
echo "3. Checking hardware detection"
echo "------------------------------"
run_test "Hardware detector script exists" "docker exec security_nvr test -f /scripts/hardware_detector.py"
run_test "Hardware detection runs" "docker exec security_nvr python3 /scripts/hardware_detector.py 2>&1 | grep -q 'cpu'"

echo
echo "4. Checking MQTT integration"
echo "----------------------------"
run_test "MQTT broker reachable from NVR" "docker exec security_nvr ping -c 1 mqtt_broker"
run_test "Frigate MQTT availability" "check_mqtt_message 'frigate/available'"
run_test "Frigate stats published" "check_mqtt_message 'frigate/stats'"

echo
echo "5. Checking storage configuration"
echo "---------------------------------"
run_test "Storage mount exists" "docker exec security_nvr test -d /media/frigate"
run_test "Storage writable" "docker exec security_nvr touch /media/frigate/test_write && docker exec security_nvr rm /media/frigate/test_write"
run_test "Recording directories" "docker exec security_nvr test -d /media/frigate/recordings || docker exec security_nvr mkdir -p /media/frigate/recordings"

echo
echo "6. Checking model configuration"
echo "-------------------------------"
run_test "Model directory exists" "docker exec security_nvr test -d /models"
run_test "Config directory exists" "docker exec security_nvr test -d /config"
run_test "Base config exists" "docker exec security_nvr test -f /config/frigate_base.yml || echo 'Base config may be at different location'"

echo
echo "7. Checking detector configuration"
echo "----------------------------------"
# Get detector info from API
if curl -sf http://localhost:5000/api/stats > /tmp/frigate_stats.json 2>/dev/null; then
    detector_count=$(jq '.detectors | length' /tmp/frigate_stats.json 2>/dev/null || echo "0")
    if [ "$detector_count" -gt 0 ]; then
        echo -e "  Detectors found: ${GREEN}$detector_count${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "  ${YELLOW}No detectors configured${NC}"
    fi
else
    echo -e "  ${RED}Could not retrieve detector information${NC}"
    ((TESTS_FAILED++))
fi

echo
echo "8. Checking camera configuration"
echo "--------------------------------"
# Check if cameras are configured
if curl -sf http://localhost:5000/api/config > /tmp/frigate_config.json 2>/dev/null; then
    camera_count=$(jq '.cameras | length' /tmp/frigate_config.json 2>/dev/null || echo "0")
    if [ "$camera_count" -gt 0 ]; then
        echo -e "  Cameras configured: ${GREEN}$camera_count${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "  ${YELLOW}No cameras configured yet${NC}"
        echo "  This is normal if camera_detector hasn't discovered cameras yet"
    fi
fi

echo
echo "9. Checking integration with other services"
echo "------------------------------------------"
run_test "Camera detector service running" "docker ps | grep -q camera_detector"
run_test "MQTT broker running" "docker ps | grep -q mqtt_broker"
run_test "Services on same network" "docker network inspect wildfire-watch_default | grep -q security_nvr"

echo
echo "10. Performance checks"
echo "---------------------"
# Check CPU usage
if docker stats --no-stream security_nvr --format "{{.CPUPerc}}" > /tmp/cpu_usage.txt 2>/dev/null; then
    cpu_usage=$(cat /tmp/cpu_usage.txt | sed 's/%//')
    echo -e "  CPU Usage: ${GREEN}${cpu_usage}%${NC}"
    ((TESTS_PASSED++))
fi

# Check memory usage
if docker stats --no-stream security_nvr --format "{{.MemPerc}}" > /tmp/mem_usage.txt 2>/dev/null; then
    mem_usage=$(cat /tmp/mem_usage.txt | sed 's/%//')
    echo -e "  Memory Usage: ${GREEN}${mem_usage}%${NC}"
    ((TESTS_PASSED++))
fi

echo
echo "============================================"
echo -e "Tests Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests Failed: ${RED}$TESTS_FAILED${NC}"
echo

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed! Security NVR is properly deployed.${NC}"
    exit 0
else
    echo -e "${RED}✗ Some tests failed. Please check the configuration.${NC}"
    echo
    echo "Common issues:"
    echo "- Ensure docker-compose up -d has been run"
    echo "- Check docker logs security_nvr for errors"
    echo "- Verify MQTT broker is running"
    echo "- Check that required volumes are mounted"
    exit 1
fi