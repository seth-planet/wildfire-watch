#!/bin/bash
# Environment setup script for running wildfire-watch tests
# Source this file before running tests: source tests/setup_test_env.sh

echo "Setting up wildfire-watch test environment..."

# Camera credentials for tests
# Set CAMERA_CREDENTIALS environment variable before running tests
# Example: export CAMERA_CREDENTIALS="admin:password"
export CAMERA_CREDENTIALS="${CAMERA_CREDENTIALS:-admin:password}"

# CUDA configuration to prevent out of memory errors
export CUDA_VISIBLE_DEVICES=0
export CUDA_MEMORY_FRACTION=0.3

# MQTT optimization for faster test execution
export MQTT_OPTIMIZATION=true

# GPIO simulation for non-Raspberry Pi systems
export GPIO_SIMULATION=true

# Additional test environment variables
export MQTT_BROKER=localhost
export MQTT_PORT=1883
export LOG_LEVEL=INFO

# Hailo/Coral hardware lockfile directory
export HARDWARE_LOCK_DIR=/tmp

echo "Test environment configured:"
echo "  CAMERA_CREDENTIALS: ${CAMERA_CREDENTIALS}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  CUDA_MEMORY_FRACTION: ${CUDA_MEMORY_FRACTION}"
echo "  MQTT_OPTIMIZATION: ${MQTT_OPTIMIZATION}"
echo "  GPIO_SIMULATION: ${GPIO_SIMULATION}"
echo "  HARDWARE_LOCK_DIR: ${HARDWARE_LOCK_DIR}"