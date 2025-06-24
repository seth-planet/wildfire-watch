# Use an ARG to receive the version from docker-compose.yml
ARG FRIGATE_VERSION=stable
FROM ghcr.io/blakeblackshear/frigate:${FRIGATE_VERSION}

# Set the working directory for clarity
WORKDIR /

# Add the root directory to Python's path to find /utils
ENV PYTHONPATH=/

# Install additional dependencies if needed
# Most tools are already in the Frigate base image
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        jq \
        pciutils \
        usbutils \
    && rm -rf /var/lib/apt/lists/* || true

# Install Python packages in a single layer
RUN python3 -m pip install --no-cache-dir \
    onnx \
    onnxruntime \
    pyudev \
    psutil \
    pyusb

# Create directories for custom models and scripts
RUN mkdir -p \
    /models/wildfire \
    /scripts \
    /config/custom \
    /var/log/frigate

# Copy custom scripts and shared utils from the correct context
# The paths are now relative to the project root (the build context)
COPY security_nvr/hardware_detector.py /scripts/
COPY security_nvr/usb_manager.py /scripts/
COPY security_nvr/storage_manager.py /scripts/
COPY security_nvr/camera_manager.py /scripts/
COPY security_nvr/entrypoint.sh /scripts/
COPY security_nvr/nvr_base_config.yml /config/

# Copy the shared utils module
COPY utils /utils

# Make scripts executable
RUN chmod +x /scripts/*.sh /scripts/*.py

# Create symlinks for easy access
RUN ln -s /scripts/hardware_detector.py /usr/local/bin/check-hardware && \
    ln -s /scripts/camera_manager.py /usr/local/bin/test-camera

# Set up health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/api/stats || exit 1

# Environment variables for configuration
ENV FRIGATE_RTSP_PASSWORD="" \
    FRIGATE_MQTT_HOST="mqtt_broker" \
    FRIGATE_MQTT_PORT="8883" \
    FRIGATE_MQTT_TLS="true" \
    FRIGATE_MODEL="auto" \
    FRIGATE_HARDWARE="auto" \
    HARDWARE_ACCEL="auto" \
    POWER_MODE="balanced" \
    USB_MOUNT_PATH="/media/frigate" \
    RECORD_RETAIN_DAYS="180" \
    DETECTION_FPS="5" \
    DETECTION_THRESHOLD="0.7" \
    LOG_LEVEL="info"

# Create non-root user and set permissions for all custom/mounted files
# This now correctly includes /utils
RUN useradd -m -s /bin/bash frigate && \
    usermod -aG video,audio,plugdev frigate && \
    chown -R frigate:frigate /config /models /scripts /utils /var/log/frigate

# Volume for persistent data
VOLUME ["/media/frigate", "/config", "/var/log/frigate"]

# Switch back to the default user (root for s6)
USER root

# Our custom entrypoint wraps the default Frigate startup
ENTRYPOINT ["/scripts/entrypoint.sh"]