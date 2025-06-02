FROM ghcr.io/blakeblackshear/frigate:stable

# Install additional dependencies for wildfire detection and hardware support
USER root

# Install system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        # Hardware detection
        pciutils \
        usbutils \
        v4l-utils \
        vainfo \
        intel-gpu-tools \
        # Storage management
        udev \
        e2fsprogs \
        # Monitoring tools
        iotop \
        htop \
        # Python packages for custom scripts
        python3-pip \
        python3-numpy \
        python3-opencv \
        # Build tools for TensorRT
        build-essential \
        cmake \
        # Network tools
        iputils-ping \
        net-tools \
        && rm -rf /var/lib/apt/lists/*

# Install Python packages for custom functionality
RUN pip3 install --no-cache-dir \
    # For model management
    onnx \
    onnxruntime \
    # For hardware detection
    pyudev \
    psutil \
    # For power management
    pySMART \
    # For USB management
    pyusb

# Create directories for custom models and scripts
RUN mkdir -p \
    /models/wildfire \
    /scripts \
    /config/custom \
    /var/log/frigate

# Copy custom scripts
COPY scripts/hardware_detector.py /scripts/
COPY scripts/usb_manager.py /scripts/
COPY scripts/model_selector.py /scripts/
COPY scripts/power_manager.py /scripts/
COPY scripts/camera_manager.py /scripts/
COPY scripts/event_search.py /scripts/
COPY scripts/entrypoint.sh /scripts/

# Copy wildfire detection models
COPY models/wildfire_coral_lite.tflite /models/wildfire/
COPY models/wildfire_coral_full.tflite /models/wildfire/
COPY models/wildfire_hailo8.hef /models/wildfire/
COPY models/wildfire_hailo8l.hef /models/wildfire/
COPY models/wildfire_tensorrt.onnx /models/wildfire/
COPY models/wildfire_openvino.xml /models/wildfire/
COPY models/wildfire_openvino.bin /models/wildfire/
COPY models/wildfire_cpu.tflite /models/wildfire/
COPY models/labels.txt /models/wildfire/

# Copy configuration templates
COPY config/frigate_base.yml /config/
COPY config/hardware_profiles.yml /config/
COPY config/power_profiles.yml /config/

# Make scripts executable
RUN chmod +x /scripts/*.sh /scripts/*.py

# Create symlinks for easy access
RUN ln -s /scripts/hardware_detector.py /usr/local/bin/check-hardware && \
    ln -s /scripts/event_search.py /usr/local/bin/find-events && \
    ln -s /scripts/camera_manager.py /usr/local/bin/test-camera && \
    ln -s /scripts/model_selector.py /usr/local/bin/test-model

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

# Create non-root user for Frigate
RUN useradd -m -s /bin/bash frigate && \
    usermod -aG video,audio,plugdev frigate && \
    chown -R frigate:frigate /config /models /scripts

# Volume for persistent data
VOLUME ["/media/frigate", "/config", "/var/log/frigate"]

# Switch back to frigate user
USER frigate

ENTRYPOINT ["/scripts/entrypoint.sh"]
CMD ["python3", "-m", "frigate"]