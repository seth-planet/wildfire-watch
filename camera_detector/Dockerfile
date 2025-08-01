FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        # Network tools
        iputils-ping \
        net-tools \
        nmap \
        arp-scan \
        # mDNS support
        avahi-daemon \
        libnss-mdns \
        dbus \
        # Video/RTSP dependencies
        ffmpeg \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        # Build dependencies
        gcc \
        g++ \
        && rm -rf /var/lib/apt/lists/*

# Configure mDNS resolution
COPY camera_detector/nsswitch.conf /etc/nsswitch.conf

# Create non-root user
RUN useradd -m -s /bin/bash detector && \
    mkdir -p /var/run/dbus && \
    chown detector:detector /var/run/dbus

WORKDIR /app

# Copy utils module
COPY utils /utils
# Add root to Python path so utils can be imported
ENV PYTHONPATH=/

# Install Python dependencies
COPY camera_detector/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy WSDL files script
COPY camera_detector/copy_wsdl.py .

# Copy WSDL files from onvif-zeep package to expected location
RUN python3.12 copy_wsdl.py

# Copy application files
COPY camera_detector/detect.py .
COPY camera_detector/entrypoint_simple.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Create directories for Frigate config
RUN mkdir -p /config/frigate && \
    chown -R detector:detector /config && \
    chmod -R 755 /utils

# Switch to non-root user
USER detector

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python3.12 -c "import json; import sys; sys.path.append('/app'); from detect import CameraDetector; d=CameraDetector(); sys.exit(0 if d.get_health()['healthy'] else 1)"

ENTRYPOINT ["entrypoint.sh"]
CMD ["python3.12", "-u", "detect.py"]