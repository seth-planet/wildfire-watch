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
COPY nsswitch.conf /etc/nsswitch.conf

# Create non-root user
RUN useradd -m -s /bin/bash detector && \
    mkdir -p /var/run/dbus && \
    chown detector:detector /var/run/dbus

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY detect.py .
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Create directories for Frigate config
RUN mkdir -p /config/frigate && \
    chown -R detector:detector /config

# Switch to non-root user
USER detector

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import json; import sys; sys.path.append('/app'); from detect import CameraDetector; d=CameraDetector(); sys.exit(0 if d.get_health()['healthy'] else 1)"

ENTRYPOINT ["entrypoint.sh"]
CMD ["python", "-u", "detect.py"]