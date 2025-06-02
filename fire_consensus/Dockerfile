FROM python:3.11-slim

# Install system dependencies for mDNS and resilience
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        avahi-daemon \
        libnss-mdns \
        dbus \
        iputils-ping \
        && rm -rf /var/lib/apt/lists/*

# Configure mDNS resolution
COPY nsswitch.conf /etc/nsswitch.conf

# Create non-root user for security
RUN useradd -m -s /bin/bash consensus && \
    mkdir -p /var/run/dbus && \
    chown consensus:consensus /var/run/dbus

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY consensus.py .
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Switch to non-root user
USER consensus

# Health check for container monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); s.settimeout(5); s.connect(('localhost', 1883)); s.close()"

ENTRYPOINT ["entrypoint.sh"]
CMD ["python", "-u", "consensus.py"]
