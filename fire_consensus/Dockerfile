FROM python:3.13-slim

# Install system dependencies for mDNS and resilience
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        avahi-daemon \
        libnss-mdns \
        dbus \
        iputils-ping \
        && rm -rf /var/lib/apt/lists/*

# Configure mDNS resolution
COPY fire_consensus/nsswitch.conf /etc/nsswitch.conf

# Create non-root user for security
RUN useradd -m -s /bin/bash consensus && \
    mkdir -p /var/run/dbus /var/run/avahi-daemon && \
    chown consensus:consensus /var/run/dbus /var/run/avahi-daemon

WORKDIR /app

# Copy utils module (context is project root)
COPY utils /utils
# Add root to Python path so utils can be imported
ENV PYTHONPATH=/

# Install Python dependencies
COPY fire_consensus/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY fire_consensus/consensus.py .
COPY fire_consensus/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Don't switch to non-root user - needed for Avahi daemon
# USER consensus

# Health check for container monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python3.12 -c "import socket; s=socket.socket(); s.settimeout(5); s.connect(('localhost', 1883)); s.close()"

ENTRYPOINT ["entrypoint.sh"]
CMD ["python3.12", "-u", "consensus.py"]
