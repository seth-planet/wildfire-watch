############################
# cam_telemetry Dockerfile
############################

# Build ARG for multi-arch support
ARG PLATFORM
FROM --platform=${PLATFORM} python:3.13-slim

# Install Avahi & mDNS resolver
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      avahi-daemon libnss-mdns dbus && \
    rm -rf /var/lib/apt/lists/*

COPY nsswitch.conf /etc/nsswitch.conf
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Create app directory
WORKDIR /app

# Copy utils module
COPY utils /utils
# Add root to Python path so utils can be imported
ENV PYTHONPATH=/

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy telemetry code
COPY telemetry.py .


# Start Avahi, then run telemetry.py
ENTRYPOINT ["entrypoint.sh"]

# Run the telemetry publisher
CMD ["python3.12", "telemetry.py"]
