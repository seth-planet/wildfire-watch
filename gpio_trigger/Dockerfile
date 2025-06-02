############################
# gpio_trigger Dockerfile
############################

# Support both amd64 (local dev) and arm64 (Pi)
ARG PLATFORM
FROM --platform=${PLATFORM} python:3.11-slim

# Install Avahi & mDNS resolver
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      avahi-daemon libnss-mdns dbus && \
    rm -rf /var/lib/apt/lists/*

COPY nsswitch.conf /etc/nsswitch.conf
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the trigger script
COPY trigger.py .

# Start Avahi, then run trigger.py
ENTRYPOINT ["entrypoint.sh"]

# Run the GPIO trigger loop
CMD ["python", "trigger.py"]
