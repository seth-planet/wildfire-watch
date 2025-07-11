############################
# gpio_trigger Dockerfile
############################

# Support both amd64 (local dev) and arm64 (Pi)
ARG PLATFORM
FROM --platform=${PLATFORM} python:3.12-slim

# Build argument for environment (test or production)
ARG BUILD_ENV=production

# Install Avahi & mDNS resolver
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      avahi-daemon libnss-mdns dbus && \
    rm -rf /var/lib/apt/lists/*

COPY gpio_trigger/nsswitch.conf /etc/nsswitch.conf
COPY gpio_trigger/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /app

# Copy utils module
COPY utils /utils

# Copy gpio_trigger module (needed for gpio_safety imports)
COPY gpio_trigger /app/gpio_trigger

# Copy debug scripts
COPY scripts/debug_mqtt_connection.py /scripts/
COPY scripts/container_connectivity_test.py /scripts/
COPY gpio_trigger/test_entrypoint.sh /test_entrypoint.sh
RUN chmod +x /test_entrypoint.sh

# Add root and app to Python path so modules can be imported
ENV PYTHONPATH=/:/app

# Fix Python output buffering in Docker
ENV PYTHONUNBUFFERED=1

# Copy all requirements files
COPY gpio_trigger/requirements*.txt ./

# Install appropriate dependencies based on BUILD_ENV
RUN if [ "$BUILD_ENV" = "test" ] && [ -f requirements_test.txt ]; then \
        pip install --no-cache-dir -r requirements_test.txt; \
    else \
        pip install --no-cache-dir -r requirements.txt || \
        (echo "Warning: Some packages failed to install, installing core dependencies only" && \
         pip install --no-cache-dir paho-mqtt>=2.0.0 python-dotenv>=1.0.0 PyYAML>=6.0.0 psutil>=5.9.0 fake-rpi>=0.7.1); \
    fi

# Start Avahi, then run trigger.py
ENTRYPOINT ["entrypoint.sh"]

# Set Python to unbuffered mode for immediate log output
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8

# Run the GPIO trigger loop with unbuffered output
CMD ["python3.12", "-u", "gpio_trigger/trigger.py"]
