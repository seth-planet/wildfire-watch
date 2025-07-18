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
# Add root to Python path so utils can be imported
ENV PYTHONPATH=/

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

# Copy the trigger script
COPY gpio_trigger/trigger.py .

# Start Avahi, then run trigger.py
ENTRYPOINT ["entrypoint.sh"]

# Run the GPIO trigger loop
CMD ["python3.12", "trigger.py"]
