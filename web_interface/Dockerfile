############################
# web_interface Dockerfile
############################

# Support both amd64 (local dev) and arm64 (Pi)
ARG PLATFORM=linux/amd64
FROM --platform=${PLATFORM} python:3.12-slim

# Build argument for environment (test or production)
ARG BUILD_ENV=production

# Install Avahi & mDNS resolver + security updates
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      avahi-daemon libnss-mdns dbus curl \
      ca-certificates && \
    apt-get upgrade -y && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -s /bin/bash webui && \
    mkdir -p /var/run/dbus /var/run/avahi-daemon && \
    chown webui:webui /var/run/dbus /var/run/avahi-daemon

# Copy configuration files
COPY web_interface/nsswitch.conf /etc/nsswitch.conf
COPY web_interface/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /app

# Copy utils module
COPY utils /utils
# More restrictive Python path
ENV PYTHONPATH=/app:/utils

# Performance optimizations for Pi
ENV PYTHONOPTIMIZE=2
ENV PYTHONDONTWRITEBYTECODE=1

# Copy requirements files
COPY web_interface/requirements*.txt ./

# Install dependencies with verification
RUN if [ "$BUILD_ENV" = "test" ] && [ -f requirements_test.txt ]; then \
        pip install --no-cache-dir -r requirements_test.txt; \
    else \
        pip install --no-cache-dir -r requirements.txt && \
        python -c "import fastapi, uvicorn, paho.mqtt, jinja2" || \
        (echo "ERROR: Critical dependencies failed to install" && exit 1); \
    fi

# Copy the entire web_interface module
COPY web_interface /app/web_interface

# Copy startup script
COPY web_interface/start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

# Set proper ownership
RUN chown -R webui:webui /app /utils && \
    chmod -R 755 /app /utils

# Precompile Python files for faster startup
RUN python -m compileall -b /app /utils

# Expose the web interface port
EXPOSE 8080

# Health check with proper timeout
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8080/api/health || exit 1

# Start with entrypoint (runs as root for avahi)
ENTRYPOINT ["entrypoint.sh"]

# Run the FastAPI application with optimized settings
# Workers=1 for Pi 3, can increase for Pi 4/5
CMD ["/usr/local/bin/start.sh"]