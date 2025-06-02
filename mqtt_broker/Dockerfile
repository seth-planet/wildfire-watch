FROM eclipse-mosquitto:2.0

# Install mDNS support for service discovery
USER root
RUN apk add --no-cache avahi avahi-tools dbus

# Copy configuration files
COPY mosquitto.conf /mosquitto/config/mosquitto.conf
COPY conf.d /mosquitto/config/conf.d/
COPY entrypoint.sh /entrypoint.sh
COPY nsswitch.conf /etc/nsswitch.conf

# Ensure proper permissions
RUN chown -R mosquitto:mosquitto /mosquitto/config && \
    chmod +x /entrypoint.sh && \
    mkdir -p /mosquitto/certs && \
    chown mosquitto:mosquitto /mosquitto/certs

# Create directories for persistence
RUN mkdir -p /mosquitto/data /mosquitto/log && \
    chown -R mosquitto:mosquitto /mosquitto/data /mosquitto/log

EXPOSE 1883 8883 9001

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/usr/sbin/mosquitto", "-c", "/mosquitto/config/mosquitto.conf"]
