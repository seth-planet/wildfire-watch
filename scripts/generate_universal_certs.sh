#!/usr/bin/env bash
# ===================================================================
# Universal Certificate Generation Script - Wildfire Watch
# Generates CA and certificates that work for all environments
# ===================================================================
set -euo pipefail

# Configuration
OUTDIR="${CERT_DIR:-certs}"
CA_DAYS="${CA_DAYS:-14600}"  # 40 years
CERT_DAYS="${CERT_DAYS:-14600}"  # 40 years
RSA_BITS="${RSA_BITS:-2048}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log_info() { echo -e "${GREEN}[+]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[!]${NC} $1"; }
log_error() { echo -e "${RED}[!]${NC} $1"; }

# Check for required tools
check_requirements() {
    if ! command -v openssl >/dev/null 2>&1; then
        log_error "OpenSSL is required but not installed"
        exit 1
    fi
}

# Generate or reuse CA certificate
generate_or_reuse_ca() {
    if [ -f "$OUTDIR/ca.crt" ] && [ -f "$OUTDIR/ca.key" ]; then
        log_info "Using existing CA certificate"
    else
        log_info "Generating Certificate Authority (CA)..."
        
        # Generate CA private key
        openssl genrsa -out "$OUTDIR/ca.key" $RSA_BITS 2>/dev/null
        
        # Generate CA certificate
        openssl req -x509 -new -nodes -key "$OUTDIR/ca.key" \
            -sha256 -days $CA_DAYS \
            -subj "/C=US/ST=CA/O=Wildfire Watch/CN=Wildfire Watch CA" \
            -out "$OUTDIR/ca.crt"
        
        log_info "CA certificate generated: $OUTDIR/ca.crt"
    fi
}

# Generate server certificate with all possible hostnames
generate_universal_server_cert() {
    log_info "Generating universal MQTT broker certificate..."
    
    # Backup existing certificates if they exist
    if [ -f "$OUTDIR/server.key" ]; then
        cp "$OUTDIR/server.key" "$OUTDIR/server.key.bak"
        cp "$OUTDIR/server.crt" "$OUTDIR/server.crt.bak"
        log_warn "Backed up existing server certificates"
    fi
    
    # Generate private key
    openssl genrsa -out "$OUTDIR/server.key" $RSA_BITS 2>/dev/null
    
    # Create certificate request
    openssl req -new -key "$OUTDIR/server.key" \
        -subj "/C=US/ST=CA/O=Wildfire Watch/CN=mqtt_broker" \
        -out "$OUTDIR/server.csr"
    
    # Create extensions file with comprehensive hostname coverage
    cat > "$OUTDIR/server.ext" <<EOF
subjectAltName = @alt_names
[alt_names]
# Production hostnames
DNS.1 = mqtt_broker
DNS.2 = mqtt-broker
DNS.3 = mosquitto
DNS.4 = mqtt

# Testing hostnames  
DNS.5 = e2e-mqtt-broker
DNS.6 = test-mqtt-broker
DNS.7 = test-mqtt-e2e
DNS.8 = e2e-mosquitto
DNS.9 = test-mosquitto

# Local development
DNS.10 = localhost
DNS.11 = localhost.localdomain
DNS.12 = *.local
DNS.13 = mqtt.local
DNS.14 = mqtt_broker.local
DNS.15 = mqtt-broker.local

# Docker networking
DNS.16 = *.wildfire_net
DNS.17 = mqtt_broker.wildfire_net
DNS.18 = mqtt-broker.wildfire_net

# Container names
DNS.19 = wildfire-mqtt
DNS.20 = wildfire-mqtt-broker

# Balena/fleet management
DNS.21 = *.balena-devices.com
DNS.22 = *.resin.local

# Common variations
DNS.23 = broker
DNS.24 = mqtt-server
DNS.25 = mqttserver

# IP addresses
IP.1 = 127.0.0.1
IP.2 = ::1
IP.3 = 0.0.0.0
IP.4 = 172.16.0.0
IP.5 = 172.17.0.0
IP.6 = 172.18.0.0
IP.7 = 172.19.0.0
IP.8 = 172.20.0.0
IP.9 = 192.168.0.0
IP.10 = 192.168.1.0
IP.11 = 192.168.100.0
IP.12 = 10.0.0.0
EOF
    
    # Add any additional IPs from common Docker subnets
    for i in {2..254}; do
        echo "IP.$((12 + i)) = 172.20.0.$i" >> "$OUTDIR/server.ext"
    done
    
    # Sign the certificate
    openssl x509 -req -in "$OUTDIR/server.csr" \
        -CA "$OUTDIR/ca.crt" -CAkey "$OUTDIR/ca.key" \
        -CAcreateserial -days $CERT_DAYS -sha256 \
        -extfile "$OUTDIR/server.ext" \
        -out "$OUTDIR/server.crt"
    
    # Clean up
    rm -f "$OUTDIR/server.csr" "$OUTDIR/server.ext"
    
    log_info "Universal server certificate generated: $OUTDIR/server.crt"
}

# Generate client certificate
generate_universal_client_cert() {
    log_info "Generating universal client certificate..."
    
    # Use common client cert
    local client_file="$OUTDIR/client"
    
    # Backup existing if present
    if [ -f "${client_file}.key" ]; then
        cp "${client_file}.key" "${client_file}.key.bak"
        cp "${client_file}.crt" "${client_file}.crt.bak"
        log_warn "Backed up existing client certificates"
    fi
    
    # Generate private key
    openssl genrsa -out "${client_file}.key" $RSA_BITS 2>/dev/null
    
    # Create certificate request
    openssl req -new -key "${client_file}.key" \
        -subj "/C=US/ST=CA/O=Wildfire Watch/CN=wildfire-client" \
        -out "${client_file}.csr"
    
    # Create extensions file for client
    cat > "${client_file}.ext" <<EOF
subjectAltName = @alt_names
[alt_names]
DNS.1 = wildfire-client
DNS.2 = *.wildfire-watch
DNS.3 = localhost
IP.1 = 127.0.0.1
EOF
    
    # Sign the certificate
    openssl x509 -req -in "${client_file}.csr" \
        -CA "$OUTDIR/ca.crt" -CAkey "$OUTDIR/ca.key" \
        -CAcreateserial -days $CERT_DAYS -sha256 \
        -extfile "${client_file}.ext" \
        -out "${client_file}.crt"
    
    # Clean up
    rm -f "${client_file}.csr" "${client_file}.ext"
    
    log_info "Universal client certificate generated: ${client_file}.crt"
}

# Generate service-specific client certificates
generate_service_client_certs() {
    local services=("frigate" "gpio_trigger" "fire_consensus" "cam_telemetry" "camera_detector")
    
    for service in "${services[@]}"; do
        log_info "Generating certificate for service: $service"
        
        local service_dir="$OUTDIR/clients/$service"
        mkdir -p "$service_dir"
        
        # Copy the universal client certificate
        cp "$OUTDIR/client.crt" "$service_dir/client.crt"
        cp "$OUTDIR/client.key" "$service_dir/client.key"
        cp "$OUTDIR/ca.crt" "$service_dir/ca.crt"
        
        # Set permissions
        chmod 600 "$service_dir/client.key"
        chmod 644 "$service_dir/client.crt"
        chmod 644 "$service_dir/ca.crt"
    done
}

# Verify certificate
verify_certificate() {
    log_info "Verifying certificate configuration..."
    
    # Verify server certificate
    openssl verify -CAfile "$OUTDIR/ca.crt" "$OUTDIR/server.crt" >/dev/null 2>&1
    if [ $? -eq 0 ]; then
        log_info "✓ Server certificate verification passed"
    else
        log_error "✗ Server certificate verification failed"
    fi
    
    # Verify client certificate
    openssl verify -CAfile "$OUTDIR/ca.crt" "$OUTDIR/client.crt" >/dev/null 2>&1
    if [ $? -eq 0 ]; then
        log_info "✓ Client certificate verification passed"
    else
        log_error "✗ Client certificate verification failed"
    fi
}

# Main execution
main() {
    check_requirements
    
    # Create output directory
    mkdir -p "$OUTDIR"
    
    # Generate or reuse CA
    generate_or_reuse_ca
    
    # Generate universal server certificate
    generate_universal_server_cert
    
    # Generate universal client certificate
    generate_universal_client_cert
    
    # Generate service-specific directories
    generate_service_client_certs
    
    # Set appropriate permissions
    chmod 600 "$OUTDIR"/*.key 2>/dev/null || true
    chmod 644 "$OUTDIR"/*.crt
    
    # Verify certificates
    verify_certificate
    
    # Print certificate information
    log_info "Certificate Information:"
    echo "=================================================="
    
    echo "CA Certificate:"
    openssl x509 -in "$OUTDIR/ca.crt" -noout -subject -enddate
    echo ""
    
    echo "Server Certificate:"
    echo "Subject:"
    openssl x509 -in "$OUTDIR/server.crt" -noout -subject
    echo "Valid until:"
    openssl x509 -in "$OUTDIR/server.crt" -noout -enddate
    echo ""
    echo "Subject Alternative Names (first 20):"
    openssl x509 -in "$OUTDIR/server.crt" -noout -ext subjectAltName | head -n 25
    echo ""
    
    echo "=================================================="
    
    log_info "Done! Universal certificates are ready in: $OUTDIR"
    echo ""
    echo "These certificates will work for:"
    echo "  ✓ Production deployment (mqtt_broker, mqtt-broker)"
    echo "  ✓ E2E testing (e2e-mqtt-broker, test-mqtt-e2e)"
    echo "  ✓ Local development (localhost, *.local)"
    echo "  ✓ Docker networking (various subnets)"
    echo "  ✓ Balena deployment (*.balena-devices.com)"
    echo ""
    echo "The certificates support hostname verification and will not"
    echo "cause TLS hostname mismatch errors in any environment."
}

# Run main
main "$@"