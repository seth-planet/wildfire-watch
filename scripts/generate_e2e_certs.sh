#!/usr/bin/env bash
# ===================================================================
# E2E Test Certificate Generation Script - Wildfire Watch
# Generates CA and certificates for E2E testing with proper hostnames
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

# Generate server certificate with E2E test hostnames
generate_e2e_server_cert() {
    log_info "Generating E2E MQTT broker certificate..."
    
    # Generate private key
    openssl genrsa -out "$OUTDIR/server.key" $RSA_BITS 2>/dev/null
    
    # Create certificate request
    openssl req -new -key "$OUTDIR/server.key" \
        -subj "/C=US/ST=CA/O=Wildfire Watch/CN=mqtt_broker" \
        -out "$OUTDIR/server.csr"
    
    # Create extensions file with all possible hostnames for E2E testing
    cat > "$OUTDIR/server.ext" <<EOF
subjectAltName = @alt_names
[alt_names]
DNS.1 = mqtt_broker
DNS.2 = mqtt-broker
DNS.3 = e2e-mqtt-broker
DNS.4 = test-mqtt-e2e
DNS.5 = localhost
DNS.6 = *.local
DNS.7 = mqtt_broker.local
DNS.8 = mqtt-broker.local
DNS.9 = e2e-mqtt-broker.local
DNS.10 = test-mqtt-e2e.local
IP.1 = 127.0.0.1
IP.2 = ::1
IP.3 = 172.20.0.2
IP.4 = 172.20.0.10
IP.5 = 192.168.100.10
EOF
    
    # Sign the certificate
    openssl x509 -req -in "$OUTDIR/server.csr" \
        -CA "$OUTDIR/ca.crt" -CAkey "$OUTDIR/ca.key" \
        -CAcreateserial -days $CERT_DAYS -sha256 \
        -extfile "$OUTDIR/server.ext" \
        -out "$OUTDIR/server.crt"
    
    # Clean up
    rm -f "$OUTDIR/server.csr" "$OUTDIR/server.ext"
    
    log_info "Server certificate generated: $OUTDIR/server.crt"
}

# Generate client certificate
generate_client_cert() {
    local client_name="$1"
    
    if [ -z "$client_name" ]; then
        log_error "Client name required"
        return 1
    fi
    
    log_info "Generating client certificate for: $client_name"
    
    # Use common client cert directory
    local client_file="$OUTDIR/client"
    
    # Generate private key
    openssl genrsa -out "${client_file}.key" $RSA_BITS 2>/dev/null
    
    # Create certificate request
    openssl req -new -key "${client_file}.key" \
        -subj "/C=US/ST=CA/O=Wildfire Watch/CN=wildfire-client" \
        -out "${client_file}.csr"
    
    # Sign the certificate
    openssl x509 -req -in "${client_file}.csr" \
        -CA "$OUTDIR/ca.crt" -CAkey "$OUTDIR/ca.key" \
        -CAcreateserial -days $CERT_DAYS -sha256 \
        -out "${client_file}.crt"
    
    # Clean up
    rm -f "${client_file}.csr"
    
    log_info "Client certificate generated: ${client_file}.crt"
}

# Main execution
main() {
    check_requirements
    
    # Create output directory
    mkdir -p "$OUTDIR"
    
    # Generate or reuse CA
    generate_or_reuse_ca
    
    # Generate server certificate with E2E hostnames
    generate_e2e_server_cert
    
    # Generate generic client certificate
    generate_client_cert "wildfire-client"
    
    # Set appropriate permissions
    chmod 600 "$OUTDIR"/*.key
    chmod 644 "$OUTDIR"/*.crt
    
    # Print certificate information
    log_info "Certificate Information:"
    echo "=================================================="
    
    echo "CA Certificate:"
    openssl x509 -in "$OUTDIR/ca.crt" -noout -subject -enddate
    echo ""
    
    echo "Server Certificate:"
    openssl x509 -in "$OUTDIR/server.crt" -noout -subject -enddate -ext subjectAltName
    echo ""
    
    echo "=================================================="
    
    log_info "Done! E2E test certificates are ready in: $OUTDIR"
    echo ""
    echo "The server certificate includes these hostnames:"
    echo "  - mqtt_broker, mqtt-broker"
    echo "  - e2e-mqtt-broker, test-mqtt-e2e"
    echo "  - localhost, *.local"
    echo ""
    echo "To use these certificates, run your E2E tests normally."
}

# Run main
main "$@"