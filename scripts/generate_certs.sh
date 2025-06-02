#!/usr/bin/env bash
# ===================================================================
# Certificate Generation Script - Wildfire Watch
# Generates CA and certificates for MQTT broker and clients
# ===================================================================
set -euo pipefail

# Configuration
OUTDIR="${CERT_DIR:-certs}"
CA_DAYS="${CA_DAYS:-3650}"  # 10 years
CERT_DAYS="${CERT_DAYS:-1825}"  # 5 years
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

# Generate CA certificate
generate_ca() {
    log_info "Generating Certificate Authority (CA)..."
    
    # Generate CA private key
    openssl genrsa -out "$OUTDIR/ca.key" $RSA_BITS 2>/dev/null
    
    # Generate CA certificate
    openssl req -x509 -new -nodes -key "$OUTDIR/ca.key" \
        -sha256 -days $CA_DAYS \
        -subj "/C=US/ST=CA/O=Wildfire Watch/CN=Wildfire Watch CA" \
        -out "$OUTDIR/ca.crt"
    
    log_info "CA certificate generated: $OUTDIR/ca.crt"
}

# Generate server certificate for MQTT broker
generate_server_cert() {
    local name="${1:-mqtt_broker}"
    
    log_info "Generating MQTT broker certificate..."
    
    # Generate private key
    openssl genrsa -out "$OUTDIR/server.key" $RSA_BITS 2>/dev/null
    
    # Create certificate request
    openssl req -new -key "$OUTDIR/server.key" \
        -subj "/C=US/ST=CA/O=Wildfire Watch/CN=$name" \
        -out "$OUTDIR/server.csr"
    
    # Create extensions file for SANs (Subject Alternative Names)
    cat > "$OUTDIR/server.ext" <<EOF
subjectAltName = @alt_names
[alt_names]
DNS.1 = mqtt_broker
DNS.2 = mqtt_broker.local
DNS.3 = localhost
DNS.4 = *.local
IP.1 = 127.0.0.1
IP.2 = ::1
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
    
    local client_dir="$OUTDIR/clients/$client_name"
    mkdir -p "$client_dir"
    
    # Generate private key
    openssl genrsa -out "$client_dir/client.key" $RSA_BITS 2>/dev/null
    
    # Create certificate request
    openssl req -new -key "$client_dir/client.key" \
        -subj "/C=US/ST=CA/O=Wildfire Watch/CN=$client_name" \
        -out "$client_dir/client.csr"
    
    # Sign the certificate
    openssl x509 -req -in "$client_dir/client.csr" \
        -CA "$OUTDIR/ca.crt" -CAkey "$OUTDIR/ca.key" \
        -CAcreateserial -days $CERT_DAYS -sha256 \
        -out "$client_dir/client.crt"
    
    # Clean up
    rm -f "$client_dir/client.csr"
    
    # Copy CA cert to client directory
    cp "$OUTDIR/ca.crt" "$client_dir/"
    
    log_info "Client certificate generated: $client_dir/client.crt"
}

# Generate all certificates
generate_all() {
    # Create output directory
    mkdir -p "$OUTDIR"
    
    # Check if CA already exists
    if [ -f "$OUTDIR/ca.crt" ] && [ -f "$OUTDIR/ca.key" ]; then
        log_warn "CA already exists. Using existing CA."
    else
        generate_ca
    fi
    
    # Generate server certificate
    generate_server_cert
    
    # Generate client certificates for common services
    local services=("frigate" "gpio_trigger" "fire_consensus" "cam_telemetry" "camera_detector")
    
    for service in "${services[@]}"; do
        generate_client_cert "$service"
    done
    
    # Set appropriate permissions
    chmod 600 "$OUTDIR"/*.key
    chmod 644 "$OUTDIR"/*.crt
    find "$OUTDIR/clients" -name "*.key" -exec chmod 600 {} \;
    find "$OUTDIR/clients" -name "*.crt" -exec chmod 644 {} \;
}

# Create tarball for easy distribution
create_distribution() {
    log_info "Creating distribution package..."
    
    # Create distribution directory
    local dist_dir="$OUTDIR/dist"
    mkdir -p "$dist_dir"
    
    # Copy required files for broker
    cp "$OUTDIR/ca.crt" "$OUTDIR/server.crt" "$OUTDIR/server.key" "$dist_dir/"
    
    # Create tarball
    tar -czf "$OUTDIR/wildfire-certs.tar.gz" -C "$dist_dir" .
    
    # Clean up
    rm -rf "$dist_dir"
    
    log_info "Distribution package created: $OUTDIR/wildfire-certs.tar.gz"
}

# Print certificate information
print_cert_info() {
    log_info "Certificate Information:"
    echo "=================================================="
    
    if [ -f "$OUTDIR/ca.crt" ]; then
        echo "CA Certificate:"
        openssl x509 -in "$OUTDIR/ca.crt" -noout -subject -enddate
        echo ""
    fi
    
    if [ -f "$OUTDIR/server.crt" ]; then
        echo "Server Certificate:"
        openssl x509 -in "$OUTDIR/server.crt" -noout -subject -enddate -ext subjectAltName
        echo ""
    fi
    
    echo "=================================================="
}

# Main execution
main() {
    check_requirements
    
    case "${1:-all}" in
        all)
            generate_all
            create_distribution
            print_cert_info
            ;;
        ca)
            mkdir -p "$OUTDIR"
            generate_ca
            ;;
        server)
            generate_server_cert "${2:-mqtt_broker}"
            ;;
        client)
            if [ -z "${2:-}" ]; then
                log_error "Usage: $0 client <client_name>"
                exit 1
            fi
            generate_client_cert "$2"
            ;;
        info)
            print_cert_info
            ;;
        *)
            echo "Usage: $0 [all|ca|server|client|info] [args...]"
            echo ""
            echo "Commands:"
            echo "  all              Generate CA, server, and client certificates"
            echo "  ca               Generate only the CA certificate"
            echo "  server [name]    Generate server certificate"
            echo "  client <name>    Generate client certificate for specified name"
            echo "  info             Display certificate information"
            echo ""
            echo "Environment variables:"
            echo "  CERT_DIR         Output directory (default: certs)"
            echo "  CA_DAYS          CA validity in days (default: 3650)"
            echo "  CERT_DAYS        Certificate validity in days (default: 1825)"
            echo "  RSA_BITS         RSA key size (default: 2048)"
            exit 1
            ;;
    esac
    
    log_info "Done! Certificates are in: $OUTDIR"
    
    # Quick setup instructions
    echo ""
    echo "Quick Setup:"
    echo "1. Copy $OUTDIR/wildfire-certs.tar.gz to your devices"
    echo "2. Extract to /mnt/data/certs/ on each device"
    echo "3. Restart the MQTT broker container"
}

# Run main
main "$@"
