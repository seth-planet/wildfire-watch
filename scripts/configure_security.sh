#!/bin/bash
# Script to configure security settings for Wildfire Watch

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=================================================="
echo "Wildfire Watch Security Configuration"
echo "=================================================="
echo

# Function to enable TLS
enable_tls() {
    echo "Enabling TLS/SSL encryption..."
    
    # Check if custom certificates exist
    if [ ! -f "$PROJECT_ROOT/certs/ca.crt" ] || grep -q "Default Development CA" "$PROJECT_ROOT/certs/ca.crt" 2>/dev/null; then
        echo "⚠️  WARNING: Using default certificates!"
        echo "These provide NO SECURITY and are public."
        echo
        read -p "Generate secure certificates now? (Y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
            "$SCRIPT_DIR/generate_certs.sh" custom
        fi
    fi
    
    # Update .env file
    if [ -f "$PROJECT_ROOT/.env" ]; then
        sed -i 's/MQTT_TLS=false/MQTT_TLS=true/' "$PROJECT_ROOT/.env"
        echo "✓ Updated .env to enable TLS"
    else
        echo "MQTT_TLS=true" >> "$PROJECT_ROOT/.env"
        echo "✓ Created .env with TLS enabled"
    fi
    
    echo
    echo "TLS enabled. Services will use port 8883 for secure MQTT."
    echo "Remember to restart services: docker-compose restart"
}

# Function to disable TLS
disable_tls() {
    echo "Disabling TLS/SSL encryption..."
    echo "⚠️  WARNING: This is insecure and should only be used for testing!"
    echo
    
    # Update .env file
    if [ -f "$PROJECT_ROOT/.env" ]; then
        sed -i 's/MQTT_TLS=true/MQTT_TLS=false/' "$PROJECT_ROOT/.env"
        echo "✓ Updated .env to disable TLS"
    else
        echo "MQTT_TLS=false" >> "$PROJECT_ROOT/.env"
        echo "✓ Created .env with TLS disabled"
    fi
    
    echo
    echo "TLS disabled. Services will use port 1883 for plain MQTT."
    echo "Remember to restart services: docker-compose restart"
}

# Function to check current status
check_status() {
    echo "Current Security Configuration:"
    echo "=============================="
    
    # Check .env
    if [ -f "$PROJECT_ROOT/.env" ]; then
        tls_status=$(grep -E "^MQTT_TLS=" "$PROJECT_ROOT/.env" | cut -d= -f2)
        if [ "$tls_status" = "true" ]; then
            echo "✓ TLS is ENABLED"
        else
            echo "✗ TLS is DISABLED (insecure)"
        fi
    else
        echo "✗ No .env file found - TLS is DISABLED by default"
    fi
    
    # Check certificates
    echo
    if [ -f "$PROJECT_ROOT/certs/ca.crt" ]; then
        if grep -q "Default Development CA" "$PROJECT_ROOT/certs/ca.crt" 2>/dev/null; then
            echo "⚠️  Using DEFAULT certificates (INSECURE)"
        else
            echo "✓ Using CUSTOM certificates"
            # Show certificate info
            echo "  CA Subject: $(openssl x509 -in "$PROJECT_ROOT/certs/ca.crt" -noout -subject | cut -d= -f2-)"
            echo "  Valid until: $(openssl x509 -in "$PROJECT_ROOT/certs/ca.crt" -noout -enddate | cut -d= -f2)"
        fi
    else
        echo "✗ No certificates found"
    fi
    
    # Check running services
    echo
    echo "Service Status:"
    if command -v docker &> /dev/null; then
        mqtt_port=$(docker ps --format "table {{.Names}}\t{{.Ports}}" | grep mqtt_broker | grep -o "8883" || echo "")
        if [ -n "$mqtt_port" ]; then
            echo "✓ MQTT broker is using TLS port 8883"
        else
            mqtt_port=$(docker ps --format "table {{.Names}}\t{{.Ports}}" | grep mqtt_broker | grep -o "1883" || echo "")
            if [ -n "$mqtt_port" ]; then
                echo "✗ MQTT broker is using plain port 1883 (insecure)"
            else
                echo "- MQTT broker is not running"
            fi
        fi
    else
        echo "- Docker not available"
    fi
}

# Main menu
case "${1:-}" in
    enable)
        enable_tls
        ;;
    disable)
        disable_tls
        ;;
    status)
        check_status
        ;;
    *)
        echo "Usage: $0 {enable|disable|status}"
        echo
        echo "Commands:"
        echo "  enable  - Enable TLS encryption (recommended)"
        echo "  disable - Disable TLS encryption (testing only)"
        echo "  status  - Show current security configuration"
        echo
        check_status
        ;;
esac