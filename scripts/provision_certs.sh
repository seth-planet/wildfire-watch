#!/usr/bin/env bash
# ===================================================================
# Certificate Provisioning Script - Wildfire Watch
# Deploys certificates to edge devices via SSH or Balena CLI
# ===================================================================
set -euo pipefail

# Configuration
CERT_DIR="${CERT_DIR:-certs}"
REMOTE_DIR="${REMOTE_DIR:-/mnt/data/certs}"
BALENA_APP="${BALENA_APP:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() { echo -e "${GREEN}[+]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[!]${NC} $1"; }
log_error() { echo -e "${RED}[!]${NC} $1"; }
log_step() { echo -e "${BLUE}[>]${NC} $1"; }

# Check requirements
check_requirements() {
    local missing_tools=()
    
    if ! command -v ssh >/dev/null 2>&1; then
        missing_tools+=("ssh")
    fi
    
    if [ -n "$BALENA_APP" ] && ! command -v balena >/dev/null 2>&1; then
        missing_tools+=("balena-cli")
    fi
    
    if [ ${#missing_tools[@]} -gt 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
}

# Validate certificates exist
validate_certs() {
    local required_files=("ca.crt" "server.crt" "server.key")
    local missing_files=()
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$CERT_DIR/$file" ]; then
            missing_files+=("$file")
        fi
    done
    
    if [ ${#missing_files[@]} -gt 0 ]; then
        log_error "Missing certificate files: ${missing_files[*]}"
        log_warn "Run ./scripts/generate_certs.sh first"
        exit 1
    fi
}

# Test SSH connectivity
test_ssh_connection() {
    local host="$1"
    
    if ssh -o ConnectTimeout=5 -o BatchMode=yes "root@$host" "echo 'SSH OK'" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Deploy certificates via SSH
deploy_via_ssh() {
    local host="$1"
    local service="${2:-all}"
    
    log_step "Deploying certificates to $host via SSH..."
    
    # Test connection
    if ! test_ssh_connection "$host"; then
        log_error "Cannot connect to $host via SSH"
        log_warn "Ensure SSH key authentication is set up for root@$host"
        return 1
    fi
    
    # Create remote directory
    ssh "root@$host" "mkdir -p $REMOTE_DIR && chmod 755 $REMOTE_DIR" || {
        log_error "Failed to create remote directory"
        return 1
    }
    
    # Copy CA and server certificates (always needed)
    log_step "Copying CA and server certificates..."
    scp -q "$CERT_DIR/ca.crt" "root@$host:$REMOTE_DIR/" || return 1
    scp -q "$CERT_DIR/server.crt" "root@$host:$REMOTE_DIR/" || return 1
    scp -q "$CERT_DIR/server.key" "root@$host:$REMOTE_DIR/" || return 1
    
    # Copy service-specific client certificates if requested
    if [ "$service" != "all" ] && [ -d "$CERT_DIR/clients/$service" ]; then
        log_step "Copying client certificates for $service..."
        ssh "root@$host" "mkdir -p $REMOTE_DIR/clients/$service"
        scp -q "$CERT_DIR/clients/$service/"* "root@$host:$REMOTE_DIR/clients/$service/" || return 1
    elif [ "$service" == "all" ] && [ -d "$CERT_DIR/clients" ]; then
        log_step "Copying all client certificates..."
        scp -r -q "$CERT_DIR/clients" "root@$host:$REMOTE_DIR/" || return 1
    fi
    
    # Set permissions
    ssh "root@$host" "find $REMOTE_DIR -name '*.key' -exec chmod 600 {} \; && find $REMOTE_DIR -name '*.crt' -exec chmod 644 {} \;"
    
    log_info "Successfully deployed certificates to $host"
    return 0
}

# Deploy certificates via Balena
deploy_via_balena() {
    local device="$1"
    local service="${2:-all}"
    
    log_step "Deploying certificates to $device via Balena..."
    
    # Check if device is online
    if ! balena device "$device" >/dev/null 2>&1; then
        log_error "Device $device not found or not accessible"
        return 1
    fi
    
    # Create tarball
    local temp_tar="/tmp/wildfire-certs-$$.tar.gz"
    tar -czf "$temp_tar" -C "$CERT_DIR" . || {
        log_error "Failed to create certificate archive"
        return 1
    }
    
    # Copy to device
    balena ssh "$device" "mkdir -p $REMOTE_DIR" || {
        log_error "Failed to create remote directory"
        rm -f "$temp_tar"
        return 1
    }
    
    # Transfer and extract
    cat "$temp_tar" | balena ssh "$device" "tar -xzf - -C $REMOTE_DIR" || {
        log_error "Failed to transfer certificates"
        rm -f "$temp_tar"
        return 1
    }
    
    # Clean up
    rm -f "$temp_tar"
    
    # Set permissions
    balena ssh "$device" "find $REMOTE_DIR -name '*.key' -exec chmod 600 {} \; && find $REMOTE_DIR -name '*.crt' -exec chmod 644 {} \;"
    
    log_info "Successfully deployed certificates to $device"
    return 0
}

# Deploy to multiple hosts
deploy_multiple() {
    local deployment_method="$1"
    shift
    local hosts=("$@")
    local failed_hosts=()
    
    for host in "${hosts[@]}"; do
        if [ "$deployment_method" == "ssh" ]; then
            if ! deploy_via_ssh "$host"; then
                failed_hosts+=("$host")
            fi
        elif [ "$deployment_method" == "balena" ]; then
            if ! deploy_via_balena "$host"; then
                failed_hosts+=("$host")
            fi
        fi
        echo ""
    done
    
    # Summary
    echo "=================================================="
    log_info "Deployment Summary:"
    echo "Total devices: ${#hosts[@]}"
    echo "Successful: $((${#hosts[@]} - ${#failed_hosts[@]}))"
    echo "Failed: ${#failed_hosts[@]}"
    
    if [ ${#failed_hosts[@]} -gt 0 ]; then
        log_error "Failed devices: ${failed_hosts[*]}"
        return 1
    fi
    
    return 0
}

# Auto-detect deployment method
auto_deploy() {
    local host="$1"
    
    # Try SSH first (faster)
    if test_ssh_connection "$host"; then
        deploy_via_ssh "$host"
    elif [ -n "$BALENA_APP" ]; then
        # Try Balena if app is configured
        deploy_via_balena "$host"
    else
        log_error "Cannot connect to $host via SSH and Balena app not configured"
        log_warn "Set BALENA_APP environment variable or ensure SSH access"
        return 1
    fi
}

# Verify deployment
verify_deployment() {
    local host="$1"
    
    log_step "Verifying deployment on $host..."
    
    if test_ssh_connection "$host"; then
        local files_found=$(ssh "root@$host" "ls -la $REMOTE_DIR/*.crt 2>/dev/null | wc -l")
        if [ "$files_found" -ge 2 ]; then
            log_info "Verification successful: $files_found certificate files found"
            return 0
        fi
    fi
    
    log_warn "Could not verify deployment"
    return 1
}

# Main execution
main() {
    check_requirements
    validate_certs
    
    case "${1:-help}" in
        ssh)
            shift
            if [ $# -lt 1 ]; then
                log_error "Usage: $0 ssh <host1> [host2...]"
                exit 1
            fi
            deploy_multiple "ssh" "$@"
            ;;
        balena)
            shift
            if [ $# -lt 1 ]; then
                log_error "Usage: $0 balena <device1> [device2...]"
                exit 1
            fi
            deploy_multiple "balena" "$@"
            ;;
        auto)
            shift
            if [ $# -lt 1 ]; then
                log_error "Usage: $0 auto <host1> [host2...]"
                exit 1
            fi
            for host in "$@"; do
                auto_deploy "$host"
                echo ""
            done
            ;;
        verify)
            shift
            if [ $# -lt 1 ]; then
                log_error "Usage: $0 verify <host>"
                exit 1
            fi
            verify_deployment "$1"
            ;;
        *)
            echo "Wildfire Watch Certificate Provisioning"
            echo ""
            echo "Usage: $0 [command] [options]"
            echo ""
            echo "Commands:"
            echo "  ssh <host1> [host2...]      Deploy via SSH to specified hosts"
            echo "  balena <dev1> [dev2...]     Deploy via Balena CLI to devices"
            echo "  auto <host1> [host2...]     Auto-detect deployment method"
            echo "  verify <host>               Verify deployment on host"
            echo ""
            echo "Options:"
            echo "  CERT_DIR       Certificate directory (default: certs)"
            echo "  REMOTE_DIR     Remote directory (default: /mnt/data/certs)"
            echo "  BALENA_APP     Balena application name for CLI deployment"
            echo ""
            echo "Examples:"
            echo "  $0 ssh 192.168.1.100 192.168.1.101"
            echo "  $0 balena device1 device2"
            echo "  $0 auto camera1.local camera2.local"
            echo "  BALENA_APP=myapp $0 balena edge-device"
            ;;
    esac
}

# Run main
main "$@"
