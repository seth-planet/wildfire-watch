#!/bin/bash
# Deploy refactored services with safety checks and rollback capability

set -euo pipefail

# Configuration
DEPLOYMENT_MODE="${1:-staging}"  # staging, canary, production
ROLLBACK_TIMEOUT=300  # 5 minutes to detect issues
LOG_FILE="/tmp/deployment_$(date +%Y%m%d_%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${2:-}$(date '+%Y-%m-%d %H:%M:%S') - $1${NC}" | tee -a "$LOG_FILE"
}

# Error handler
error_exit() {
    log "ERROR: $1" "$RED"
    exit 1
}

# Success message
success() {
    log "SUCCESS: $1" "$GREEN"
}

# Warning message
warning() {
    log "WARNING: $1" "$YELLOW"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error_exit "Docker is not installed"
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error_exit "Docker Compose is not installed"
    fi
    
    # Check MQTT tools
    if ! command -v mosquitto_sub &> /dev/null; then
        warning "mosquitto-clients not installed - skipping MQTT validation"
    fi
    
    # Check for required files
    if [[ ! -f "docker-compose.yml" ]]; then
        error_exit "docker-compose.yml not found"
    fi
    
    if [[ ! -f ".env" ]]; then
        error_exit ".env file not found"
    fi
    
    success "Prerequisites check passed"
}

# Backup current configuration
backup_current() {
    log "Creating backup of current configuration..."
    
    BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # Backup compose files
    cp docker-compose*.yml "$BACKUP_DIR/" 2>/dev/null || true
    
    # Backup environment files
    cp .env* "$BACKUP_DIR/" 2>/dev/null || true
    
    # Backup current service states
    docker-compose ps > "$BACKUP_DIR/service_states.txt"
    
    # Export current images
    for service in camera_detector fire_consensus gpio_trigger; do
        if docker-compose ps | grep -q "$service"; then
            IMAGE=$(docker-compose images -q "$service" 2>/dev/null || true)
            if [[ -n "$IMAGE" ]]; then
                echo "$service=$IMAGE" >> "$BACKUP_DIR/image_versions.txt"
            fi
        fi
    done
    
    success "Backup created in $BACKUP_DIR"
    echo "$BACKUP_DIR" > /tmp/last_backup_dir.txt
}

# Validate environment variables
validate_environment() {
    log "Validating environment configuration..."
    
    # Check for GPIO pin conflicts
    declare -A pins
    while IFS='=' read -r key value; do
        if [[ $key =~ _PIN$ ]] && [[ $value =~ ^[0-9]+$ ]] && [[ $value -gt 0 ]]; then
            if [[ -n "${pins[$value]:-}" ]]; then
                error_exit "GPIO pin conflict: Pin $value used by both $key and ${pins[$value]}"
            fi
            pins[$value]=$key
        fi
    done < .env
    
    # Validate required variables
    required_vars=(
        "MQTT_BROKER"
        "MQTT_PORT"
        "MAX_ENGINE_RUNTIME"
        "CONSENSUS_THRESHOLD"
    )
    
    for var in "${required_vars[@]}"; do
        if ! grep -q "^$var=" .env; then
            error_exit "Required variable $var not found in .env"
        fi
    done
    
    success "Environment validation passed"
}

# Deploy services based on mode
deploy_services() {
    local mode=$1
    log "Deploying services in $mode mode..."
    
    case "$mode" in
        staging)
            # Full deployment to staging
            docker-compose -f docker-compose.yml up -d
            ;;
        canary)
            # Deploy only GPIO trigger as canary
            docker-compose -f docker-compose.yml up -d gpio_trigger
            ;;
        production)
            # Rolling deployment
            for service in gpio_trigger fire_consensus camera_detector; do
                log "Updating $service..."
                docker-compose -f docker-compose.yml up -d --no-deps "$service"
                sleep 10  # Wait for service to stabilize
                validate_service "$service" || error_exit "Service $service failed validation"
            done
            ;;
        *)
            error_exit "Unknown deployment mode: $mode"
            ;;
    esac
    
    success "Services deployed in $mode mode"
}

# Validate individual service
validate_service() {
    local service=$1
    log "Validating $service..."
    
    # Check if container is running
    if ! docker-compose ps | grep -q "${service}.*Up"; then
        return 1
    fi
    
    # Check for recent errors
    if docker-compose logs --tail=50 "$service" 2>&1 | grep -qi "error\|exception\|fatal"; then
        warning "Errors found in $service logs"
    fi
    
    # Validate health endpoint if available
    case "$service" in
        gpio_trigger)
            validate_mqtt_health "gpio_trigger" || return 1
            ;;
        camera_detector|fire_consensus)
            validate_mqtt_health "$service" || return 1
            ;;
    esac
    
    return 0
}

# Validate MQTT health reporting
validate_mqtt_health() {
    local service=$1
    
    if ! command -v mosquitto_sub &> /dev/null; then
        warning "Skipping MQTT validation - mosquitto_sub not available"
        return 0
    fi
    
    log "Checking MQTT health for $service..."
    
    # Try to get health message
    if timeout 10 mosquitto_sub -h "${MQTT_BROKER:-localhost}" \
        -p "${MQTT_PORT:-1883}" \
        -t "system/$service/health" -C 1 &>/dev/null; then
        success "$service health reporting confirmed"
        return 0
    else
        # Check legacy topic for GPIO trigger
        if [[ "$service" == "gpio_trigger" ]]; then
            if timeout 10 mosquitto_sub -h "${MQTT_BROKER:-localhost}" \
                -p "${MQTT_PORT:-1883}" \
                -t "system/trigger_telemetry" -C 1 &>/dev/null; then
                warning "$service using legacy health topic"
                return 0
            fi
        fi
        return 1
    fi
}

# Monitor deployment
monitor_deployment() {
    log "Monitoring deployment for $ROLLBACK_TIMEOUT seconds..."
    
    local start_time=$(date +%s)
    local issues=0
    
    while [[ $(($(date +%s) - start_time)) -lt $ROLLBACK_TIMEOUT ]]; do
        # Check all services
        for service in camera_detector fire_consensus gpio_trigger; do
            if ! validate_service "$service"; then
                ((issues++))
                warning "Issue detected with $service (count: $issues)"
                
                if [[ $issues -gt 3 ]]; then
                    error_exit "Too many issues detected - consider rollback"
                fi
            fi
        done
        
        # Show progress
        local elapsed=$(($(date +%s) - start_time))
        echo -ne "\rMonitoring... $elapsed/$ROLLBACK_TIMEOUT seconds (issues: $issues)\033[K"
        
        sleep 10
    done
    
    echo ""  # New line after progress
    success "Deployment monitoring complete - $issues issues detected"
}

# Rollback deployment
rollback() {
    log "Initiating rollback..." "$RED"
    
    # Get last backup directory
    if [[ -f /tmp/last_backup_dir.txt ]]; then
        BACKUP_DIR=$(cat /tmp/last_backup_dir.txt)
        
        if [[ -d "$BACKUP_DIR" ]]; then
            log "Restoring from backup: $BACKUP_DIR"
            
            # Stop current services
            docker-compose down
            
            # Restore configuration
            cp "$BACKUP_DIR"/*.yml . 2>/dev/null || true
            cp "$BACKUP_DIR"/.env* . 2>/dev/null || true
            
            # Restart services
            docker-compose up -d
            
            success "Rollback complete"
        else
            error_exit "Backup directory not found: $BACKUP_DIR"
        fi
    else
        error_exit "No backup information found"
    fi
}

# Generate deployment report
generate_report() {
    log "Generating deployment report..."
    
    REPORT_FILE="deployment_report_$(date +%Y%m%d_%H%M%S).md"
    
    cat > "$REPORT_FILE" << EOF
# Deployment Report

**Date**: $(date)
**Mode**: $DEPLOYMENT_MODE
**Status**: Success

## Service Status
$(docker-compose ps)

## Health Check Results
EOF

    # Add health check results
    for service in camera_detector fire_consensus gpio_trigger; do
        echo "### $service" >> "$REPORT_FILE"
        if validate_service "$service"; then
            echo "✅ Healthy" >> "$REPORT_FILE"
        else
            echo "❌ Issues detected" >> "$REPORT_FILE"
        fi
        echo "" >> "$REPORT_FILE"
    done
    
    # Add recent logs
    echo "## Recent Logs" >> "$REPORT_FILE"
    tail -n 50 "$LOG_FILE" >> "$REPORT_FILE"
    
    success "Report generated: $REPORT_FILE"
}

# Main deployment flow
main() {
    log "Starting deployment process for mode: $DEPLOYMENT_MODE"
    
    # Pre-deployment checks
    check_prerequisites
    validate_environment
    backup_current
    
    # Deploy
    deploy_services "$DEPLOYMENT_MODE"
    
    # Post-deployment validation
    sleep 30  # Wait for services to stabilize
    monitor_deployment
    
    # Generate report
    generate_report
    
    success "Deployment completed successfully!"
    log "Log file: $LOG_FILE"
}

# Handle script arguments
case "${1:-help}" in
    staging|canary|production)
        main
        ;;
    rollback)
        rollback
        ;;
    help|--help|-h)
        echo "Usage: $0 {staging|canary|production|rollback}"
        echo ""
        echo "Deployment modes:"
        echo "  staging    - Deploy all services to staging environment"
        echo "  canary     - Deploy GPIO trigger only as canary"
        echo "  production - Rolling deployment to production"
        echo "  rollback   - Rollback to previous deployment"
        echo ""
        echo "Example:"
        echo "  $0 staging"
        echo "  $0 production"
        echo "  $0 rollback"
        exit 0
        ;;
    *)
        error_exit "Invalid argument: $1 (use --help for usage)"
        ;;
esac