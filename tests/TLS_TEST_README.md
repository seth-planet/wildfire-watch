# TLS Test Guide

## Overview
The wildfire-watch project includes comprehensive TLS/SSL tests for secure MQTT communication. Many of these tests require specific infrastructure setup to run successfully.

## Test Categories

### 1. Certificate Management Tests (Always Run)
These tests verify certificate files exist and are valid:
- `test_certificates_exist`
- `test_certificate_validity`
- `test_default_certificate_warning`

### 2. Service Configuration Tests (Always Run)
These tests verify service configuration without requiring infrastructure:
- `test_services_read_tls_config`
- `test_docker_compose_tls_config`
- `test_services_mount_certificates`
- `test_generate_certs_script_exists`

### 3. TLS Connection Tests (Require Infrastructure)
These tests are SKIPPED unless TLS infrastructure is available:
- `test_tls_port_listening`
- `test_tls_connection_with_valid_cert`
- `test_tls_publish_subscribe`
- `test_service_tls_connection`
- `test_invalid_certificate_rejected`

## Running TLS Tests

### Basic Test Run (No Infrastructure Required)
```bash
pytest tests/test_tls_integration_consolidated.py -v
```

This will run certificate and configuration tests, skipping connection tests.

### Full TLS Test Run (Infrastructure Required)

1. **Generate Certificates** (if not already present):
```bash
./scripts/generate_certs.sh custom
```

2. **Start MQTT Broker with TLS**:
```bash
MQTT_TLS=true docker-compose up mqtt_broker -d
```

3. **Run Tests**:
```bash
MQTT_TLS=true pytest tests/test_tls_integration_consolidated.py -v
```

## Environment Variables

- `MQTT_TLS=true` - Enable TLS for MQTT connections
- `MQTT_HOST=localhost` - MQTT broker hostname
- `MQTT_PORT=1883` - Non-TLS MQTT port
- `MQTT_TLS_PORT=8883` - TLS-enabled MQTT port

## Certificate Requirements

Certificates must be present in the `certs/` directory:
- `ca.crt` - Certificate Authority
- `server.crt` - Server certificate
- `server.key` - Server private key
- `client.crt` - Client certificate (optional)
- `client.key` - Client private key (optional)

## Skip Conditions

Tests are automatically skipped when:
1. MQTT broker is not running on TLS port (8883)
2. Required certificate files are missing
3. `MQTT_TLS` environment variable is not set to 'true'

## Troubleshooting

### "MQTT TLS not available" Skip Message
- Ensure MQTT broker is running with TLS enabled
- Check port 8883 is accessible
- Verify `MQTT_TLS=true` is set

### Certificate Errors
- Run `./scripts/generate_certs.sh custom` to generate certificates
- Ensure certificates are not expired
- Check file permissions on certificate files

### Connection Failures
- Verify MQTT broker configuration includes TLS settings
- Check firewall rules for port 8883
- Ensure certificate CN matches hostname

## Production vs Development

- **Development**: Use default certificates (marked as insecure)
- **Production**: Generate custom certificates with proper CN and security
- **Testing**: Either certificate set works, but custom is recommended

## Integration with CI/CD

For CI/CD pipelines:
1. Skip TLS connection tests if infrastructure is not available
2. Always run certificate validation and configuration tests
3. Consider a separate TLS test job with full infrastructure