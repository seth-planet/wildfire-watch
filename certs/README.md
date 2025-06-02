# ⚠️ Default Certificates - INSECURE BY DESIGN ⚠️

This directory contains **intentionally insecure** default certificates that are part of the public repository. They exist solely to make initial deployment easy - the system will work "out of the box" without any certificate configuration.

## 🚨 SECURITY WARNING 🚨

**THESE CERTIFICATES ARE PUBLIC AND PROVIDE NO SECURITY!**

- The private keys are available to anyone who downloads this repository
- Anyone can decrypt your MQTT traffic if you use these certificates
- Anyone can impersonate your services
- **DO NOT USE IN PRODUCTION**

## Files in this Directory

- `ca.crt` - Default Certificate Authority (PUBLIC)
- `ca.key` - CA private key with password 'wildfire' (PUBLIC)
- `server.crt` - Default MQTT broker certificate
- `server.key` - Server private key (PUBLIC)
- `*.crt/key` - Default client certificates for services

## For Development/Testing

These certificates are perfect for:
- Local development
- Testing deployments
- Learning the system
- Demos and proof-of-concepts

## For Production Use

**You MUST generate your own certificates:**

```bash
# Generate secure certificates
cd wildfire-watch
./scripts/generate_certs.sh custom

# Follow the prompts to create secure certificates
# The script will ask for:
# - CA password (keep this safe!)
# - Organization details
# - Server hostnames/IPs
```

## How to Replace Default Certificates

1. **Generate Custom Certificates**
   ```bash
   ./scripts/generate_certs.sh custom
   ```

2. **Deploy to Devices**
   ```bash
   # Copy to all devices
   ./scripts/provision_certs.sh auto device1.local device2.local
   
   # Or manually copy certs/ directory to /mnt/data/certs/
   ```

3. **Restart Services**
   ```bash
   docker-compose restart
   ```

## Certificate Details

**Default CA Password**: `wildfire` (yes, this is public)

**Certificate Validity**: 
- CA: 10 years from generation
- Server/Client: 5 years from generation

**Subject Names**:
- CA: `/C=US/ST=CA/O=Wildfire Watch OSS/CN=Default Development CA`
- Server: `/C=US/ST=CA/O=Wildfire Watch OSS/CN=mqtt-broker-default`

**Supported Domains** (in server certificate):
- mqtt_broker
- mqtt_broker.local
- localhost
- *.local
- *.wildfire-watch.local
- All private IP ranges (10.x, 172.16.x, 192.168.x)

## Why Include Insecure Certificates?

1. **Zero-Configuration Testing**: New users can test the system immediately
2. **Learning Curve**: Focus on understanding the system before security
3. **Development Speed**: Developers don't need to generate certs to contribute
4. **Troubleshooting**: Removes TLS as a variable when debugging

## Security Best Practices

When you're ready for production:

1. **Generate unique certificates** for your deployment
2. **Use strong CA password** (20+ characters recommended)
3. **Limit certificate scope** to your actual hostnames/IPs
4. **Store CA key offline** after generating certificates
5. **Rotate certificates** every 1-2 years
6. **Monitor for unauthorized connections**
7. **Never commit custom certificates** to version control

## Remember

These default certificates are like the default password "admin/admin" - they get you started, but you must change them before real use!
