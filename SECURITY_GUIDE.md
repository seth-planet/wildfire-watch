# 🔐 Security Guide - Wildfire Watch

## Overview

Wildfire Watch ships with **default security settings** that prioritize ease of deployment. This means:

✅ **Works out of the box** - No configuration needed for testing
⚠️ **Not secure by default** - Must be hardened for production
🔧 **Easy to secure** - Simple scripts to generate custom certificates

## Default Security Configuration

### What's Included

1. **Default TLS Certificates** (INSECURE)
   - Located in `certs/default/`
   - Password for CA: `wildfire` (public)
   - Provides encryption but NO authentication
   - Anyone can decrypt traffic with these certs

2. **Open MQTT Access**
   - No authentication required
   - Any device can publish/subscribe
   - Suitable for isolated networks only

3. **Automatic Service Discovery**
   - mDNS/Avahi enabled
   - Services advertise on local network
   - Convenient but reveals system presence

## Production Security Checklist

### 🚨 Critical (Do First)

- [ ] **Replace default certificates**
  ```bash
  ./scripts/generate_certs.sh custom
  ```

- [ ] **Deploy custom certificates**
  ```bash
  ./scripts/provision_certs.sh auto all-devices
  ```

- [ ] **Restart all services**
  ```bash
  docker-compose restart
  ```

### 🔒 Important (Do Next)

- [ ] **Enable MQTT authentication**
  ```bash
  # Create password file
  docker exec mqtt_broker mosquitto_passwd -c /mosquitto/config/passwd admin
  
  # Update mosquitto.conf
  echo "password_file /mosquitto/config/passwd" >> mosquitto.conf
  echo "allow_anonymous false" >> mosquitto.conf
  ```

- [ ] **Limit network access**
  ```bash
  # Firewall rules (example for iptables)
  iptables -A INPUT -p tcp --dport 1883 -s 192.168.1.0/24 -j ACCEPT
  iptables -A INPUT -p tcp --dport 1883 -j DROP
  ```

- [ ] **Disable unnecessary services**
  ```yaml
  # In docker-compose.yml, comment out:
  # - Avahi/mDNS if not needed
  # - WebSocket listener if not using web UI
  ```

### 🛡️ Recommended (Additional Hardening)

- [ ] **Network isolation**
  - Place cameras on separate VLAN
  - Use firewall between camera and control networks
  - Limit outbound connections

- [ ] **Certificate management**
  - Store CA key offline after generating certs
  - Use unique client certificates per device
  - Implement certificate rotation schedule

- [ ] **Monitoring**
  - Enable MQTT logging
  - Monitor for failed authentication
  - Alert on new device connections

## Step-by-Step Security Setup

### 1. Generate Secure Certificates

```bash
cd wildfire-watch
./scripts/generate_certs.sh custom

# You'll be prompted for:
# - CA password (use 20+ characters)
# - Organization details
# - Server hostnames/IPs
```

### 2. Secure the CA Key

```bash
# After generating certificates, secure the CA key
mv certs/ca.key /secure/offline/storage/
# Only bring it back when generating new certificates
```

### 3. Deploy Certificates

```bash
# Option A: Automated deployment
./scripts/provision_certs.sh auto mqtt-broker.local camera1.local

# Option B: Manual deployment
scp -r certs/* root@device:/mnt/data/certs/
```

### 4. Enable Authentication

```bash
# Create admin user
docker exec mqtt_broker mosquitto_passwd -c /mosquitto/config/passwd admin

# Create service accounts
docker exec mqtt_broker mosquitto_passwd -b /mosquitto/config/passwd camera1 camera1pass
docker exec mqtt_broker mosquitto_passwd -b /mosquitto/config/passwd frigate frigatepass

# Update configuration
# Edit mqtt_broker/mosquitto.conf:
allow_anonymous false
password_file /mosquitto/config/passwd
```

### 5. Configure ACLs

Create `mqtt_broker/acl.conf`:
```conf
# Admin can access everything
user admin
topic readwrite #

# Cameras can only publish detections
user camera1
topic write fire/detection/camera1
topic read fire/trigger

# Frigate can publish events and read config
user frigate
topic write frigate/events
topic read frigate/config/#
```

### 6. Network Security

```bash
# Example firewall rules
# Allow MQTT only from local network
ufw allow from 192.168.1.0/24 to any port 1883
ufw allow from 192.168.1.0/24 to any port 8883

# Block everything else
ufw deny 1883
ufw deny 8883
```

## Security Levels

### 🟢 Development/Testing
- Use default certificates
- No authentication
- Full network access
- All services enabled

### 🟡 Home/Private Network
- Replace default certificates
- Consider authentication
- Limit to local network
- Disable unused services

### 🔴 Internet-Exposed/Commercial
- Custom certificates required
- Strong authentication mandatory
- Strict firewall rules
- Network isolation
- Regular security audits
- Certificate rotation
- Intrusion detection

## Common Security Mistakes

### ❌ Don't Do This

1. **Using default certs in production**
   - Anyone can decrypt your traffic
   - System vulnerable to impersonation

2. **Exposing MQTT to internet without auth**
   - Bots scan for open MQTT brokers
   - Can be used for attacks

3. **Weak passwords**
   - Avoid: admin/admin, password123
   - Use: Complex 20+ character passwords

4. **Not updating certificates**
   - Expired certs cause outages
   - Old certs may have vulnerabilities

### ✅ Do This Instead

1. **Generate unique certificates**
   - Each deployment gets custom certs
   - Strong CA password

2. **Use authentication + TLS**
   - Defense in depth
   - Encrypted and authenticated

3. **Strong unique passwords**
   - Use password manager
   - Different password per service

4. **Certificate rotation plan**
   - Calendar reminders
   - Tested replacement procedure

## Troubleshooting Security Issues

### Certificate Problems

```bash
# Test TLS connection
openssl s_client -connect mqtt_broker:8883 -CAfile certs/ca.crt

# Verify certificate
openssl x509 -in certs/server.crt -text -noout

# Check certificate dates
openssl x509 -in certs/server.crt -dates -noout
```

### Authentication Issues

```bash
# Test MQTT with auth
mosquitto_pub -h mqtt_broker -p 8883 \
  --cafile certs/ca.crt \
  -u admin -P adminpass \
  -t test -m "hello"

# Check password file
docker exec mqtt_broker cat /mosquitto/config/passwd
```

### Permission Problems

```bash
# Fix certificate permissions
chmod 600 certs/*.key
chmod 644 certs/*.crt

# Fix directory permissions
chmod 755 /mnt/data/certs
```

## Security Resources

### Certificate Management
- [OpenSSL Cookbook](https://www.feistyduck.com/library/openssl-cookbook/)
- [Let's Encrypt](https://letsencrypt.org/) (for internet-facing deployments)
- [Certificate Transparency](https://certificate.transparency.dev/)

### MQTT Security
- [MQTT Security Fundamentals](https://www.hivemq.com/blog/mqtt-security-fundamentals/)
- [Eclipse Mosquitto Security](https://mosquitto.org/documentation/authentication-methods/)
- [OWASP IoT Security](https://owasp.org/www-project-internet-of-things/)

### Network Security
- [VLAN Setup Guide](https://www.cisco.com/c/en/us/support/docs/lan-switching/vlan/10023-3.html)
- [pfSense Firewall](https://www.pfsense.org/)
- [Fail2ban](https://www.fail2ban.org/) for intrusion prevention

## Getting Help

If you need security assistance:

1. **Check logs** for certificate/auth errors
2. **Test with openssl** to verify TLS
3. **Review firewall rules** for access issues
4. **Ask in discussions** (don't post private keys!)
5. **Consider professional audit** for commercial use

Remember: Security is a process, not a destination. Start with the basics and improve over time!
