# Contributing to Wildfire Watch

Thank you for your interest in improving Wildfire Watch! This guide covers development setup, coding standards, and submission process.

## Development Setup

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- Git
- Hardware for testing (Coral/Hailo/GPIO optional)

### Clone and Setup

```bash
# Fork on GitHub first
git clone https://github.com/seth-planet/wildfire-watch.git
cd wildfire-watch

# Add upstream
git remote add upstream https://github.com/seth-planet/wildfire-watch.git

# Create development environment
cp .env.example .env.dev
echo "COMPOSE_FILE=docker-compose.yml:docker-compose.dev.yml" >> .env.dev
```

### Development Environment

```bash
# Start services in development mode
docker-compose --env-file .env.dev up

# Run with code mounting for hot reload
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Run specific service
docker-compose up mqtt-broker camera-detector
```

## Code Style

### Python
- Follow PEP 8
- Use type hints
- Maximum line length: 100 characters
- Use f-strings for formatting

### Docker
- Use specific base image versions
- Multi-stage builds for smaller images
- Run as non-root user
- Use .dockerignore

```dockerfile
# Good
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.11-slim
COPY --from=builder /root/.local /root/.local
```

### Shell Scripts
- Use `#!/usr/bin/env bash`
- Set `set -euo pipefail`
- Quote variables
- Check command existence

```bash
# Good
#!/usr/bin/env bash
set -euo pipefail

if ! command -v docker &> /dev/null; then
    echo "Docker not found"
    exit 1
fi
```

## Making Changes

### Branch Strategy

```bash
# Feature branch
git checkout -b feature/add-thermal-camera-support

# Bugfix branch  
git checkout -b fix/mqtt-reconnection-issue

# Keep updated with upstream
git fetch upstream
git rebase upstream/main
```

### Commit Messages

Follow conventional commits:
```
type(scope): subject

body

footer
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style
- `refactor`: Refactoring
- `test`: Tests
- `chore`: Maintenance

Examples:
```bash
git commit -m "feat(camera): add ONVIF PTZ support"
git commit -m "fix(consensus): handle network timeouts"
git commit -m "docs: update Raspberry Pi setup guide"
```

### Code Review Checklist

Before submitting PR:
- [ ] Tests pass
- [ ] Documentation updated
- [ ] No hardcoded values
- [ ] Error handling added
- [ ] Logging appropriate
- [ ] Performance considered
- [ ] Security reviewed

## Adding Features

### New Camera Support

1. Add discovery method in `camera_detector/detect.py`
2. Update documentation
3. Add tests
4. Submit PR

Example:
```python
def discover_hikvision_cameras(self):
    """Discover Hikvision cameras using SADP protocol."""
    # Implementation
    pass
```

### New AI Model

1. Add model to `models/`
2. Update `security_nvr/model_config.yml`
3. Test inference speed
4. Document performance

### New Hardware Support

1. Add detection in `security_nvr/hardware_detect.py`
2. Update configuration
3. Test on actual hardware
4. Update compatibility matrix

## Documentation

### Code Documentation

```python
def calculate_consensus(detections: List[Detection]) -> ConsensusResult:
    """
    Calculate multi-camera consensus for fire detection.
    
    Args:
        detections: List of Detection objects from cameras
        
    Returns:
        ConsensusResult with decision and confidence
        
    Raises:
        ValueError: If detections list is empty
        
    Example:
        >>> detections = [Detection(...), Detection(...)]
        >>> result = calculate_consensus(detections)
        >>> print(result.should_trigger)
    """
```

### README Updates

- Keep concise
- Include examples
- Update table of contents
- Test all commands

## Release Process

### Version Numbering

Follow semantic versioning:
- MAJOR.MINOR.PATCH
- 1.0.0 → 1.0.1 (bugfix)
- 1.0.0 → 1.1.0 (feature)
- 1.0.0 → 2.0.0 (breaking)

### Release Checklist

1. Update version in:
   - `docker-compose.yml`
   - `balena.yml`
   - `README.md`

2. Update CHANGELOG.md

3. Run full test suite

4. Tag release:
   ```bash
   git tag -a v1.2.0 -m "Release version 1.2.0"
   git push upstream v1.2.0
   ```

## Getting Help

### Development Chat
- Discord: #development channel
- GitHub Discussions: Technical questions

### Debugging

```bash
# Enable verbose logging
LOG_LEVEL=DEBUG docker-compose up

# Access service shell
docker exec -it camera-detector /bin/bash

# View real-time logs
docker logs -f fire-consensus
```

### Common Issues

**Import errors:**
- Check PYTHONPATH
- Verify requirements.txt

**Docker build failures:**
- Clear cache: `docker system prune`
- Check base image availability

**Test failures:**
- Check test database
- Verify mock setup

## Code of Conduct

- Be respectful
- Welcome newcomers
- Focus on what's best for community
- Show empathy

## Recognition

Contributors are recognized in:
- README.md contributors section
- Release notes
- Annual community report

Thank you for contributing to wildfire safety!
