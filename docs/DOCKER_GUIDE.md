# Docker Development Workflow

This document describes common Docker routines for building, testing, and deploying CivicShield AI services.

## Local Image Build

```bash
# Build FastAPI service image
docker build -t civicshield-api:latest -f Dockerfile .

# Run container locally
docker run -p 8000:8000 civicshield-api:latest
```

## Docker Compose Orchestration

The `docker-compose.yml` launches the API, PostgreSQL database, and Redis cache:

```bash
# Start all services in the background
docker-compose up -d

# Check running container health
docker-compose ps

# Tail application logs
docker-compose logs -f api
```

## Container Health Checks

The production image incorporates an automated health probe hitting `http://localhost:8000/health`:
- Interval: 30s
- Timeout: 5s
- Retries: 3
