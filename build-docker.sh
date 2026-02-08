#!/bin/bash
# Optimized Docker build script with BuildKit for faster builds

# Enable Docker BuildKit for better caching and parallel builds
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

echo "Building Docker image with optimizations..."
echo "This may take 10-15 minutes on first build, but subsequent builds will be faster."

# Build with BuildKit
docker-compose build --progress=plain

echo "Build complete! You can now run: docker-compose up"
