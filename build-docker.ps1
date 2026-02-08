# Optimized Docker build script for PowerShell (Windows)
# This enables BuildKit for faster builds with better caching

Write-Host "Building Docker image with optimizations..." -ForegroundColor Green
Write-Host "This may take 10-15 minutes on first build, but subsequent builds will be faster." -ForegroundColor Yellow

# Enable Docker BuildKit for better caching and parallel builds
$env:DOCKER_BUILDKIT = "1"
$env:COMPOSE_DOCKER_CLI_BUILD = "1"

# Build with BuildKit
docker-compose build --progress=plain

Write-Host "Build complete! You can now run: docker-compose up" -ForegroundColor Green
