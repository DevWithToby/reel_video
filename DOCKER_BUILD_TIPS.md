# Docker Build Optimization Tips

## Quick Build (Recommended)

Use the optimized build script for Windows:

```powershell
.\build-docker.ps1
```

Or manually with BuildKit enabled:

```powershell
$env:DOCKER_BUILDKIT = "1"
$env:COMPOSE_DOCKER_CLI_BUILD = "1"
docker-compose build
```

## Why the Build is Slow

The Docker build downloads several large packages:
- **opencv-python**: ~60 MB (largest package)
- **moviepy**: ~20 MB
- **scipy**: ~15 MB
- **librosa**: ~10 MB
- Plus system dependencies (~200 MB)

**Total download**: ~300+ MB on first build

## Build Time Expectations

- **First build**: 10-20 minutes (depending on internet speed)
- **Subsequent builds**: 2-5 minutes (uses cached layers)

## Optimizations Applied

1. ✅ **Staged installation**: Packages installed in stages for better caching
2. ✅ **Increased timeouts**: 900 seconds for large packages
3. ✅ **More retries**: 10 retries for failed downloads
4. ✅ **Prefer binary wheels**: Uses pre-built packages (faster)
5. ✅ **BuildKit enabled**: Better parallel builds and caching

## Troubleshooting

### If build times out:
1. Check your internet connection
2. Try building again (it will resume from cached layers)
3. Use a faster internet connection if possible

### If specific package fails:
The Dockerfile has retry logic - it will automatically retry failed packages.

### To rebuild from scratch:
```powershell
docker-compose build --no-cache
```

## Alternative: Use Pre-built Base Image

If builds are consistently slow, consider using a pre-built Python image with common packages already installed.
