# Docker Guide

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop) installed and running
- OpenAI API key

## Quick Start

1. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

2. **Build and run:**
   ```bash
   docker-compose up --build
   ```

3. **Access the application:**
   Open your browser to `http://localhost:8000`

## Docker Commands

### Start the container:
```bash
docker-compose up
```

### Start in background (detached mode):
```bash
docker-compose up -d
```

### View logs:
```bash
docker-compose logs -f
```

### Stop the container:
```bash
docker-compose down
```

### Stop and remove volumes (cleans up uploads/outputs):
```bash
docker-compose down -v
```

### Rebuild after code changes:
```bash
docker-compose up --build
```

### Execute commands inside container:
```bash
docker-compose exec viral-reel-recreator bash
```

## Volume Mounts

The following directories are mounted as volumes to persist data:
- `./uploads` - Uploaded videos
- `./outputs` - Generated videos
- `./temp` - Temporary processing files
- `./logs` - Processing logs and blueprints

## Environment Variables

Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your-api-key-here
```

## Troubleshooting

### Container won't start
- Check if port 8000 is already in use: `netstat -an | grep 8000`
- Check Docker logs: `docker-compose logs`

### Out of memory errors
- Video processing can be memory-intensive
- Increase Docker Desktop memory limit (Settings → Resources → Memory)

### Slow processing
- Ensure Docker has enough CPU cores allocated
- Check Docker Desktop resource settings

### Permission errors (Linux/Mac)
- Ensure Docker has permission to access the mounted volumes
- Try: `sudo chown -R $USER:$USER uploads outputs temp logs`

## Building Without Docker Compose

If you prefer to use Docker directly:

```bash
# Build the image
docker build -t viral-reel-recreator .

# Run the container
docker run -d \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your-api-key-here \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/temp:/app/temp \
  -v $(pwd)/logs:/app/logs \
  --name viral-reel-recreator \
  viral-reel-recreator
```


