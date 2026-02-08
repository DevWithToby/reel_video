# Viral Reel Recreator

A system that extracts structural style from trending Instagram Reels and generates new product ads following the same pacing, timing, and motion patterns - **without copying content**.

## Core Principle

> **LLM reasons about structure. Code handles pixels. Never mix the two.**

The system never sends raw video to the LLM. Instead, it extracts objective style signals (timing, pacing, motion) as JSON, uses the LLM to adapt this structure for a product, then programmatically renders a new video.

## Architecture

```
Input Video (MP4)
→ Style Extraction (CV + Audio)
→ Style Blueprint JSON
→ LLM Creative Expansion
→ Ad Blueprint JSON
→ Video Renderer
→ Final Reel (MP4, 9:16)
```

## Features

- ✅ **Style Extraction**: Analyzes video structure (shots, motion, timing) and audio (BPM, beats)
- ✅ **LLM Creative Director**: Adapts structure for new products without copying content
- ✅ **Asset Generation**: Creates images for each shot (MVP uses placeholders)
- ✅ **Video Rendering**: Renders 9:16 videos with motion effects and text overlays
- ✅ **Guardrails**: Adds timing jitter, logs all outputs, ensures originality
- ✅ **Docker Support**: Run everything in containers - no local installation needed
- ✅ **Web Interface**: Beautiful, modern UI for easy video creation

## Installation

### Option 1: Docker (Recommended - No local installation needed)

1. Install Docker and Docker Compose:
   - [Docker Desktop](https://www.docker.com/products/docker-desktop) (includes Docker Compose)

2. Create `.env` file with your OpenAI API key:
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

3. Build and run with Docker Compose:
```bash
docker-compose up --build
```

The application will be available at `http://localhost:8000`

### Option 2: Local Installation

1. Install Python 3.10+

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### With Docker:

```bash
# Start the container
docker-compose up

# Or run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

The server will start on `http://localhost:8000`

### Without Docker:

```bash
# Start the server
uvicorn main:app --reload
```

The server will start on `http://localhost:8000`

### Web Interface (Recommended)

1. Open your browser and navigate to `http://localhost:8000`
2. Upload a trending reel video (MP4 format)
3. Enter your product description
4. Optionally specify brand tone
5. Click "Create Reel" and wait for processing
6. Download your generated reel when ready!

### API Usage (Alternative)

#### Upload a reel:

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "video_file=@trending_reel.mp4" \
  -F "product_description=A magnetic cable organizer that keeps your desk clean" \
  -F "brand_tone=playful and modern"
```

Response:
```json
{
  "job_id": "uuid-here",
  "status": "processing",
  "message": "Video uploaded successfully. Processing started."
}
```

#### Check status:

```bash
curl "http://localhost:8000/status/{job_id}"
```

#### Download result:

```bash
curl "http://localhost:8000/download/{job_id}" --output result.mp4
```

## Quick Start (Docker)

```bash
# 1. Clone or download this repository
cd video_reel

# 2. Create .env file with your OpenAI API key
echo "OPENAI_API_KEY=your-api-key-here" > .env

# 3. Build and start
docker-compose up --build

# 4. Open browser to http://localhost:8000
```

That's it! No Python, pip, or other dependencies needed.

## Project Structure

```
video_reel/
├── main.py                 # FastAPI endpoints
├── style_extractor.py      # Step 2: Video/audio analysis
├── llm_director.py         # Step 3: LLM creative director
├── asset_generator.py      # Step 4: Image generation
├── video_renderer.py       # Step 5: Video rendering
├── pipeline.py             # Step 6: Orchestration
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker container definition
├── docker-compose.yml      # Docker Compose configuration
├── .dockerignore           # Files to exclude from Docker build
├── static/                 # Frontend files
│   ├── index.html          # Main HTML page
│   ├── styles.css          # Styling
│   └── app.js              # Frontend JavaScript
├── uploads/                # Uploaded videos (created at runtime)
├── outputs/                # Rendered videos (created at runtime)
├── temp/                   # Temporary assets (created at runtime)
└── logs/                   # JSON blueprints and logs (created at runtime)
```

## API Endpoints

- `GET /` - Serve frontend web interface
- `POST /upload` - Upload video and product description
- `GET /status/{job_id}` - Check processing status
- `GET /download/{job_id}` - Download final video
- `GET /health` - Health check

## MVP Limitations

- Asset generation uses placeholder images (integrate DALL-E/Stable Diffusion for production)
- Text detection is heuristic-based (use OCR for production)
- Motion detection is simplified (use optical flow for production)
- Background music not automatically extracted (manual upload)

## Next Steps for Production

1. Integrate DALL-E or Stable Diffusion for image generation
2. Add OCR for better text overlay detection
3. Implement optical flow for precise motion detection
4. Add automatic background music extraction
5. Use database for job storage instead of in-memory
6. Add authentication and rate limiting
7. Implement video quality optimization

## License

MIT

