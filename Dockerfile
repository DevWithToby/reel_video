# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for video/audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    imagemagick \
    nodejs \
    npm \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    fonts-dejavu-core \
    fonts-liberation \
    fonts-freefont-ttf \
    && rm -rf /var/lib/apt/lists/*

# Configure ImageMagick policy to allow text rendering
# This is needed for MoviePy's TextClip to work
# Allow reading/writing temporary files and fonts
RUN for policy_file in /etc/ImageMagick-*/policy.xml; do \
        if [ -f "$policy_file" ]; then \
            sed -i 's/<policy domain="path" rights="none" pattern="@\*"/<policy domain="path" rights="read|write" pattern="@\*"/' "$policy_file" && \
            sed -i 's/<policy domain="coder" rights="none" pattern="PDF"/<policy domain="coder" rights="read|write" pattern="PDF"/' "$policy_file" && \
            sed -i 's/<policy domain="coder" rights="none" pattern="PS"/<policy domain="coder" rights="read|write" pattern="PS"/' "$policy_file"; \
        fi; \
    done || echo "ImageMagick policy configuration completed"

# Upgrade pip to latest version for better timeout handling
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies in stages for better caching and reliability
# Stage 1: Install small, fast packages first
RUN pip install --no-cache-dir --timeout=600 --retries=10 \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    python-multipart==0.0.6 \
    pydantic==2.5.0 \
    aiofiles==23.2.1

# Stage 2: Install medium packages
RUN pip install --no-cache-dir --timeout=600 --retries=10 \
    numpy==1.24.3 \
    pillow==10.1.0 \
    httpx

# Stage 3: Install scientific computing packages
RUN pip install --no-cache-dir --timeout=600 --retries=10 \
    scipy==1.11.4 \
    librosa==0.10.1 \
    soundfile==0.12.1

# Stage 4: Install video processing packages (largest, most likely to timeout)
# Using --prefer-binary for faster installation of pre-built wheels
RUN pip install --no-cache-dir --prefer-binary --timeout=900 --retries=10 moviepy==1.0.3 || \
    (sleep 5 && pip install --no-cache-dir --prefer-binary --timeout=900 --retries=10 moviepy==1.0.3)

# Stage 5: Install opencv (largest package ~60MB, install last)
# Using --prefer-binary to use pre-built wheel instead of building from source
RUN pip install --no-cache-dir --prefer-binary --timeout=900 --retries=10 opencv-python==4.8.1.78 || \
    (sleep 10 && pip install --no-cache-dir --prefer-binary --timeout=900 --retries=10 opencv-python==4.8.1.78)

# Stage 6: Install OpenAI (should be fast)
RUN pip install --no-cache-dir --timeout=600 --retries=10 "openai>=1.12.0"

# Stage 7: Install yt-dlp for video URL downloading
RUN pip install --no-cache-dir --timeout=600 --retries=10 "yt-dlp>=2024.1.0"

# Stage 8: Install Hugging Face hub and requests for AI image generation
RUN pip install --no-cache-dir --timeout=600 --retries=10 \
    "huggingface_hub>=0.20.0" \
    "requests>=2.31.0"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads outputs temp logs static

# Expose FastAPI port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

