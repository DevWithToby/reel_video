"""
FastAPI backend for Viral Reel Recreator
Step 1: API Endpoints
"""
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Body, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Any, List
import uuid
import os
import json
from datetime import datetime
from pipeline import ReelPipeline
from video_downloader import VideoDownloader
import numpy as np


def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to Python native types for JSON serialization.
    Handles numpy bool_, int64, float64, ndarray, etc.
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def log_video_upload(job_id: str, source_type: str, source: str, product_description: str, brand_tone: Optional[str]):
    """Log trending video upload details before processing"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("\n" + "="*70)
    print(f"📥 NEW VIDEO UPLOAD - {timestamp}")
    print("="*70)
    print(f"  Job ID:             {job_id}")
    print(f"  Source Type:        {source_type}")
    print(f"  Source:             {source}")
    print(f"  Product Description: {product_description[:100]}{'...' if len(product_description) > 100 else ''}")
    print(f"  Brand Tone:         {brand_tone or 'Not specified'}")
    print("="*70 + "\n")

app = FastAPI(title="Viral Reel Recreator API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create directories for uploads and outputs
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("temp", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("product_assets", exist_ok=True)

# Initialize pipeline
pipeline = ReelPipeline()

# Initialize video downloader
video_downloader = VideoDownloader()

# In-memory job storage (use database in production)
job_storage = {}


@app.post("/upload")
async def upload_reel(
    request: Request,
    background_tasks: BackgroundTasks,
    video_file: Optional[UploadFile] = File(None),
    video_url: Optional[str] = Form(None),
    product_description: str = Form(...),
    brand_tone: Optional[str] = Form(None),
    product_logo: Optional[UploadFile] = File(None)
):
    """
    Upload a trending reel (file or URL) and product description.
    Returns a job_id for tracking the processing pipeline.
    Processing happens in the background.
    """
    # Validate input: must have either file or URL
    if not video_file and not video_url:
        return JSONResponse({
            "error": "Either video_file or video_url must be provided"
        }, status_code=400)
    
    if video_file and video_url:
        return JSONResponse({
            "error": "Please provide either video_file or video_url, not both"
        }, status_code=400)
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    video_path = None
    
    # Handle video URL
    if video_url:
        try:
            # Download video from URL
            filename = f"{job_id}_video.mp4"
            video_path = video_downloader.download_video(video_url, filename)
        except Exception as e:
            return JSONResponse({
                "error": f"Failed to download video: {str(e)}"
            }, status_code=400)
    
    # Handle file upload
    elif video_file:
        video_path = f"uploads/{job_id}_{video_file.filename}"
        with open(video_path, "wb") as f:
            content = await video_file.read()
            f.write(content)
    
    # Save product images (handle multiple files manually from form data)
    product_image_paths = []
    form = await request.form()
    product_images = form.getlist("product_images")  # Get all files with name "product_images"
    if product_images:
        product_assets_dir = f"product_assets/{job_id}"
        os.makedirs(product_assets_dir, exist_ok=True)
        for idx, img_file in enumerate(product_images):
            if isinstance(img_file, UploadFile) and img_file.filename:
                ext = os.path.splitext(img_file.filename)[1] or '.jpg'
                img_path = f"{product_assets_dir}/product_{idx+1}{ext}"
                with open(img_path, "wb") as f:
                    content = await img_file.read()
                    f.write(content)
                product_image_paths.append(img_path)
    
    # Save product logo
    product_logo_path = None
    if product_logo and product_logo.filename:
        product_assets_dir = f"product_assets/{job_id}"
        os.makedirs(product_assets_dir, exist_ok=True)
        ext = os.path.splitext(product_logo.filename)[1] or '.png'
        product_logo_path = f"{product_assets_dir}/logo{ext}"
        with open(product_logo_path, "wb") as f:
            content = await product_logo.read()
            f.write(content)
    
    # Log the video upload before processing
    source_type = "URL" if video_url else "File Upload"
    source = video_url if video_url else video_file.filename
    log_video_upload(job_id, source_type, source, product_description, brand_tone)
    
    # source_type: "url" or "file" so frontend can show "Download original" when URL
    source_type = "url" if video_url else "file"
    job_storage[job_id] = {
        "job_id": job_id,
        "video_path": video_path,
        "original_video_path": video_path,
        "source_type": source_type,
        "product_description": product_description,
        "brand_tone": brand_tone,
        "product_image_paths": product_image_paths,
        "product_logo_path": product_logo_path,
        "status": "processing"
    }
    
    # Start background processing
    background_tasks.add_task(
        process_reel_background,
        job_id,
        video_path,
        product_description,
        brand_tone,
        source_type,
        product_image_paths,
        product_logo_path
    )
    
    return JSONResponse({
        "job_id": job_id,
        "status": "processing",
        "source_type": source_type,
        "message": "Video uploaded successfully. Processing started."
    })


async def process_reel_background(
    job_id: str,
    video_path: str,
    product_description: str,
    brand_tone: Optional[str],
    source_type: str = "file",
    product_image_paths: List[str] = None,
    product_logo_path: str = None
):
    """Background task to process the reel. Merges result with existing job so original_video_path is kept."""
    try:
        result = pipeline.process(
            job_id, 
            video_path, 
            product_description, 
            brand_tone,
            product_image_paths or [],
            product_logo_path
        )
        result["original_video_path"] = video_path
        result["source_type"] = source_type
        job_storage[job_id] = result
    except Exception as e:
        existing = job_storage.get(job_id, {})
        job_storage[job_id] = {
            **existing,
            "job_id": job_id,
            "status": "failed",
            "error": str(e)
        }


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """
    Check the status of a processing job.
    """
    if job_id not in job_storage:
        return JSONResponse({
            "job_id": job_id,
            "status": "not_found"
        }, status_code=404)
    
    job_data = job_storage[job_id]
    # Convert numpy types to Python native types for JSON serialization
    job_data = convert_numpy_types(job_data)
    return JSONResponse(job_data)


@app.get("/download/{job_id}/original")
async def download_original_video(job_id: str):
    """
    Download the original video (from URL or file upload).
    Available as soon as the job exists, before or during processing.
    """
    if job_id not in job_storage:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    job_data = job_storage[job_id]
    path = job_data.get("original_video_path") or job_data.get("video_path")
    if not path or not os.path.exists(path):
        return JSONResponse({"error": "Original video file not found"}, status_code=404)
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"{job_id}_original.mp4"
    )


@app.post("/download-from-url")
async def download_from_url(
    background_tasks: BackgroundTasks,
    video_url: str = Body(..., embed=True)
):
    """
    Download the video from the given URL and return it as a file.
    No job or form submission required; user can download right after pasting the URL.
    """
    video_url = video_url.strip()
    if not video_url:
        return JSONResponse({"error": "video_url is required"}, status_code=400)
    try:
        from urllib.parse import urlparse
        urlparse(video_url)
    except Exception:
        return JSONResponse({"error": "Invalid URL"}, status_code=400)
    # Use temp dir and unique filename; clean up after sending
    fetch_dir = os.path.join("temp", "fetch")
    os.makedirs(fetch_dir, exist_ok=True)
    filename = f"{uuid.uuid4()}_fetch.mp4"
    try:
        from video_downloader import VideoDownloader
        fetch_downloader = VideoDownloader(download_dir=fetch_dir)
        path = fetch_downloader.download_video(video_url, filename)
    except Exception as e:
        return JSONResponse(
            {"error": f"Failed to download video: {str(e)}"},
            status_code=400
        )
    if not path or not os.path.exists(path):
        return JSONResponse({"error": "Downloaded file not found"}, status_code=500)

    def cleanup(p: str):
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    background_tasks.add_task(cleanup, path)
    return FileResponse(
        path,
        media_type="video/mp4",
        filename="downloaded_reel.mp4"
    )


@app.get("/download/{job_id}")
async def download_video(job_id: str):
    """
    Download the final rendered video.
    """
    if job_id not in job_storage:
        return JSONResponse({
            "error": "Job not found"
        }, status_code=404)
    
    job_data = job_storage[job_id]
    
    if job_data.get("status") != "completed":
        return JSONResponse({
            "error": f"Job status: {job_data.get('status')}"
        }, status_code=400)
    
    video_path = job_data.get("output_video_path")
    if not video_path or not os.path.exists(video_path):
        return JSONResponse({
            "error": "Video file not found"
        }, status_code=404)
    
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"{job_id}_reel.mp4"
    )


@app.get("/")
async def root():
    """Serve the frontend"""
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

