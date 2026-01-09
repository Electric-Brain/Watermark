from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone
import cv2
import numpy as np
import ffmpeg
import tempfile
import shutil
import asyncio
from concurrent.futures import ThreadPoolExecutor

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create upload directory
UPLOAD_DIR = ROOT_DIR / 'uploads'
PROCESSED_DIR = ROOT_DIR / 'processed'
UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=2)

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Models
class VideoUploadResponse(BaseModel):
    video_id: str
    filename: str
    size: int
    duration: float
    width: int
    height: int
    fps: float

class WatermarkRemovalRequest(BaseModel):
    video_id: str
    x: int
    y: int
    width: int
    height: int

class ProcessingStatus(BaseModel):
    video_id: str
    status: str  # 'processing', 'completed', 'failed'
    progress: float
    message: Optional[str] = None
    output_path: Optional[str] = None

# Store processing status in memory (in production, use Redis or database)
processing_status = {}

def get_video_info(video_path: str):
    """Extract video metadata using ffprobe"""
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        
        if not video_stream:
            raise ValueError("No video stream found")
        
        duration = float(probe['format']['duration'])
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        
        # Calculate FPS
        fps_str = video_stream.get('r_frame_rate', '30/1')
        num, den = map(int, fps_str.split('/'))
        fps = num / den if den != 0 else 30.0
        
        return {
            'duration': duration,
            'width': width,
            'height': height,
            'fps': fps
        }
    except Exception as e:
        logger.error(f"Error getting video info: {str(e)}")
        raise

def remove_watermark_from_frame(frame, x, y, w, h):
    """Remove watermark from a single frame using inpainting"""
    # Create a mask for the watermark region
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask[y:y+h, x:x+w] = 255
    
    # Apply inpainting using Telea method (fast and effective)
    result = cv2.inpaint(frame, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    return result

def process_video_frames(input_path, output_path, x, y, w, h, video_id):
    """Process video and remove watermark from all frames"""
    try:
        # Update status
        processing_status[video_id]['status'] = 'processing'
        processing_status[video_id]['progress'] = 0
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create temporary directory for frames
        temp_dir = tempfile.mkdtemp()
        frames_dir = Path(temp_dir) / 'frames'
        frames_dir.mkdir(exist_ok=True)
        
        logger.info(f"Processing {total_frames} frames...")
        
        # Process frames
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Remove watermark
            processed_frame = remove_watermark_from_frame(frame, x, y, w, h)
            
            # Save frame
            frame_path = frames_dir / f'frame_{frame_count:06d}.png'
            cv2.imwrite(str(frame_path), processed_frame)
            
            frame_count += 1
            
            # Update progress
            progress = (frame_count / total_frames) * 100
            processing_status[video_id]['progress'] = progress
            
            if frame_count % 30 == 0:
                logger.info(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")
        
        cap.release()
        
        logger.info("Reconstructing video with audio...")
        
        # Reconstruct video with audio using ffmpeg
        temp_video = Path(temp_dir) / 'temp_video.mp4'
        
        # Create video from frames
        (
            ffmpeg
            .input(str(frames_dir / 'frame_%06d.png'), framerate=fps)
            .output(str(temp_video), vcodec='libx264', pix_fmt='yuv420p', crf=23)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        
        # Copy audio from original video to new video
        try:
            (
                ffmpeg
                .output(
                    ffmpeg.input(str(temp_video)).video,
                    ffmpeg.input(input_path).audio,
                    str(output_path),
                    vcodec='copy',
                    acodec='aac',
                    shortest=None
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error:
            # If audio copying fails, use video without audio
            logger.warning("Could not copy audio, using video without audio")
            shutil.copy(temp_video, output_path)
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        # Update status
        processing_status[video_id]['status'] = 'completed'
        processing_status[video_id]['progress'] = 100
        processing_status[video_id]['output_path'] = str(output_path)
        
        logger.info(f"Video processing completed: {output_path}")
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        processing_status[video_id]['status'] = 'failed'
        processing_status[video_id]['message'] = str(e)
        raise

@api_router.get("/")
async def root():
    return {"message": "Watermark Remover API"}

@api_router.post("/upload", response_model=VideoUploadResponse)
async def upload_video(file: UploadFile = File(...)):
    """Upload video file"""
    try:
        # Validate file size (max 110MB)
        content = await file.read()
        file_size = len(content)
        
        if file_size > 110 * 1024 * 1024:  # 110MB
            raise HTTPException(status_code=400, detail="File size exceeds 110MB limit")
        
        # Generate unique video ID
        video_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix or '.mp4'
        filename = f"{video_id}{file_extension}"
        file_path = UPLOAD_DIR / filename
        
        # Save file
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # Get video info
        try:
            info = get_video_info(str(file_path))
        except Exception as e:
            # Clean up invalid file
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="Invalid video file format")
        
        # Validate video duration (max 2 minutes)
        if info['duration'] > 120:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="Video duration exceeds 2 minutes limit")
        
        # Validate resolution (max 1080p)
        if info['height'] > 1080 or info['width'] > 1920:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="Video resolution exceeds 1080p limit")
        
        return VideoUploadResponse(
            video_id=video_id,
            filename=file.filename,
            size=file_size,
            duration=info['duration'],
            width=info['width'],
            height=info['height'],
            fps=info['fps']
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@api_router.post("/process")
async def process_video(request: WatermarkRemovalRequest):
    """Start watermark removal process"""
    try:
        # Find video file
        video_files = list(UPLOAD_DIR.glob(f"{request.video_id}.*"))
        if not video_files:
            raise HTTPException(status_code=404, detail="Video not found")
        
        input_path = video_files[0]
        output_path = PROCESSED_DIR / f"{request.video_id}_processed.mp4"
        
        # Initialize processing status
        processing_status[request.video_id] = {
            'status': 'queued',
            'progress': 0,
            'message': 'Processing started',
            'output_path': None
        }
        
        # Start processing in background
        loop = asyncio.get_event_loop()
        loop.run_in_executor(
            executor,
            process_video_frames,
            str(input_path),
            str(output_path),
            request.x,
            request.y,
            request.width,
            request.height,
            request.video_id
        )
        
        return {"message": "Processing started", "video_id": request.video_id}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@api_router.get("/status/{video_id}", response_model=ProcessingStatus)
async def get_processing_status(video_id: str):
    """Get processing status"""
    if video_id not in processing_status:
        raise HTTPException(status_code=404, detail="Processing status not found")
    
    return ProcessingStatus(
        video_id=video_id,
        **processing_status[video_id]
    )

@api_router.get("/download/{video_id}")
async def download_video(video_id: str):
    """Download processed video"""
    output_path = PROCESSED_DIR / f"{video_id}_processed.mp4"
    
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Processed video not found")
    
    return FileResponse(
        path=str(output_path),
        media_type='video/mp4',
        filename=f"watermark_removed_{video_id}.mp4"
    )

@api_router.get("/video/{video_id}")
async def get_uploaded_video(video_id: str):
    """Get uploaded video for preview"""
    video_files = list(UPLOAD_DIR.glob(f"{video_id}.*"))
    if not video_files:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return FileResponse(
        path=str(video_files[0]),
        media_type='video/mp4'
    )

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
    executor.shutdown(wait=True)