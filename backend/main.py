from utils.apple_silicon import enable_apple_silicon_optimizations
enable_apple_silicon_optimizations()

import uvicorn
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from api.routers import movement, feedback
from core.config import settings
from utils.video.processor import VideoProcessor
from utils.logger import setup_logging

# Setup logging
logger = setup_logging()
logger.info("Starting Muscle Memory API")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="AI-Enhanced Fitness Analysis Service",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
VideoProcessor.create_output_dirs()

# Include routers
app.include_router(movement.router, prefix=settings.API_V1_STR, tags=["Movement Analysis"])
app.include_router(feedback.router, prefix=settings.API_V1_STR, tags=["AI Feedback"])

# Mount static folders for output files
os.makedirs("output", exist_ok=True)
app.mount("/output", StaticFiles(directory="output"), name="output")

@app.get("/")
def read_root():
    return {"message": "Welcome to Muscle Memory API", "version": "0.1.0"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)