from pydantic_settings import BaseSettings
from typing import Optional, List
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """
    Application settings.
    """
    # API Config
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Muscle Memory"
    
    # MediaPipe Config
    MEDIAPIPE_MODEL_COMPLEXITY: int = 2  # 0, 1, or 2 (higher = more accurate but slower)
    
    # Processing Config
    TEMP_VIDEO_DIR: str = "temp/videos"
    TEMP_UPLOAD_DIR: str = "temp/uploads"
    OUTPUT_DIR: str = "output"
    ANNOTATED_VIDEO_DIR: str = "output/annotated_videos"
    
    # Video Analysis Config
    MAX_VIDEO_SIZE_MB: int = 50
    SUPPORTED_VIDEO_FORMATS: List[str] = ["mp4", "mov", "avi"]
    
    # AI Agent Config
    AI_MODEL_PATH: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create global settings object
settings = Settings()

# Ensure temporary directories exist
os.makedirs(settings.TEMP_VIDEO_DIR, exist_ok=True)
os.makedirs(settings.TEMP_UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
os.makedirs(settings.ANNOTATED_VIDEO_DIR, exist_ok=True)