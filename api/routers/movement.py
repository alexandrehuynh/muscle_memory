from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse
import os
import tempfile
import shutil
from typing import List, Optional
import uuid
from datetime import datetime

from models.movement import MovementPipeline, PipelineConfig
from models.movement.types import AnalysisType
from utils.video.processor import VideoProcessor
from utils.serialization import CustomJSONResponse, sanitize_for_json
from fastapi.responses import JSONResponse

router = APIRouter()

# Create a storage directory for uploaded files
UPLOAD_DIR = "temp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/analyze")
async def analyze_movement(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    model_complexity: int = Query(1, ge=0, le=2, description="MediaPipe model complexity (0-2)"),
    save_annotated_video: bool = Query(False, description="Whether to save the annotated video"),
    selected_joints: List[str] = Query(None, description="Specific joints to analyze"),
    analysis_type: Optional[AnalysisType] = Query(None, description="Type of analysis to perform")
):
    """
    Analyze movement from uploaded video file.
    Returns joint angles, movement metrics, and detected exercise type.
    """
    try:
        # Generate unique ID for this analysis
        analysis_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save uploaded file
        file_extension = os.path.splitext(video.filename)[1]
        temp_file_path = os.path.join(UPLOAD_DIR, f"{analysis_id}{file_extension}")
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # Define output path if annotated video is requested
        output_path = None
        if save_annotated_video:
            output_dir = os.path.join("output", "annotated_videos")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"analyzed_{timestamp}_{analysis_id}{file_extension}")
        
        # Create pipeline configuration
        analysis_type_str = analysis_type.value if analysis_type else "detailed"
        pipeline_config = PipelineConfig(
            model_complexity=model_complexity,
            analysis_type=analysis_type_str,
            process_every_n_frames=1 if analysis_type_str == "detailed" else 2
        )
        
        # Create analyzer and process video
        pipeline = MovementPipeline(pipeline_config)
        success, message = pipeline.process_video(temp_file_path, output_path, selected_joints)
        
        if not success:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            raise HTTPException(status_code=400, detail=message)
        
        # Analyze movement data
        analysis_success, analysis_results, analysis_message = pipeline.analyze_movement(selected_joints)
        
        if not analysis_success:
            raise HTTPException(status_code=400, detail=analysis_message)
        
        # Background task to clean up temp file
        async def cleanup_temp_file(temp_file_path):
            """Clean up temp file."""
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        
        background_tasks.add_task(cleanup_temp_file, temp_file_path)
        
        # Prepare response
        response = {
            "message": message,
            "analysis_id": analysis_id,
            "exercise_type": analysis_results.get("exercise_type", "unknown"),
            "metrics": sanitize_for_json(analysis_results.get("metrics", {})),
            "statistics": sanitize_for_json(analysis_results.get("statistics", {})),
            "annotated_video": output_path if save_annotated_video else None
        }

        return CustomJSONResponse(content=response)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/exercises")
async def get_supported_exercises():
    """Get list of supported exercise types for analysis."""
    return {
        "supported_exercises": [
            {"id": "squat", "name": "Squat", "description": "Analysis of squat movement patterns"},
            {"id": "lunge", "name": "Lunge", "description": "Analysis of lunge movement patterns"},
            {"id": "pushup", "name": "Push-up", "description": "Analysis of push-up movement patterns"}
        ]
    }

@router.get("/joints")
async def get_available_joints():
    """Get list of joint angles that can be analyzed."""
    joint_calculator = MovementPipeline().joint_calculator
    joints = joint_calculator.get_available_joints()
    
    return {
        "available_joints": joints
    }