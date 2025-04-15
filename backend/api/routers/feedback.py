from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any, Optional

from models.ai_agent.feedback_generator import FeedbackGenerator

class FeedbackRequest(BaseModel):
    movement_data: Dict[str, Any]
    user_profile: Optional[Dict[str, Any]] = None

router = APIRouter()

@router.post("/feedback")
async def generate_feedback(request: FeedbackRequest):
    """
    Generate AI coaching feedback based on movement data.
    """
    try:
        feedback_generator = FeedbackGenerator()
        feedback = feedback_generator.generate_feedback(request.movement_data)
        
        # Add user-specific adjustments if profile is provided
        if request.user_profile:
            feedback = _adjust_for_user_profile(feedback, request.user_profile)
        
        return feedback
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _adjust_for_user_profile(feedback: Dict[str, Any], user_profile: Dict[str, Any]) -> Dict[str, Any]:
    """Adjust feedback based on user profile information."""
    # This is a placeholder for future personalization
    
    # Example: Adjust for experience level
    experience_level = user_profile.get("experience_level", "intermediate")
    
    if experience_level == "beginner":
        # Simplify feedback for beginners
        feedback["recommendations"] = feedback["recommendations"][:3]  # Limit to top 3 recommendations
        feedback["form_cues"] = feedback["form_cues"][:3]  # Limit to top 3 cues
    elif experience_level == "advanced":
        # Add more technical details for advanced users
        feedback["technical_details"] = True
    
    # Example: Adjust for goals
    goals = user_profile.get("goals", [])
    
    if "mobility" in goals:
        feedback["recommendations"].insert(0, "Focus on increasing mobility in your hip and ankle joints.")
    elif "strength" in goals:
        feedback["recommendations"].insert(0, "Consider adding resistance to this movement to increase strength gains.")
    
    return feedback