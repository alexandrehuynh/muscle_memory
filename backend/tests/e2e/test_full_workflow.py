import pytest
import os
import requests
import time
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# Skip tests if no test data is available
pytestmark = pytest.mark.skipif(
    not os.path.exists("test_data/videos/squat_test.mp4"),
    reason="Test video data not available"
)

class TestFullWorkflow:
    
    def test_analyze_and_feedback(self):
        """Test the complete workflow from upload to feedback."""
        test_video = "test_data/videos/squat_test.mp4"
        
        # Skip test if test video not available
        if not os.path.exists(test_video):
            pytest.skip("Test video not available")
        
        # Step 1: Upload video for analysis
        with open(test_video, "rb") as f:
            files = {"video": (os.path.basename(test_video), f, "video/mp4")}
            response = client.post(
                "/api/v1/analyze",
                files=files,
                params={"model_complexity": 0, "save_annotated_video": True}
            )
        
        assert response.status_code == 200
        analysis_result = response.json()
        assert "analysis_id" in analysis_result
        
        # Step 2: Generate feedback based on analysis results
        feedback_request = {
            "movement_data": {
                "exercise_type": analysis_result["exercise_type"],
                "metrics": analysis_result["metrics"],
                "statistics": analysis_result.get("statistics", {})
            },
            "user_profile": {
                "experience_level": "intermediate",
                "goals": ["strength", "mobility"]
            }
        }
        
        feedback_response = client.post("/api/v1/feedback", json=feedback_request)
        
        assert feedback_response.status_code == 200
        feedback_result = feedback_response.json()
        assert "feedback" in feedback_result
        assert "recommendations" in feedback_result
        assert "form_cues" in feedback_result
        
        # Step 3: Verify output files were created
        if "annotated_video" in analysis_result and analysis_result["annotated_video"]:
            # Check if file exists (might need to adjust path if it's relative)
            video_path = analysis_result["annotated_video"]
            assert os.path.exists(video_path), f"Annotated video not found at {video_path}"