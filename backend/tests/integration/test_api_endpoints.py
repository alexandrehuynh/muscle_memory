import pytest
from fastapi.testclient import TestClient
import os
from main import app

client = TestClient(app)

class TestAPIEndpoints:
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_get_supported_exercises(self):
        """Test getting supported exercises."""
        response = client.get("/api/v1/exercises")
        assert response.status_code == 200
        assert "supported_exercises" in response.json()
        exercises = response.json()["supported_exercises"]
        assert len(exercises) > 0
        
        # Check structure of first exercise
        first_exercise = exercises[0]
        assert "id" in first_exercise
        assert "name" in first_exercise
        assert "description" in first_exercise
    
    def test_get_available_joints(self):
        """Test getting available joints."""
        response = client.get("/api/v1/joints")
        assert response.status_code == 200
        assert "available_joints" in response.json()
        joints = response.json()["available_joints"]
        assert len(joints) > 0
        assert "LEFT_KNEE" in joints
        assert "RIGHT_KNEE" in joints
    
    def test_analyze_video(self):
        """Test video analysis endpoint."""
        test_video = "test_data/videos/squat_test.mp4"
        
        # Skip test if test video not available
        if not os.path.exists(test_video):
            pytest.skip("Test video not available")
        
        with open(test_video, "rb") as f:
            files = {"video": (os.path.basename(test_video), f, "video/mp4")}
            response = client.post(
                "/api/v1/analyze",
                files=files,
                params={"model_complexity": 0, "save_annotated_video": True}
            )
        
        assert response.status_code == 200
        assert "analysis_id" in response.json()
        assert "exercise_type" in response.json()
    
    def test_generate_feedback(self):
        """Test feedback generation endpoint."""
        # Mock analysis results
        analysis_results = {
            "exercise_type": "squat",
            "metrics": {
                "symmetry_score": 85,
                "smoothness_score": 90,
                "range_of_motion_score": 75,
                "depth_score": "Deep",
                "knee_safety": "Good knee angle maintained"
            }
        }
        
        # Create request data
        request_data = {
            "movement_data": analysis_results,
            "user_profile": {
                "experience_level": "intermediate",
                "goals": ["strength", "mobility"]
            }
        }
        
        response = client.post("/api/v1/feedback", json=request_data)
        
        assert response.status_code == 200
        assert "feedback" in response.json()
        assert "recommendations" in response.json()
        assert "form_cues" in response.json()