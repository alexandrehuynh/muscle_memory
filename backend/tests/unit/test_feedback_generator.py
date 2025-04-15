import pytest
from models.ai_agent.feedback_generator import FeedbackGenerator

class TestFeedbackGenerator:
    
    def setup_method(self):
        """Set up test instance."""
        self.generator = FeedbackGenerator()
    
    def test_init(self):
        """Test initialization."""
        assert self.generator.exercise_templates is not None
        assert "squat" in self.generator.exercise_templates
        assert "lunge" in self.generator.exercise_templates
        assert "pushup" in self.generator.exercise_templates
    
    def test_generate_symmetry_feedback(self):
        """Test symmetry feedback generation."""
        high_feedback = self.generator._generate_symmetry_feedback(95)
        assert "well balanced" in high_feedback
        
        med_feedback = self.generator._generate_symmetry_feedback(80)
        assert "good balance" in med_feedback
        
        low_feedback = self.generator._generate_symmetry_feedback(50)
        assert "significant" in low_feedback
    
    def test_generate_feedback_squat(self):
        """Test feedback generation for squat."""
        # Create mock analysis results for a squat
        analysis_results = {
            "exercise_type": "squat",
            "metrics": {
                "symmetry_score": 85,
                "smoothness_score": 90,
                "range_of_motion_score": 75,
                "depth_score": "Deep",
                "knee_safety": "Good knee angle maintained"
            },
            "statistics": {}
        }
        
        feedback = self.generator.generate_feedback(analysis_results)
        
        assert "feedback" in feedback
        assert "recommendations" in feedback
        assert "form_cues" in feedback
        assert "scores" in feedback
        assert "Great depth" in feedback["feedback"]
        assert len(feedback["recommendations"]) > 0
        assert len(feedback["form_cues"]) > 0