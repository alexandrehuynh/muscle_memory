import pytest
import os
import shutil
import tempfile
from models.movement.analyzer import MovementAnalyzer
from models.ai_agent.feedback_generator import FeedbackGenerator

# Skip tests if no test data is available
pytestmark = pytest.mark.skipif(
    not os.path.exists("test_data/videos/squat_test.mp4"),
    reason="Test video data not available"
)

class TestMovementPipeline:
    
    @classmethod
    def setup_class(cls):
        """Set up resources for all tests."""
        # Create temporary directory for outputs
        cls.temp_dir = tempfile.mkdtemp()
    
    @classmethod
    def teardown_class(cls):
        """Clean up after all tests."""
        # Remove temporary directory
        shutil.rmtree(cls.temp_dir)
    
    def test_full_pipeline(self):
        """Test the full movement analysis and feedback pipeline."""
        # Skip test if no test video is available
        test_video = "test_data/videos/squat_test.mp4"
        if not os.path.exists(test_video):
            pytest.skip("Test video not available")
        
        # 1. Process video
        analyzer = MovementAnalyzer(model_complexity=0)
        output_path = os.path.join(self.temp_dir, "output_video.mp4")
        success, message = analyzer.process_video(test_video, output_path)
        
        assert success, f"Video processing failed: {message}"
        
        # 2. Analyze movement
        analysis_success, analysis_results, analysis_message = analyzer.analyze_movement()
        
        assert analysis_success, f"Movement analysis failed: {analysis_message}"
        assert "exercise_type" in analysis_results
        assert "metrics" in analysis_results
        assert "statistics" in analysis_results
        
        # 3. Generate feedback
        feedback_generator = FeedbackGenerator()
        feedback = feedback_generator.generate_feedback(analysis_results)
        
        assert "feedback" in feedback
        assert "recommendations" in feedback
        assert "form_cues" in feedback
        assert "scores" in feedback