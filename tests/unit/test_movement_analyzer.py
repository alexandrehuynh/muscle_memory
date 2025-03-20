import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from models.movement.analyzer import MovementAnalyzer

class TestMovementAnalyzer:
    
    def setup_method(self):
        """Set up test instance."""
        self.analyzer = MovementAnalyzer(model_complexity=0)
    
    def test_init(self):
        """Test initialization."""
        assert self.analyzer.pose_analyzer is not None
        assert 'squat' in self.analyzer.exercise_types
        assert 'lunge' in self.analyzer.exercise_types
        assert 'pushup' in self.analyzer.exercise_types
    
    @patch('models.movement.analyzer.PoseAnalyzer')
    def test_process_video_success(self, mock_pose_analyzer):
        """Test successful video processing."""
        # Setup mock
        mock_pose_instance = MagicMock()
        mock_pose_analyzer.return_value = mock_pose_instance
        mock_pose_instance.process_video.return_value = (True, "Success message")
        mock_pose_instance.detected_joints = {"LEFT_KNEE", "RIGHT_KNEE"}
        
        # Create analyzer with mock
        analyzer = MovementAnalyzer(model_complexity=0)
        
        # Call method
        success, message = analyzer.process_video("test_video.mp4")
        
        # Assertions
        assert success is True
        assert "Success" in message
        mock_pose_instance.process_video.assert_called_once()
    
    @patch('models.movement.analyzer.PoseAnalyzer')
    def test_process_video_failure(self, mock_pose_analyzer):
        """Test failed video processing."""
        # Setup mock
        mock_pose_instance = MagicMock()
        mock_pose_analyzer.return_value = mock_pose_instance
        mock_pose_instance.process_video.return_value = (False, "Error message")
        mock_pose_instance.detected_joints = set()
        
        # Create analyzer with mock
        analyzer = MovementAnalyzer(model_complexity=0)
        
        # Call method
        success, message = analyzer.process_video("test_video.mp4")
        
        # Assertions
        assert success is False
        assert "Error" in message
    
    def test_detect_squat(self):
        """Test squat detection."""
        # Create DataFrame that looks like a squat (knee angles go down and up)
        knee_angles = pd.Series([170, 160, 140, 100, 90, 100, 140, 160, 170])
        df = pd.DataFrame({"LEFT_KNEE": knee_angles})
        
        # Test
        result = self.analyzer._detect_squat(df)
        assert result is True
        
        # Test non-squat pattern
        knee_angles = pd.Series([170, 165, 160, 165, 170])
        df = pd.DataFrame({"LEFT_KNEE": knee_angles})
        result = self.analyzer._detect_squat(df)
        assert result is False