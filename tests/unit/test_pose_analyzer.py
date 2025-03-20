import pytest
import numpy as np
import os
import cv2
from models.movement.mediapipe.pose_analyzer import PoseAnalyzer

class TestPoseAnalyzer:
    
    def setup_method(self):
        """Set up test instance."""
        self.analyzer = PoseAnalyzer(model_complexity=0)  # Use lightweight model for tests
    
    def test_init(self):
        """Test initialization of PoseAnalyzer."""
        assert self.analyzer.mp_pose is not None
        assert self.analyzer.pose is not None
        assert isinstance(self.analyzer.angles_data, list)
        assert len(self.analyzer.detected_joints) == 0
    
    def test_calculate_angle(self):
        """Test angle calculation."""
        # Test with 90 degree angle
        a = [0, 0, 0]
        b = [0, 0, 0]
        c = [1, 0, 0]
        angle = self.analyzer.calculate_angle(a, b, c)
        assert np.isclose(angle, 90.0)
        
        # Test with 180 degree angle
        a = [-1, 0, 0]
        b = [0, 0, 0]
        c = [1, 0, 0]
        angle = self.analyzer.calculate_angle(a, b, c)
        assert np.isclose(angle, 180.0)
    
    def test_get_landmark_coords(self):
        """Test extracting coordinates from a landmark."""
        # Create a mock landmark
        class MockLandmark:
            def __init__(self):
                self.x = 0.5
                self.y = 0.6
                self.z = 0.7
        
        landmark = MockLandmark()
        coords = self.analyzer.get_landmark_coords(landmark)
        assert coords == [0.5, 0.6, 0.7]
    
    def test_calculate_angles_empty(self):
        """Test calculating angles with empty results."""
        class MockResults:
            def __init__(self):
                self.pose_landmarks = None
                self.pose_world_landmarks = None
        
        results = MockResults()
        angles = self.analyzer.calculate_angles(results, ["LEFT_KNEE"])
        assert angles == {}