import pytest
import numpy as np
import cv2
from unittest.mock import MagicMock, patch
from models.movement.visualization.skeleton_renderer import SkeletonRenderer, VisualizationMode, VisualizationConfig
import mediapipe as mp

class TestSkeletonRenderer:
    
    def setup_method(self):
        """Set up test instance."""
        self.config = VisualizationConfig(
            mode=VisualizationMode.ANALYSIS,
            exercise_type='squat',
            camera_angle='front',
            show_reference_lines=True,
            show_angles=True,
            show_motion_trails=True
        )
        self.renderer = SkeletonRenderer(self.config)
        
        # Create mock landmarks
        self.mock_landmarks = MagicMock()
        self.mock_landmarks.__getitem__.return_value = MagicMock(x=0.5, y=0.5, z=0.5, visibility=1.0)
        
        # Create mock image
        self.mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    def test_init(self):
        """Test initialization."""
        assert self.renderer.config == self.config
        assert self.renderer.mp_pose is not None
        assert self.renderer.mp_drawing is not None
        assert 'squat' in self.renderer.exercise_configs
        assert 'lunge' in self.renderer.exercise_configs
        assert 'pushup' in self.renderer.exercise_configs
    
    def test_generate_virtual_landmarks(self):
        """Test virtual landmark generation."""
        virtual_landmarks = self.renderer._generate_virtual_landmarks(self.mock_landmarks)
        
        assert 'CERVICAL' in virtual_landmarks
        assert 'THORACIC' in virtual_landmarks
        assert 'PLUMB_LINE' in virtual_landmarks
        
        # Check coordinates
        assert all(0 <= coord <= 1 for coord in virtual_landmarks['CERVICAL'])
        assert all(0 <= coord <= 1 for coord in virtual_landmarks['THORACIC'])
        assert all(0 <= coord <= 1 for coord in virtual_landmarks['PLUMB_LINE'])
    
    def test_draw_reference_lines(self):
        """Test reference line drawing."""
        virtual_landmarks = self.renderer._generate_virtual_landmarks(self.mock_landmarks)
        
        # Create a copy of the image to draw on
        image = self.mock_image.copy()
        
        # Draw reference lines
        self.renderer._draw_reference_lines(image, self.mock_landmarks, virtual_landmarks)
        
        # Check if image was modified
        assert not np.array_equal(image, self.mock_image)
    
    def test_draw_angles(self):
        """Test angle drawing."""
        angles = {
            'LEFT_KNEE': 90,
            'RIGHT_KNEE': 95,
            'LEFT_HIP': 100,
            'RIGHT_HIP': 105
        }
        
        # Create a copy of the image to draw on
        image = self.mock_image.copy()
        
        # Draw angles
        self.renderer._draw_angles(image, self.mock_landmarks, angles)
        
        # Check if image was modified
        assert not np.array_equal(image, self.mock_image)
    
    def test_draw_motion_trails(self):
        """Test motion trail drawing."""
        # Create a copy of the image to draw on
        image = self.mock_image.copy()
        
        # Draw motion trails
        self.renderer._draw_motion_trails(image, self.mock_landmarks)
        
        # Check if image was modified
        assert not np.array_equal(image, self.mock_image)
        
        # Check if previous landmarks were stored
        assert self.renderer.previous_landmarks == self.mock_landmarks
    
    def test_draw_enhanced_skeleton(self):
        """Test enhanced skeleton drawing."""
        virtual_landmarks = self.renderer._generate_virtual_landmarks(self.mock_landmarks)
        
        # Create a copy of the image to draw on
        image = self.mock_image.copy()
        
        # Draw enhanced skeleton
        self.renderer._draw_enhanced_skeleton(image, self.mock_landmarks, virtual_landmarks)
        
        # Check if image was modified
        assert not np.array_equal(image, self.mock_image)
    
    def test_render(self):
        """Test complete rendering."""
        angles = {
            'LEFT_KNEE': 90,
            'RIGHT_KNEE': 95,
            'LEFT_HIP': 100,
            'RIGHT_HIP': 105
        }
        
        # Render complete visualization
        result = self.renderer.render(self.mock_image, self.mock_landmarks, angles)
        
        # Check if result is a valid image
        assert isinstance(result, np.ndarray)
        assert result.shape == self.mock_image.shape
        assert not np.array_equal(result, self.mock_image)
    
    def test_render_realtime_mode(self):
        """Test rendering in real-time mode."""
        # Create renderer with real-time mode
        config = VisualizationConfig(mode=VisualizationMode.REALTIME)
        renderer = SkeletonRenderer(config)
        
        angles = {
            'LEFT_KNEE': 90,
            'RIGHT_KNEE': 95
        }
        
        # Render in real-time mode
        result = renderer.render(self.mock_image, self.mock_landmarks, angles)
        
        # Check if result is a valid image
        assert isinstance(result, np.ndarray)
        assert result.shape == self.mock_image.shape
        assert not np.array_equal(result, self.mock_image)
    
    def test_render_no_landmarks(self):
        """Test rendering with no landmarks."""
        result = self.renderer.render(self.mock_image, None)
        
        # Should return original image
        assert np.array_equal(result, self.mock_image)
    
    def test_render_no_angles(self):
        """Test rendering with no angles."""
        result = self.renderer.render(self.mock_image, self.mock_landmarks)
        
        # Should still modify image
        assert isinstance(result, np.ndarray)
        assert result.shape == self.mock_image.shape
        assert not np.array_equal(result, self.mock_image)
    
    def test_exercise_specific_config(self):
        """Test exercise-specific configuration."""
        # Test squat configuration
        self.renderer.config.exercise_type = 'squat'
        config = self.renderer._get_exercise_config()
        assert 'key_joints' in config
        assert 'reference_lines' in config
        assert 'angle_thresholds' in config
        
        # Test lunge configuration
        self.renderer.config.exercise_type = 'lunge'
        config = self.renderer._get_exercise_config()
        assert 'key_joints' in config
        assert 'reference_lines' in config
        assert 'angle_thresholds' in config
        
        # Test pushup configuration
        self.renderer.config.exercise_type = 'pushup'
        config = self.renderer._get_exercise_config()
        assert 'key_joints' in config
        assert 'reference_lines' in config
        assert 'angle_thresholds' in config
        
        # Test unknown exercise
        self.renderer.config.exercise_type = 'unknown'
        config = self.renderer._get_exercise_config()
        assert config == {} 