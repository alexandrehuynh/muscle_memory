import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import math

class VisualizationMode(Enum):
    """Enum for different visualization modes."""
    SIMPLE = "simple"    # Basic skeleton visualization
    DETAILED = "detailed"  # More detailed visualization with angles and references

@dataclass
class VisualizationConfig:
    """Configuration for skeleton visualization."""
    mode: VisualizationMode = VisualizationMode.SIMPLE
    show_angles: bool = True
    show_joint_names: bool = False
    confidence_threshold: float = 0.5
    line_thickness: int = 2
    joint_radius: int = 5
    smoothing_factor: float = 0.3
    show_reference_lines: bool = False
    left_color: Tuple[int, int, int] = (0, 0, 255)   # Red for left side
    right_color: Tuple[int, int, int] = (255, 0, 0)  # Blue for right side
    center_color: Tuple[int, int, int] = (0, 255, 0) # Green for central points

class SkeletonVisualizer:
    """
    SkeletonVisualizer handles ONLY the visualization of skeletal landmarks.
    
    Responsibilities:
    - Drawing the skeleton based on detected landmarks
    - Visualizing joint angles
    - Applying visual smoothing for stability
    - Adding reference lines and annotations
    
    This class does NOT handle:
    - Pose detection
    - Joint angle calculations
    - Video processing
    - Exercise classification
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize the skeleton visualizer.
        
        Args:
            config: Configuration for visualization
        """
        self.config = config or VisualizationConfig()
        
        # MediaPipe pose solution for constants
        self.mp_pose = mp.solutions.pose
        
        # Define the connections between landmarks for drawing
        self.connections = [
            # Torso
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER),
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_HIP),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_HIP),
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP),
            
            # Left arm
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_ELBOW),
            (self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.LEFT_WRIST),
            (self.mp_pose.PoseLandmark.LEFT_WRIST, self.mp_pose.PoseLandmark.LEFT_PINKY),
            (self.mp_pose.PoseLandmark.LEFT_WRIST, self.mp_pose.PoseLandmark.LEFT_INDEX),
            (self.mp_pose.PoseLandmark.LEFT_PINKY, self.mp_pose.PoseLandmark.LEFT_INDEX),
            
            # Right arm
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_ELBOW),
            (self.mp_pose.PoseLandmark.RIGHT_ELBOW, self.mp_pose.PoseLandmark.RIGHT_WRIST),
            (self.mp_pose.PoseLandmark.RIGHT_WRIST, self.mp_pose.PoseLandmark.RIGHT_PINKY),
            (self.mp_pose.PoseLandmark.RIGHT_WRIST, self.mp_pose.PoseLandmark.RIGHT_INDEX),
            (self.mp_pose.PoseLandmark.RIGHT_PINKY, self.mp_pose.PoseLandmark.RIGHT_INDEX),
            
            # Left leg
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_KNEE),
            (self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.LEFT_ANKLE),
            (self.mp_pose.PoseLandmark.LEFT_ANKLE, self.mp_pose.PoseLandmark.LEFT_HEEL),
            (self.mp_pose.PoseLandmark.LEFT_HEEL, self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
            
            # Right leg
            (self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_KNEE),
            (self.mp_pose.PoseLandmark.RIGHT_KNEE, self.mp_pose.PoseLandmark.RIGHT_ANKLE),
            (self.mp_pose.PoseLandmark.RIGHT_ANKLE, self.mp_pose.PoseLandmark.RIGHT_HEEL),
            (self.mp_pose.PoseLandmark.RIGHT_HEEL, self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX),
            
            # Face
            (self.mp_pose.PoseLandmark.NOSE, self.mp_pose.PoseLandmark.LEFT_EYE_INNER),
            (self.mp_pose.PoseLandmark.LEFT_EYE_INNER, self.mp_pose.PoseLandmark.LEFT_EYE),
            (self.mp_pose.PoseLandmark.LEFT_EYE, self.mp_pose.PoseLandmark.LEFT_EYE_OUTER),
            (self.mp_pose.PoseLandmark.NOSE, self.mp_pose.PoseLandmark.RIGHT_EYE_INNER),
            (self.mp_pose.PoseLandmark.RIGHT_EYE_INNER, self.mp_pose.PoseLandmark.RIGHT_EYE),
            (self.mp_pose.PoseLandmark.RIGHT_EYE, self.mp_pose.PoseLandmark.RIGHT_EYE_OUTER),
            (self.mp_pose.PoseLandmark.LEFT_EYE_OUTER, self.mp_pose.PoseLandmark.LEFT_EAR),
            (self.mp_pose.PoseLandmark.RIGHT_EYE_OUTER, self.mp_pose.PoseLandmark.RIGHT_EAR),
            (self.mp_pose.PoseLandmark.NOSE, self.mp_pose.PoseLandmark.MOUTH_LEFT),
            (self.mp_pose.PoseLandmark.MOUTH_LEFT, self.mp_pose.PoseLandmark.MOUTH_RIGHT),
            (self.mp_pose.PoseLandmark.NOSE, self.mp_pose.PoseLandmark.MOUTH_RIGHT)
        ]
        
        # Previous landmarks for smoothing
        self.previous_landmarks = None
    
    def _get_landmark_pixel_coordinates(self, landmark, image_width, image_height) -> Tuple[int, int]:
        """
        Convert normalized landmark coordinates to pixel coordinates.
        
        Args:
            landmark: MediaPipe landmark with normalized coordinates
            image_width: Width of the image
            image_height: Height of the image
            
        Returns:
            Tuple of (x, y) pixel coordinates
        """
        x = min(int(landmark.x * image_width), image_width - 1)
        y = min(int(landmark.y * image_height), image_height - 1)
        return (x, y)
    
    def _get_connection_color(self, start_landmark, end_landmark) -> Tuple[int, int, int]:
        """
        Determine color based on whether the connection is left, right, or central.
        
        Args:
            start_landmark: Starting landmark enum
            end_landmark: Ending landmark enum
            
        Returns:
            BGR color tuple
        """
        # Left side landmarks typically have 'LEFT' in their name
        if ('LEFT' in start_landmark.name and 'LEFT' in end_landmark.name):
            return self.config.left_color
        # Right side landmarks typically have 'RIGHT' in their name
        elif ('RIGHT' in start_landmark.name and 'RIGHT' in end_landmark.name):
            return self.config.right_color
        # Central connections (between left and right or central landmarks)
        else:
            return self.config.center_color
    
    def _smooth_landmarks(self, current_landmarks: Dict, previous_landmarks: Optional[Dict]) -> Dict:
        """
        Apply temporal smoothing to landmarks for visual stability.
        
        Args:
            current_landmarks: Current frame landmarks
            previous_landmarks: Previous frame landmarks
            
        Returns:
            Smoothed landmarks
        """
        if previous_landmarks is None or self.config.smoothing_factor <= 0:
            return current_landmarks
            
        smoothed_landmarks = {}
        
        # Get common indices between current and previous landmarks
        common_indices = set(current_landmarks.keys()).intersection(set(previous_landmarks.keys()))
        
        for idx in common_indices:
            # Skip landmarks with low confidence
            current_confidence = getattr(current_landmarks[idx], 'visibility', 0.0)
            if current_confidence < self.config.confidence_threshold:
                smoothed_landmarks[idx] = current_landmarks[idx]
                continue
                
            # Create smoothed landmark
            # Use the same class as the current landmark
            landmark_cls = type(current_landmarks[idx])
            smoothed = landmark_cls()
            
            # Apply weighted average
            alpha = self.config.smoothing_factor
            previous = previous_landmarks[idx]
            current = current_landmarks[idx]
            
            # Set coordinates with smoothing
            smoothed.x = alpha * previous.x + (1 - alpha) * current.x
            smoothed.y = alpha * previous.y + (1 - alpha) * current.y
            smoothed.z = alpha * previous.z + (1 - alpha) * current.z
            
            # Keep current visibility
            smoothed.visibility = current.visibility
            
            smoothed_landmarks[idx] = smoothed
        
        # Add landmarks that are only in current
        for idx in current_landmarks:
            if idx not in common_indices:
                smoothed_landmarks[idx] = current_landmarks[idx]
                
        return smoothed_landmarks
    
    def _draw_reference_lines(self, image: np.ndarray, landmarks: Dict) -> np.ndarray:
        """
        Draw reference lines for better movement visualization.
        
        Args:
            image: Image to draw on
            landmarks: Dictionary of landmarks
            
        Returns:
            Image with reference lines
        """
        if not self.config.show_reference_lines or landmarks is None:
            return image
            
        height, width, _ = image.shape
        
        # Draw vertical reference line through shoulders midpoint
        left_shoulder_idx = self.mp_pose.PoseLandmark.LEFT_SHOULDER.value
        right_shoulder_idx = self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        
        if left_shoulder_idx in landmarks and right_shoulder_idx in landmarks:
            left_shoulder = landmarks[left_shoulder_idx]
            right_shoulder = landmarks[right_shoulder_idx]
            
            # Calculate midpoint
            mid_x = int((left_shoulder.x + right_shoulder.x) * width / 2)
            
            # Draw vertical line from top to bottom
            cv2.line(image, (mid_x, 0), (mid_x, height), (200, 200, 200), 1, cv2.LINE_AA)
        
        return image
    
    def _draw_angles(self, image: np.ndarray, landmarks: Dict, angles: Dict) -> np.ndarray:
        """
        Draw joint angles on the image.
        
        Args:
            image: Image to draw on
            landmarks: Dictionary of landmarks
            angles: Dictionary of joint angles and confidence values
            
        Returns:
            Image with angles drawn
        """
        if not self.config.show_angles or not angles:
            return image
            
        height, width, _ = image.shape
        
        # Define joint mappings for visualization
        joint_landmark_mappings = {
            'LEFT_SHOULDER': self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            'RIGHT_SHOULDER': self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            'LEFT_ELBOW': self.mp_pose.PoseLandmark.LEFT_ELBOW.value,
            'RIGHT_ELBOW': self.mp_pose.PoseLandmark.RIGHT_ELBOW.value,
            'LEFT_WRIST': self.mp_pose.PoseLandmark.LEFT_WRIST.value,
            'RIGHT_WRIST': self.mp_pose.PoseLandmark.RIGHT_WRIST.value,
            'LEFT_HIP': self.mp_pose.PoseLandmark.LEFT_HIP.value,
            'RIGHT_HIP': self.mp_pose.PoseLandmark.RIGHT_HIP.value,
            'LEFT_KNEE': self.mp_pose.PoseLandmark.LEFT_KNEE.value,
            'RIGHT_KNEE': self.mp_pose.PoseLandmark.RIGHT_KNEE.value,
            'LEFT_ANKLE': self.mp_pose.PoseLandmark.LEFT_ANKLE.value,
            'RIGHT_ANKLE': self.mp_pose.PoseLandmark.RIGHT_ANKLE.value
        }
        
        # Draw angles for joints
        for joint_name, angle_data in angles.items():
            if joint_name not in joint_landmark_mappings:
                continue
                
            landmark_idx = joint_landmark_mappings[joint_name]
            if landmark_idx not in landmarks:
                continue
                
            angle = angle_data.get('angle', 0)
            confidence = angle_data.get('confidence', 0)
            
            # Skip joints with low confidence
            if confidence < self.config.confidence_threshold:
                continue
                
            # Get landmark coordinates
            landmark = landmarks[landmark_idx]
            x, y = self._get_landmark_pixel_coordinates(landmark, width, height)
            
            # Draw angle text
            text = f"{int(angle)}°"
            cv2.putText(image, text, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, text, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        return image
    
    def visualize(self, image: np.ndarray, landmarks: Dict, angles: Optional[Dict] = None) -> np.ndarray:
        """
        Render the skeleton on the image.
        
        Args:
            image: Input image
            landmarks: Dictionary of landmarks indexed by landmark enum values
            angles: Optional dictionary of joint angles
            
        Returns:
            Image with visualized skeleton
        """
        if landmarks is None or not landmarks:
            return image
            
        # Apply smoothing if previous landmarks exist
        if self.previous_landmarks is not None:
            landmarks = self._smooth_landmarks(landmarks, self.previous_landmarks)
        
        # Store current landmarks for next frame
        self.previous_landmarks = landmarks
        
        # Get image dimensions
        height, width, _ = image.shape
        
        # Draw reference lines if enabled
        if self.config.show_reference_lines:
            image = self._draw_reference_lines(image, landmarks)
        
        # Draw connections between landmarks
        for connection in self.connections:
            start_idx = connection[0].value
            end_idx = connection[1].value
            
            if start_idx not in landmarks or end_idx not in landmarks:
                continue
                
            # Get start and end landmarks
            start_landmark = landmarks[start_idx]
            end_landmark = landmarks[end_idx]
            
            # Check confidence
            start_confidence = getattr(start_landmark, 'visibility', 0.0)
            end_confidence = getattr(end_landmark, 'visibility', 0.0)
            if (start_confidence < self.config.confidence_threshold or 
                end_confidence < self.config.confidence_threshold):
                continue
            
            # Get pixel coordinates
            start_point = self._get_landmark_pixel_coordinates(start_landmark, width, height)
            end_point = self._get_landmark_pixel_coordinates(end_landmark, width, height)
            
            # Determine color based on the connection
            color = self._get_connection_color(connection[0], connection[1])
            
            # Draw the line
            cv2.line(image, start_point, end_point, color, self.config.line_thickness, cv2.LINE_AA)
        
        # Draw landmarks
        for idx, landmark in landmarks.items():
            # Check confidence
            confidence = getattr(landmark, 'visibility', 0.0)
            if confidence < self.config.confidence_threshold:
                continue
                
            # Get pixel coordinates
            point = self._get_landmark_pixel_coordinates(landmark, width, height)
            
            # Determine color
            color = (100, 100, 100)  # Default gray
            landmark_enum = None
            
            # Find the enum for this landmark index
            for pose_landmark in self.mp_pose.PoseLandmark:
                if pose_landmark.value == idx:
                    landmark_enum = pose_landmark
                    break
            
            if landmark_enum:
                if 'LEFT' in landmark_enum.name:
                    color = self.config.left_color
                elif 'RIGHT' in landmark_enum.name:
                    color = self.config.right_color
                else:
                    color = self.config.center_color
            
            # Draw the landmark
            cv2.circle(image, point, self.config.joint_radius, color, -1, cv2.LINE_AA)
            
            # Draw joint name if enabled
            if self.config.show_joint_names and landmark_enum:
                cv2.putText(image, landmark_enum.name, (point[0] + 5, point[1] - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Draw angles if provided and enabled
        if angles:
            image = self._draw_angles(image, landmarks, angles)
        
        return image
    
    def release(self):
        """Release any resources held by the visualizer."""
        self.previous_landmarks = None 