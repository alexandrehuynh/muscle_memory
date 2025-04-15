import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import math
import time

# Initialize MP pose solution for constants
mp_pose = mp.solutions.pose
PoseLandmark = mp_pose.PoseLandmark

# Define face landmark indices using PoseLandmark enum values
FACE_LANDMARK_INDICES = [
    PoseLandmark.NOSE.value,
    PoseLandmark.LEFT_EYE_INNER.value,
    PoseLandmark.LEFT_EYE.value,
    PoseLandmark.LEFT_EYE_OUTER.value,
    PoseLandmark.RIGHT_EYE_INNER.value,
    PoseLandmark.RIGHT_EYE.value,
    PoseLandmark.RIGHT_EYE_OUTER.value,
    PoseLandmark.LEFT_EAR.value,
    PoseLandmark.RIGHT_EAR.value,
    PoseLandmark.MOUTH_LEFT.value,
    PoseLandmark.MOUTH_RIGHT.value
]

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
    visibility_threshold: float = 0.5  # For landmark visibility checks
    connection_thickness: int = 5      # Line thickness for connections
    landmark_radius: int = 5
    landmark_thickness: int = 2
    smoothing_factor: float = 0.3
    show_reference_lines: bool = False
    left_color: Tuple[int, int, int] = (66, 245, 114)  # Green
    right_color: Tuple[int, int, int] = (245, 117, 66) # Orange
    center_color: Tuple[int, int, int] = (245, 66, 236) # Purple
    simplified_torso: bool = True                     # Use simplified torso with center line
    hide_face_landmarks: bool = True                  # Whether to hide face landmarks
    display_velocity: bool = False
    display_landmarks: bool = True
    
    @property
    def show_face_landmarks(self) -> bool:
        """Getter for show_face_landmarks (inverse of hide_face_landmarks)."""
        return not self.hide_face_landmarks
    
    def __post_init__(self):
        """Validate configuration values and set defaults."""
        # Ensure visibility_threshold is between 0 and 1
        self.visibility_threshold = max(0.0, min(1.0, self.visibility_threshold))

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
    
    def __init__(self, config: VisualizationConfig):
        """Initialize the visualizer with the provided configuration."""
        self.config = config
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp_pose
        
        # Load pose model with specified confidence thresholds
        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5,  # Default values since config may not have these
            min_tracking_confidence=0.5
        )
        
        # Initialize the connections
        self._initialize_connections()
        
        # MediaPipe pose solution for constants
        self.mp_pose = mp_pose
        
        # Define the connections between landmarks for drawing
        self.connections = self._create_connections()
        
        # Previous landmarks for smoothing
        self.previous_landmarks = None
    
    def _create_connections(self) -> List[Tuple[int, int]]:
        """
        Create connections for pose landmarks.
        
        Returns:
            List of connections as tuples of landmark indices
        """
        connections = [
            # Arms
            (PoseLandmark.LEFT_SHOULDER.value, PoseLandmark.LEFT_ELBOW.value),
            (PoseLandmark.LEFT_ELBOW.value, PoseLandmark.LEFT_WRIST.value),
            (PoseLandmark.RIGHT_SHOULDER.value, PoseLandmark.RIGHT_ELBOW.value),
            (PoseLandmark.RIGHT_ELBOW.value, PoseLandmark.RIGHT_WRIST.value),
            
            # Legs
            (PoseLandmark.LEFT_HIP.value, PoseLandmark.LEFT_KNEE.value),
            (PoseLandmark.LEFT_KNEE.value, PoseLandmark.LEFT_ANKLE.value),
            (PoseLandmark.RIGHT_HIP.value, PoseLandmark.RIGHT_KNEE.value),
            (PoseLandmark.RIGHT_KNEE.value, PoseLandmark.RIGHT_ANKLE.value),
        ]
        
        # Only add torso box if not using simplified torso
        if not self.config.simplified_torso:
            connections.extend([
                # Torso
                (PoseLandmark.LEFT_SHOULDER.value, PoseLandmark.RIGHT_SHOULDER.value),
                (PoseLandmark.LEFT_SHOULDER.value, PoseLandmark.LEFT_HIP.value),
                (PoseLandmark.RIGHT_SHOULDER.value, PoseLandmark.RIGHT_HIP.value),
                (PoseLandmark.LEFT_HIP.value, PoseLandmark.RIGHT_HIP.value),
            ])
        
        # Add face connections if not hidden
        if not self.config.hide_face_landmarks:
            connections.extend([
                # Face
                (PoseLandmark.NOSE.value, PoseLandmark.LEFT_EYE_INNER.value),
                (PoseLandmark.LEFT_EYE_INNER.value, PoseLandmark.LEFT_EYE.value),
                (PoseLandmark.LEFT_EYE.value, PoseLandmark.LEFT_EYE_OUTER.value),
                (PoseLandmark.LEFT_EYE_OUTER.value, PoseLandmark.LEFT_EAR.value),
                (PoseLandmark.NOSE.value, PoseLandmark.RIGHT_EYE_INNER.value),
                (PoseLandmark.RIGHT_EYE_INNER.value, PoseLandmark.RIGHT_EYE.value),
                (PoseLandmark.RIGHT_EYE.value, PoseLandmark.RIGHT_EYE_OUTER.value),
                (PoseLandmark.RIGHT_EYE_OUTER.value, PoseLandmark.RIGHT_EAR.value),
                (PoseLandmark.LEFT_SHOULDER.value, PoseLandmark.NOSE.value),
                (PoseLandmark.RIGHT_SHOULDER.value, PoseLandmark.NOSE.value),
            ])
        
        return connections
    
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
    
    def _get_connection_color(self, start_landmark_idx: int, end_landmark_idx: int) -> Tuple[int, int, int]:
        """Get color for a connection based on body side."""
        # Convert indices to PoseLandmark enum values
        try:
            start_landmark = PoseLandmark(start_landmark_idx)
            end_landmark = PoseLandmark(end_landmark_idx)
        except ValueError:
            # If conversion fails, return center color
            return self.config.center_color
            
        # Left side connections use left color
        left_landmarks = [
            PoseLandmark.LEFT_SHOULDER,
            PoseLandmark.LEFT_ELBOW,
            PoseLandmark.LEFT_WRIST,
            PoseLandmark.LEFT_HIP,
            PoseLandmark.LEFT_KNEE, 
            PoseLandmark.LEFT_ANKLE,
            PoseLandmark.LEFT_EYE,
            PoseLandmark.LEFT_EYE_INNER,
            PoseLandmark.LEFT_EYE_OUTER,
            PoseLandmark.LEFT_EAR
        ]
        
        # Right side connections use right color
        right_landmarks = [
            PoseLandmark.RIGHT_SHOULDER,
            PoseLandmark.RIGHT_ELBOW,
            PoseLandmark.RIGHT_WRIST,
            PoseLandmark.RIGHT_HIP,
            PoseLandmark.RIGHT_KNEE,
            PoseLandmark.RIGHT_ANKLE,
            PoseLandmark.RIGHT_EYE,
            PoseLandmark.RIGHT_EYE_INNER,
            PoseLandmark.RIGHT_EYE_OUTER,
            PoseLandmark.RIGHT_EAR
        ]
        
        # Determine color based on connection ends
        if start_landmark in left_landmarks and end_landmark in left_landmarks:
            return self.config.left_color
        elif start_landmark in right_landmarks and end_landmark in right_landmarks:
            return self.config.right_color
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
        left_shoulder_idx = PoseLandmark.LEFT_SHOULDER.value
        right_shoulder_idx = PoseLandmark.RIGHT_SHOULDER.value
        
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
            'LEFT_SHOULDER': PoseLandmark.LEFT_SHOULDER.value,
            'RIGHT_SHOULDER': PoseLandmark.RIGHT_SHOULDER.value,
            'LEFT_ELBOW': PoseLandmark.LEFT_ELBOW.value,
            'RIGHT_ELBOW': PoseLandmark.RIGHT_ELBOW.value,
            'LEFT_WRIST': PoseLandmark.LEFT_WRIST.value,
            'RIGHT_WRIST': PoseLandmark.RIGHT_WRIST.value,
            'LEFT_HIP': PoseLandmark.LEFT_HIP.value,
            'RIGHT_HIP': PoseLandmark.RIGHT_HIP.value,
            'LEFT_KNEE': PoseLandmark.LEFT_KNEE.value,
            'RIGHT_KNEE': PoseLandmark.RIGHT_KNEE.value,
            'LEFT_ANKLE': PoseLandmark.LEFT_ANKLE.value,
            'RIGHT_ANKLE': PoseLandmark.RIGHT_ANKLE.value
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
            text = f"{int(angle)}"
            cv2.putText(image, text, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, text, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        return image
    
    def _calculate_midpoint(self, point1, point2) -> Tuple[float, float, float]:
        """
        Calculate midpoint between two points.
        
        Args:
            point1: First point with x, y, z coordinates
            point2: Second point with x, y, z coordinates
            
        Returns:
            Tuple of (x, y, z) coordinates
        """
        mid_x = (point1.x + point2.x) / 2
        mid_y = (point1.y + point2.y) / 2
        mid_z = (point1.z + point2.z) / 2
        
        # Create a new Point-like object with the same structure as landmarks
        class Point:
            def __init__(self, x, y, z, visibility=1.0):
                self.x = x
                self.y = y
                self.z = z
                self.visibility = visibility
        
        # Calculate average visibility
        visibility = (getattr(point1, 'visibility', 1.0) + getattr(point2, 'visibility', 1.0)) / 2
        
        return Point(mid_x, mid_y, mid_z, visibility)
    
    def _draw_simplified_torso(self, image: np.ndarray, landmarks: Dict) -> np.ndarray:
        """
        Draw simplified torso as a single center line.
        
        Args:
            image: Image to draw on
            landmarks: Dictionary of landmarks
            
        Returns:
            Image with simplified torso line
        """
        height, width, _ = image.shape
        
        # Check if required landmarks exist
        left_shoulder_idx = PoseLandmark.LEFT_SHOULDER.value
        right_shoulder_idx = PoseLandmark.RIGHT_SHOULDER.value
        left_hip_idx = PoseLandmark.LEFT_HIP.value
        right_hip_idx = PoseLandmark.RIGHT_HIP.value
        
        if (left_shoulder_idx not in landmarks or right_shoulder_idx not in landmarks or
            left_hip_idx not in landmarks or right_hip_idx not in landmarks):
            return image
            
        # Get landmarks
        left_shoulder = landmarks[left_shoulder_idx]
        right_shoulder = landmarks[right_shoulder_idx]
        left_hip = landmarks[left_hip_idx]
        right_hip = landmarks[right_hip_idx]
        
        # Calculate midpoints
        mid_shoulder = self._calculate_midpoint(left_shoulder, right_shoulder)
        mid_hip = self._calculate_midpoint(left_hip, right_hip)
        
        # Get pixel coordinates
        mid_shoulder_point = self._get_landmark_pixel_coordinates(mid_shoulder, width, height)
        mid_hip_point = self._get_landmark_pixel_coordinates(mid_hip, width, height)
        
        # Draw the center line
        color = self.config.center_color
        cv2.line(image, mid_shoulder_point, mid_hip_point, color, self.config.connection_thickness, cv2.LINE_AA)
        
        return image
    
    def visualize(self, frame: np.ndarray, landmarks: List, angles: Optional[Dict] = None) -> np.ndarray:
        """
        Visualize the pose landmarks on the frame.
        
        Args:
            frame: The frame to visualize on
            landmarks: The landmarks to visualize (list of NormalizedLandmark or list of coordinates)
            angles: Optional dictionary of joint angles
        
        Returns:
            The frame with visualizations
        """
        if landmarks is None or len(landmarks) == 0:
            return frame
        
        # Create a copy of the frame
        output_frame = frame.copy()
        
        # Draw reference lines if enabled
        if self.config.show_reference_lines:
            output_frame = self._draw_reference_lines(output_frame, landmarks)
        
        # Store landmarks for velocity calculation
        if self.config.display_velocity:
            current_time = time.time()
            self._update_velocity(landmarks, current_time)
        
        # Draw connections between landmarks
        self._draw_landmark_connections(output_frame, landmarks)
        
        # Draw landmarks if enabled
        if self.config.display_landmarks:
            self._draw_landmarks(output_frame, landmarks)
        
        # Draw angles if provided
        if angles and self.config.show_angles:
            self._draw_angles(output_frame, landmarks, angles)
        
        return output_frame
        
    def release(self):
        """Release any resources held by the visualizer."""
        self.previous_landmarks = None

    def _draw_landmark_connections(self, frame: np.ndarray, landmarks: List) -> None:
        """Draw connections between landmarks."""
        height, width, _ = frame.shape
        
        # Check if landmarks are NormalizedLandmark objects or lists
        is_normalized_landmark = hasattr(landmarks[0], 'x') if landmarks else False
        
        if self.config.simplified_torso:
            # Draw simplified torso (center line)
            left_shoulder_idx = PoseLandmark.LEFT_SHOULDER.value
            right_shoulder_idx = PoseLandmark.RIGHT_SHOULDER.value
            left_hip_idx = PoseLandmark.LEFT_HIP.value
            right_hip_idx = PoseLandmark.RIGHT_HIP.value
            
            if all(idx < len(landmarks) for idx in [left_shoulder_idx, right_shoulder_idx, left_hip_idx, right_hip_idx]):
                # Calculate midpoints
                if is_normalized_landmark:
                    # For NormalizedLandmark objects
                    mid_shoulder_x = int((landmarks[left_shoulder_idx].x + landmarks[right_shoulder_idx].x) * width / 2)
                    mid_shoulder_y = int((landmarks[left_shoulder_idx].y + landmarks[right_shoulder_idx].y) * height / 2)
                    mid_hip_x = int((landmarks[left_hip_idx].x + landmarks[right_hip_idx].x) * width / 2)
                    mid_hip_y = int((landmarks[left_hip_idx].y + landmarks[right_hip_idx].y) * height / 2)
                else:
                    # For list-based landmarks
                    mid_shoulder_x = int((landmarks[left_shoulder_idx][0] + landmarks[right_shoulder_idx][0]) * width / 2)
                    mid_shoulder_y = int((landmarks[left_shoulder_idx][1] + landmarks[right_shoulder_idx][1]) * height / 2)
                    mid_hip_x = int((landmarks[left_hip_idx][0] + landmarks[right_hip_idx][0]) * width / 2)
                    mid_hip_y = int((landmarks[left_hip_idx][1] + landmarks[right_hip_idx][1]) * height / 2)
                
                # Define points and color for simplified torso
                start_point = (mid_shoulder_x, mid_shoulder_y)
                end_point = (mid_hip_x, mid_hip_y)
                color = self.config.center_color
                
                # Draw center line
                cv2.line(frame, start_point, end_point, color, self.config.connection_thickness)
        
        # Draw other connections
        for connection in self.connections:
            start_idx = connection[0]
            end_idx = connection[1]
            
            # Skip if either landmark is out of range
            if start_idx >= len(landmarks) or end_idx >= len(landmarks):
                continue
                
            # Skip if either landmark has low visibility
            if is_normalized_landmark:
                start_visibility = getattr(landmarks[start_idx], 'visibility', 0.0)
                end_visibility = getattr(landmarks[end_idx], 'visibility', 0.0)
            else:
                start_visibility = landmarks[start_idx][2]
                end_visibility = landmarks[end_idx][2]
                
            if start_visibility < self.config.confidence_threshold or end_visibility < self.config.confidence_threshold:
                continue
            
            # Skip face landmarks if hide_face_landmarks is enabled
            if self.config.hide_face_landmarks:
                if start_idx in FACE_LANDMARK_INDICES or end_idx in FACE_LANDMARK_INDICES:
                    continue
            
            # Get pixel coordinates
            if is_normalized_landmark:
                start_point = (int(landmarks[start_idx].x * width), int(landmarks[start_idx].y * height))
                end_point = (int(landmarks[end_idx].x * width), int(landmarks[end_idx].y * height))
            else:
                start_point = (int(landmarks[start_idx][0] * width), int(landmarks[start_idx][1] * height))
                end_point = (int(landmarks[end_idx][0] * width), int(landmarks[end_idx][1] * height))
            
            # Get color based on connection
            color = self._get_connection_color(start_idx, end_idx)
            
            # Draw line
            cv2.line(frame, start_point, end_point, color, self.config.connection_thickness)

    def _draw_landmarks(self, image: np.ndarray, landmarks: List):
        """
        Draw landmarks on the image.
        
        Args:
            image: Image to draw on
            landmarks: List of landmark coordinates or NormalizedLandmark objects
        """
        h, w, _ = image.shape
        
        # Check if landmarks are NormalizedLandmark objects or lists
        is_normalized_landmark = hasattr(landmarks[0], 'x') if landmarks else False
        
        for idx, landmark in enumerate(landmarks):
            # Skip landmarks with low visibility
            if is_normalized_landmark:
                visibility = getattr(landmark, 'visibility', 0.0)
            else:
                visibility = landmark[2]
                
            if visibility < self.config.confidence_threshold:
                continue
            
            # Skip face landmarks if configured to hide them
            if self.config.hide_face_landmarks and idx in FACE_LANDMARK_INDICES:
                continue
            
            try:
                # Determine color based on left/right side
                landmark_name = PoseLandmark(idx).name
                if "LEFT" in landmark_name:
                    color = self.config.left_color
                elif "RIGHT" in landmark_name:
                    color = self.config.right_color
                else:
                    color = self.config.center_color
            except ValueError:
                # If idx is not a valid PoseLandmark enum value
                color = self.config.center_color
                    
            # Draw the landmark
            if is_normalized_landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
            else:
                x, y = int(landmark[0] * w), int(landmark[1] * h)
                
            cv2.circle(image, (x, y), self.config.landmark_radius, color, self.config.landmark_thickness)
            
    def _initialize_connections(self):
        """Initialize the connections between landmarks."""
        # Clear existing connections
        self.connections = []
        
        # Don't add any face connections if hide_face_landmarks is enabled
        if not self.config.hide_face_landmarks:
            # Face mesh connections
            # Face oval
            self.connections.extend([
                (PoseLandmark.NOSE.value, PoseLandmark.RIGHT_EYE_INNER.value),
                (PoseLandmark.RIGHT_EYE_INNER.value, PoseLandmark.RIGHT_EYE.value),
                (PoseLandmark.RIGHT_EYE.value, PoseLandmark.RIGHT_EYE_OUTER.value),
                (PoseLandmark.RIGHT_EYE_OUTER.value, PoseLandmark.RIGHT_EAR.value),
                
                (PoseLandmark.NOSE.value, PoseLandmark.LEFT_EYE_INNER.value),
                (PoseLandmark.LEFT_EYE_INNER.value, PoseLandmark.LEFT_EYE.value),
                (PoseLandmark.LEFT_EYE.value, PoseLandmark.LEFT_EYE_OUTER.value),
                (PoseLandmark.LEFT_EYE_OUTER.value, PoseLandmark.LEFT_EAR.value),
                
                (PoseLandmark.MOUTH_RIGHT.value, PoseLandmark.MOUTH_LEFT.value),
            ])
        
        # Add body connections
        # Right body side
        self.connections.extend([
            (PoseLandmark.RIGHT_SHOULDER.value, PoseLandmark.RIGHT_ELBOW.value),
            (PoseLandmark.RIGHT_ELBOW.value, PoseLandmark.RIGHT_WRIST.value),
            (PoseLandmark.RIGHT_HIP.value, PoseLandmark.RIGHT_KNEE.value),
            (PoseLandmark.RIGHT_KNEE.value, PoseLandmark.RIGHT_ANKLE.value),
        ])
        
        # Left body side
        self.connections.extend([
            (PoseLandmark.LEFT_SHOULDER.value, PoseLandmark.LEFT_ELBOW.value),
            (PoseLandmark.LEFT_ELBOW.value, PoseLandmark.LEFT_WRIST.value),
            (PoseLandmark.LEFT_HIP.value, PoseLandmark.LEFT_KNEE.value),
            (PoseLandmark.LEFT_KNEE.value, PoseLandmark.LEFT_ANKLE.value),
        ])
        
        # Only add torso box connections if simplified_torso is False
        if not self.config.simplified_torso:
            # Torso box
            self.connections.extend([
                (PoseLandmark.LEFT_SHOULDER.value, PoseLandmark.RIGHT_SHOULDER.value),
                (PoseLandmark.RIGHT_SHOULDER.value, PoseLandmark.RIGHT_HIP.value),
                (PoseLandmark.RIGHT_HIP.value, PoseLandmark.LEFT_HIP.value),
                (PoseLandmark.LEFT_HIP.value, PoseLandmark.LEFT_SHOULDER.value),
            ])
    
    def _update_velocity(self, landmarks: List[List[float]], current_time: float):
        """
        Update velocity calculations for landmarks.
        
        Args:
            landmarks: Current frame landmarks
            current_time: Current timestamp
        """
        # This is a stub method for future implementation
        # Will be used to calculate and display velocity of landmarks
        pass 