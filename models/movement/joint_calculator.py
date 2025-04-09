import numpy as np
import mediapipe as mp
from typing import Dict, List, Optional, Tuple, Union, Any
import math

class JointCalculator:
    """
    JointCalculator handles ONLY the calculation of joint angles from landmarks.
    
    Responsibilities:
    - Joint angle calculations
    - Joint mapping definitions
    - Confidence scoring for calculated angles
    
    This class does NOT handle:
    - Pose detection
    - Video processing
    - Visualization
    - Exercise classification
    """
    
    def __init__(self):
        """Initialize the joint calculator."""
        # Initialize MediaPipe pose for landmarking constants
        self.mp_pose = mp.solutions.pose
        
        # Define joint mappings for angle calculations
        self.joint_mappings = self._create_joint_mappings()
    
    def _create_joint_mappings(self) -> Dict:
        """
        Create mappings for joint connections to calculate angles.
        
        Returns:
            Dictionary of joint mappings
        """
        return {
            'LEFT_SHOULDER': (self.mp_pose.PoseLandmark.LEFT_EAR, self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_ELBOW),
            'LEFT_ELBOW': (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.LEFT_WRIST),
            'LEFT_WRIST': (self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.LEFT_WRIST, self.mp_pose.PoseLandmark.LEFT_INDEX),
            'LEFT_HIP': (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_KNEE),
            'LEFT_KNEE': (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.LEFT_ANKLE),
            'LEFT_ANKLE': (self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.LEFT_ANKLE, self.mp_pose.PoseLandmark.LEFT_HEEL),
            'RIGHT_SHOULDER': (self.mp_pose.PoseLandmark.RIGHT_EAR, self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_ELBOW),
            'RIGHT_ELBOW': (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_ELBOW, self.mp_pose.PoseLandmark.RIGHT_WRIST),
            'RIGHT_WRIST': (self.mp_pose.PoseLandmark.RIGHT_ELBOW, self.mp_pose.PoseLandmark.RIGHT_WRIST, self.mp_pose.PoseLandmark.RIGHT_INDEX),
            'RIGHT_HIP': (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_KNEE),
            'RIGHT_KNEE': (self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_KNEE, self.mp_pose.PoseLandmark.RIGHT_ANKLE),
            'RIGHT_ANKLE': (self.mp_pose.PoseLandmark.RIGHT_KNEE, self.mp_pose.PoseLandmark.RIGHT_ANKLE, self.mp_pose.PoseLandmark.RIGHT_HEEL),
            # Additional joints for better analysis
            'NECK': (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.NOSE, self.mp_pose.PoseLandmark.RIGHT_SHOULDER),
            'TORSO': (self.mp_pose.PoseLandmark.NOSE, self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP)
        }
    
    def calculate_angle(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """
        Calculate the angle between three points in 3D space.
        
        Args:
            a: First point coordinates [x, y, z]
            b: Second point (vertex) coordinates [x, y, z]
            c: Third point coordinates [x, y, z]
            
        Returns:
            Angle in degrees or 0 if calculation fails
        """
        try:
            # Convert to numpy arrays if they aren't already
            a = np.array(a)
            b = np.array(b)
            c = np.array(c)
            
            # Create vectors
            ba = a - b
            bc = c - b
            
            # Calculate dot product and magnitudes
            dot_product = np.dot(ba, bc)
            magnitude_ba = np.linalg.norm(ba)
            magnitude_bc = np.linalg.norm(bc)
            
            # Calculate cosine of the angle
            cosine_angle = dot_product / (magnitude_ba * magnitude_bc)
            
            # Make sure the value is valid for arccos
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            
            # Calculate the angle
            angle_rad = np.arccos(cosine_angle)
            angle_deg = np.degrees(angle_rad)
            
            return float(angle_deg)
            
        except Exception:
            # Return default angle on error
            return 0.0
    
    def get_landmark_coords(self, landmark) -> Tuple[float, float, float]:
        """
        Extract coordinates from a landmark.
        
        Args:
            landmark: MediaPipe landmark object
            
        Returns:
            Tuple of (x, y, z) coordinates
        """
        if hasattr(landmark, 'x') and hasattr(landmark, 'y') and hasattr(landmark, 'z'):
            return [landmark.x, landmark.y, landmark.z]
        elif isinstance(landmark, (list, tuple)) and len(landmark) >= 3:
            return landmark[:3]
        else:
            # Return default coordinates on error
            return [0.0, 0.0, 0.0]
    
    def calculate_joint_angles(self, landmarks: Dict, joints_to_process: Optional[List[str]] = None) -> Dict:
        """
        Calculate angles for specified joints.
        
        Args:
            landmarks: Dictionary of detected landmarks
            joints_to_process: Optional list of joints to process (all if None)
            
        Returns:
            Dictionary of joint angles and confidence scores
        """
        # If no specific joints provided, process all joints
        if not joints_to_process:
            joints_to_process = list(self.joint_mappings.keys())
        
        angles_data = {}
        
        for joint_name in joints_to_process:
            if joint_name not in self.joint_mappings:
                continue
                
            # Get three landmarks defining the angle
            landmark_a, landmark_b, landmark_c = self.joint_mappings[joint_name]
            
            # Check if all landmarks are available
            if (landmark_a.value not in landmarks or 
                landmark_b.value not in landmarks or 
                landmark_c.value not in landmarks):
                continue
            
            # Get landmark objects
            a = landmarks[landmark_a.value]
            b = landmarks[landmark_b.value]
            c = landmarks[landmark_c.value]
            
            # Extract coordinates
            coords_a = self.get_landmark_coords(a)
            coords_b = self.get_landmark_coords(b)
            coords_c = self.get_landmark_coords(c)
            
            # Calculate angle
            angle = self.calculate_angle(coords_a, coords_b, coords_c)
            
            # Calculate confidence score (minimum visibility of the three landmarks)
            confidences = [
                getattr(a, 'visibility', 0.0),
                getattr(b, 'visibility', 0.0),
                getattr(c, 'visibility', 0.0)
            ]
            confidence = min(confidences)
            
            # Store angle and confidence
            angles_data[joint_name] = {
                'angle': angle,
                'confidence': confidence
            }
        
        return angles_data
    
    def add_custom_joint(self, joint_name: str, landmark_a: Any, landmark_b: Any, landmark_c: Any) -> bool:
        """
        Add a custom joint mapping.
        
        Args:
            joint_name: Name of the joint
            landmark_a: First landmark
            landmark_b: Second landmark (vertex)
            landmark_c: Third landmark
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.joint_mappings[joint_name] = (landmark_a, landmark_b, landmark_c)
            return True
        except Exception:
            return False
    
    def get_available_joints(self) -> List[str]:
        """
        Get list of available joint names.
        
        Returns:
            List of joint names
        """
        return list(self.joint_mappings.keys()) 