import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Generator
import os
import json
from datetime import datetime

class PoseAnalyzer:
    """
    Core class for analyzing human poses using MediaPipe.
    This class handles the following:
    - Video and image processing
    - Pose landmark detection
    - Angle calculations between joints
    - Data collection for analysis
    """
    
    def __init__(self, model_complexity: int = 2, min_detection_confidence: float = 0.5, 
                 min_tracking_confidence: float = 0.7, enable_segmentation: bool = True):
        """
        Initialize the PoseAnalyzer with MediaPipe configuration.
        
        Args:
            model_complexity: MediaPipe model complexity (0, 1, or 2)
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
            enable_segmentation: Whether to enable body segmentation
        """
        # Initialize MediaPipe drawing and pose solutions
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        
        # Store configuration
        self.config = {
            'model_complexity': model_complexity,
            'min_detection_confidence': min_detection_confidence,
            'min_tracking_confidence': min_tracking_confidence,
            'enable_segmentation': enable_segmentation,
            'process_every_n_frames': 1
        }
        
        # Initialize pose detection model
        self.pose = self.configure_mediapipe()
        
        # Initialize variables to store angles data
        self.angles_data = []
        self.detected_joints = set()
        
        # Define joint mappings for angle calculations
        self.joint_mappings = self._create_joint_mappings()
    
    def _create_joint_mappings(self) -> Dict:
        """Create mappings for joint connections to calculate angles."""
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
            'RIGHT_ANKLE': (self.mp_pose.PoseLandmark.RIGHT_KNEE, self.mp_pose.PoseLandmark.RIGHT_ANKLE, self.mp_pose.PoseLandmark.RIGHT_HEEL)
        }
        
    def configure_mediapipe(self):
        """Configure and return a MediaPipe Pose instance."""
        return self.mp_pose.Pose(
            min_detection_confidence=self.config['min_detection_confidence'],
            min_tracking_confidence=self.config['min_tracking_confidence'],
            model_complexity=self.config['model_complexity'],
            enable_segmentation=self.config['enable_segmentation'],
            smooth_segmentation=True,
            static_image_mode=False
        )
        
    def calculate_angle(self, a: List[float], b: List[float], c: List[float]) -> float:
        """
        Calculate the angle between three points in 3D space.
        
        Args:
            a: First point coordinates [x, y, z]
            b: Second point (vertex) coordinates [x, y, z]
            c: Third point coordinates [x, y, z]
            
        Returns:
            Angle in degrees
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate angle using dot product
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def get_landmark_coords(self, landmark) -> List[float]:
        """Extract coordinates from a landmark."""
        return [landmark.x, landmark.y, landmark.z]
    
    def process_video(self, video_path: str, output_path: str = None, 
                      joints_to_process: List[str] = None) -> Generator[int, None, Tuple[bool, str]]:
        """
        Process a video file to detect and analyze poses.
        
        Args:
            video_path: Path to the input video file
            output_path: Path to save the output video (optional)
            joints_to_process: List of joint names to analyze
            
        Returns:
            Generator yielding progress percentage
            Final tuple of (success, message)
        """
        # Reset data
        self.angles_data = []
        self.detected_joints.clear()
        
        # Set default joints if none provided
        if joints_to_process is None:
            joints_to_process = list(self.joint_mappings.keys())
        
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, f"Error: Unable to open video file {video_path}"
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output video writer if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        else:
            out = None
        
        frame_count = 0
        try:
            with self.pose as pose:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    if frame_count % self.config['process_every_n_frames'] != 0:
                        continue
                    
                    # Process the frame
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = pose.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                    if results.pose_landmarks:
                        # Draw pose landmarks if output is required
                        if out:
                            self.mp_drawing.draw_landmarks(
                                image,
                                results.pose_landmarks,
                                self.mp_pose.POSE_CONNECTIONS,
                                self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                            )
                        
                        # Calculate joint angles
                        angles = self.calculate_angles(results, joints_to_process)
                        self.angles_data.append(angles)
                        self.detected_joints.update(angles.keys())
                    
                    # Write the annotated frame to output video
                    if out:
                        out.write(image)
                    
                    # Update progress
                    yield int((frame_count / total_frames) * 100)
        
        finally:
            cap.release()
            if out:
                out.release()
        
        if frame_count == 0:
            return False, f"Error: No frames were processed from {video_path}"
        
        return True, f"Successfully processed {frame_count} frames from {video_path}"
    
    def process_image(self, image_path: str, output_path: str = None, 
                      joints_to_process: List[str] = None) -> Tuple[bool, str]:
        """
        Process a single image to detect and analyze pose.
        
        Args:
            image_path: Path to the input image
            output_path: Path to save the annotated output image
            joints_to_process: List of joint names to analyze
            
        Returns:
            Tuple of (success, message)
        """
        # Reset data
        self.angles_data = []
        self.detected_joints.clear()
        
        # Set default joints if none provided
        if joints_to_process is None:
            joints_to_process = list(self.joint_mappings.keys())
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return False, f"Error: Unable to read image file {image_path}"
        
        # Process image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        with self.pose as pose:
            results = pose.process(image_rgb)
            
            if results.pose_landmarks:
                # Draw landmarks if output is required
                if output_path:
                    self.mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                    )
                
                # Calculate joint angles
                angles = self.calculate_angles(results, joints_to_process)
                self.angles_data.append(angles)
                self.detected_joints.update(angles.keys())
                
                # Save annotated image if output path provided
                if output_path:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    cv2.imwrite(output_path, image)
                
                return True, f"Successfully processed image"
            else:
                return False, "No pose landmarks detected in the image"
    
    def calculate_angles(self, results, joints_to_process: List[str]) -> Dict:
        """
        Calculate angles for specified joints from pose landmarks.
        
        Args:
            results: MediaPipe pose process results
            joints_to_process: List of joint names to analyze
            
        Returns:
            Dictionary of joint angles with confidences
        """
        angles = {}
        landmarks = results.pose_world_landmarks or results.pose_landmarks
        if not landmarks:
            return angles
        
        for joint in joints_to_process:
            if joint in self.joint_mappings:
                p1, p2, p3 = self.joint_mappings[joint]
                try:
                    coords1 = self.get_landmark_coords(landmarks.landmark[p1.value])
                    coords2 = self.get_landmark_coords(landmarks.landmark[p2.value])
                    coords3 = self.get_landmark_coords(landmarks.landmark[p3.value])
                    
                    visibility = min(landmarks.landmark[p.value].visibility for p in (p1, p2, p3))
                    
                    if visibility > 0.5:
                        angle = self.calculate_angle(coords1, coords2, coords3)
                        angles[joint] = {
                            'angle': angle,
                            'confidence': visibility
                        }
                    else:
                        angles[joint] = {
                            'angle': None,
                            'confidence': visibility,
                            'error': 'Low visibility'
                        }
                except Exception as e:
                    angles[joint] = {
                        'angle': None,
                        'confidence': 0,
                        'error': str(e)
                    }
        
        return angles
    
    def get_angles_data(self) -> List[Dict]:
        """Get the collected angles data."""
        return self.angles_data
    
    def get_detected_joints(self) -> set:
        """Get the set of detected joints."""
        return self.detected_joints