import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from dataclasses import dataclass
import os

@dataclass
class PoseEstimatorConfig:
    """Configuration for pose detection."""
    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    enable_segmentation: bool = False
    smooth_landmarks: bool = True
    max_cached_frames: int = 10  # For landmarks stabilization

class PoseEstimator:
    """
    PoseEstimator handles ONLY the detection of body landmarks using MediaPipe.
    
    Responsibilities:
    - Interfacing with MediaPipe for pose detection
    - Providing confidence values for detected landmarks
    - Basic landmark stabilization (caching and smoothing)
    - Converting between coordinate formats
    
    This class does NOT handle:
    - Joint angle calculations
    - Video processing
    - Visualization
    - Exercise classification
    """
    
    def __init__(self, config: Optional[PoseEstimatorConfig] = None):
        """
        Initialize the pose estimator.
        
        Args:
            config: Configuration for pose detection
        """
        self.config = config or PoseEstimatorConfig()
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = None  # Lazy initialization
        
        # Cache for landmark stabilization
        self.landmark_cache = []
        self.confidence_cache = []
    
    def _initialize_detector(self):
        """Initialize the MediaPipe pose detector if not already initialized."""
        if self.pose is None:
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=self.config.model_complexity,
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
                enable_segmentation=self.config.enable_segmentation,
                smooth_landmarks=self.config.smooth_landmarks
            )
    
    def detect_landmarks(self, frame: np.ndarray) -> Tuple[Dict, Dict, bool]:
        """
        Detect pose landmarks in a single frame.
        
        Args:
            frame: Input image frame as numpy array
            
        Returns:
            Tuple of (landmarks_dict, confidence_dict, success)
        """
        # Initialize detector if not already done
        self._initialize_detector()
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(rgb_frame)
        
        # Check if any pose was detected
        if not results.pose_landmarks:
            return {}, {}, False
        
        # Convert results to dictionary format
        landmarks_dict = {}
        confidence_dict = {}
        
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            landmarks_dict[idx] = landmark
            confidence_dict[idx] = landmark.visibility
        
        # Update cache for stabilization
        self._update_cache(landmarks_dict, confidence_dict)
        
        return landmarks_dict, confidence_dict, True
    
    def _update_cache(self, landmarks: Dict, confidence: Dict):
        """
        Update the landmark and confidence caches.
        
        Args:
            landmarks: Dictionary of landmarks
            confidence: Dictionary of confidence values
        """
        # Add to cache
        self.landmark_cache.append(landmarks)
        self.confidence_cache.append(confidence)
        
        # Maintain maximum cache size
        while len(self.landmark_cache) > self.config.max_cached_frames:
            self.landmark_cache.pop(0)
            self.confidence_cache.pop(0)
    
    def get_stabilized_landmarks(self) -> Dict:
        """
        Get landmarks stabilized using a weighted average of recent frames.
        
        Returns:
            Dictionary of stabilized landmarks
        """
        if not self.landmark_cache:
            return {}
        
        # If only one frame, return it directly
        if len(self.landmark_cache) == 1:
            return self.landmark_cache[0]
        
        # Create a weighted average based on recency and confidence
        result = {}
        
        # Get all landmark indices from the most recent detection
        landmark_indices = self.landmark_cache[-1].keys()
        
        for idx in landmark_indices:
            # Find all frames where this landmark exists
            valid_frames = [(i, cache) for i, cache in enumerate(self.landmark_cache) if idx in cache]
            
            if not valid_frames:
                continue
            
            # Apply weighted averaging
            x_sum = y_sum = z_sum = visibility_sum = 0.0
            total_weight = 0.0
            
            for i, cache in valid_frames:
                # Calculate weight: more recent = higher weight
                # Also consider confidence
                frame_age = len(self.landmark_cache) - 1 - i  # 0 for most recent
                recency_weight = 1.0 / (1.0 + frame_age)  # Decay with age
                
                # Use visibility as confidence weight
                confidence_weight = self.confidence_cache[i].get(idx, 0.1)
                
                # Combined weight
                weight = recency_weight * confidence_weight
                
                landmark = cache[idx]
                x_sum += landmark.x * weight
                y_sum += landmark.y * weight
                z_sum += landmark.z * weight
                visibility_sum += landmark.visibility * weight
                total_weight += weight
            
            # Create stabilized landmark
            if total_weight > 0:
                # Create a new landmark with the same attributes
                landmark_cls = type(self.landmark_cache[-1][idx])
                landmark = landmark_cls()
                
                # Set the stabilized values
                landmark.x = x_sum / total_weight
                landmark.y = y_sum / total_weight
                landmark.z = z_sum / total_weight
                landmark.visibility = visibility_sum / total_weight
                
                result[idx] = landmark
        
        return result
    
    def get_landmark_confidence(self, landmark_index: int) -> float:
        """
        Get confidence value for a specific landmark.
        
        Args:
            landmark_index: Index of the landmark
            
        Returns:
            Confidence value (0-1)
        """
        if not self.confidence_cache:
            return 0.0
            
        # Get confidence from most recent detection
        return self.confidence_cache[-1].get(landmark_index, 0.0)
    
    def clear_cache(self):
        """Clear the landmark and confidence caches."""
        self.landmark_cache.clear()
        self.confidence_cache.clear()
    
    def get_normalized_landmark_coordinates(self, landmark_index: int) -> Tuple[float, float, float]:
        """
        Get normalized (0-1) coordinates for a landmark.
        
        Args:
            landmark_index: Index of the landmark
            
        Returns:
            Tuple of (x, y, z) coordinates
        """
        if not self.landmark_cache:
            return (0.0, 0.0, 0.0)
            
        landmarks = self.landmark_cache[-1]
        if landmark_index not in landmarks:
            return (0.0, 0.0, 0.0)
            
        landmark = landmarks[landmark_index]
        return (landmark.x, landmark.y, landmark.z)
    
    def get_landmark_names(self) -> Dict[int, str]:
        """
        Get mapping of landmark indices to readable names.
        
        Returns:
            Dictionary mapping landmark indices to names
        """
        return {landmark.value: landmark.name for landmark in self.mp_pose.PoseLandmark}
    
    def release(self):
        """Release any resources held by the pose estimator."""
        if self.pose:
            # Close MediaPipe resources
            self.pose.close()
            self.pose = None
        
        # Clear caches
        self.clear_cache() 