import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import mediapipe as mp
from dataclasses import dataclass, field
from enum import Enum
import logging
import math

class VisualizationMode(Enum):
    """Enum for different visualization modes."""
    REALTIME = "realtime"  # Simplified skeleton for live coaching
    ANALYSIS = "analysis"  # Detailed skeleton for post-workout review

@dataclass
class VisualizationConfig:
    """Configuration for skeleton visualization."""
    mode: VisualizationMode = VisualizationMode.REALTIME
    exercise_type: Optional[str] = None
    camera_angle: str = "front"  # front, side, or 45
    show_reference_lines: bool = True
    show_angles: bool = True
    show_motion_trails: bool = False
    confidence_threshold: float = 0.5
    smoothing_window: int = 5
    smoothing_factor: float = 0.3  # Reduced from 0.5 to 0.3 for better responsiveness
    min_detection_threshold: float = 0.2  # Discard very low confidence detections
    line_style: str = "solid"  # Options: "solid", "dashed", "outlined"
    show_labels: bool = False
    critical_joints: List[str] = field(default_factory=list)
    motion_trail_length: int = 5  # Reduced from 10 to 5
    response_priority: bool = False  # When True, reduces smoothing for more responsive tracking
    fade_in_frames: int = 5  # Number of frames for fade-in effect
    fade_out_frames: int = 8  # Number of frames for fade-out effect
    hysteresis_range: float = 0.1  # Range for hysteresis to prevent flickering

class SkeletonRenderer:
    """
    Enhanced skeleton visualization system with anatomical improvements and reference lines.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize the skeleton renderer.
        
        Args:
            config: Optional configuration for visualization
        """
        self.config = config or VisualizationConfig()
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Add maximum history size
        self.max_history_size = 200  # Maximum frames to store
        
        # Define safety checks for coordinate validation
        self.SAFETY_CHECKS = {
            'max_retries': 3,
            'coord_threshold': 1.5,  # Allow slight overshoot for motion trails
            'confidence_floor': 0.1
        }
        
        # Colors for different visualization elements (BGR format)
        self.colors = {
            'skeleton': (66, 117, 245),  # Orange-red
            'skeleton_low_conf': (66, 117, 245, 128),  # Orange-red with low opacity
            'joints': (230, 66, 245),    # Pink
            'joints_low_conf': (230, 66, 245, 128),  # Pink with low opacity
            'reference': (230, 245, 66), # Bright cyan
            'angles': (66, 245, 230),   # Yellow
            'feedback_good': (46, 235, 46),  # Brighter green
            'feedback_bad': (46, 46, 235),   # Brighter red
            'motion_trail': (255, 76, 127),   # Brighter purple
            'motion_trail_old': (76, 114, 255),  # Different color for older trails
            'spine_line': (255, 255, 255),  # White
            'shoulder_line': (255, 255, 255),  # White
            'hip_line': (255, 255, 255),  # White
            'knee_projection': (86, 237, 227),  # Brighter orange
            'left_limb': (255, 0, 0),  # Red for left limb
            'right_limb': (0, 0, 255)  # Blue for right limb
        }
        
        # Store previous landmarks for motion trails and smoothing
        self.previous_landmarks = None
        self.previous_landmarks_confidence = 0.0  # Track confidence of previous landmarks
        self.frames_since_good_detection = 0  # Track how many frames since last good detection
        self.motion_trail = []
        self.max_trail_length = 5  # Reduced from 10 to 5
        self.previous_landmark_history = []
        self.motion_trail_length = getattr(self.config, 'motion_trail_length', 5)  # Reduced from 10 to 5
        
        # For confidence tracking with exponential weighted moving average
        self.landmark_confidence_ewma = {}  # Store EWMA of confidence values
        self.ewma_alpha = 0.3  # Weight for new values vs historical (0.3 means 30% new, 70% historical)
        
        # Enhanced visibility state tracking for hysteresis and fade effects
        self.landmark_visibility_state = {}  # 0=hidden, 1=visible, values in between for fading
        self.fade_in_counter = {}  # Track frames during fade-in
        self.fade_out_counter = {}  # Track frames during fade-out
        
        # Initialize exercise-specific configurations
        self._init_exercise_configs()
    
    def _init_exercise_configs(self):
        """Initialize exercise-specific visualization configurations."""
        self.exercise_configs = {
            'squat': {
                'key_joints': ['LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_HIP', 'RIGHT_HIP'],
                'reference_lines': ['vertical', 'hip_line', 'knee_projection'],
                'angle_thresholds': {
                    'knee': (70, 120),  # Min, max knee angle
                    'hip': (90, 180),   # Min, max hip angle
                    'spine': (160, 180)  # Min, max spine angle (should stay straight)
                },
                'depth_reference': 0.65  # Target squat depth (relative to standing height)
            },
            'lunge': {
                'key_joints': ['LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_HIP', 'RIGHT_HIP'],
                'reference_lines': ['vertical', 'hip_line', 'front_knee_projection'],
                'angle_thresholds': {
                    'knee': (60, 120),
                    'hip': (90, 180),
                    'front_knee': (80, 100)  # Front knee should be close to 90 degrees
                }
            },
            'pushup': {
                'key_joints': ['LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_SHOULDER', 'RIGHT_SHOULDER'],
                'reference_lines': ['vertical', 'shoulder_line', 'elbow_projection'],
                'angle_thresholds': {
                    'elbow': (70, 160),
                    'shoulder': (90, 180),
                    'spine': (160, 180)  # Spine should remain straight
                }
            }
        }
    
    def _get_exercise_config(self) -> Dict:
        """Get the configuration for the current exercise type."""
        return self.exercise_configs.get(self.config.exercise_type, {})
    
    def _validate_landmarks(self, landmarks: Dict) -> bool:
        """
        Ensure landmarks use image-relative coordinates.
        
        Args:
            landmarks: Dictionary of MediaPipe pose landmarks
            
        Returns:
            True if landmarks are valid, False otherwise
        """
        if not landmarks:
            return False
            
        for idx, landmark in landmarks.items():
            # Check for coordinate validity (should be normalized to 0-1 range)
            # Allow slight overshooting for potential motion trails or edge cases
            if not (0 <= landmark.x <= self.SAFETY_CHECKS['coord_threshold'] and 
                    0 <= landmark.y <= self.SAFETY_CHECKS['coord_threshold']):
                logging.warning(f"Invalid landmark coordinates at index {idx}: {landmark.x}, {landmark.y}")
                return False
                
            # Ensure visibility attribute exists
            if not hasattr(landmark, 'visibility'):
                logging.warning(f"Landmark at index {idx} missing visibility attribute")
                return False
                
        return True

    def _generate_virtual_landmarks(self, landmarks: Dict) -> Dict:
        """
        Generate additional virtual landmarks for enhanced visualization.
        
        Args:
            landmarks: Dictionary of MediaPipe pose landmarks indexed by PoseLandmark enum values
            
        Returns:
            Dictionary of virtual landmarks
        """
        virtual_landmarks = {}
        
        try:
            # Check if necessary landmarks exist with sufficient visibility
            required_landmarks = [
                self.mp_pose.PoseLandmark.NOSE,
                self.mp_pose.PoseLandmark.LEFT_SHOULDER, 
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                self.mp_pose.PoseLandmark.LEFT_HIP, 
                self.mp_pose.PoseLandmark.RIGHT_HIP
            ]
            
            if not all(landmark.value in landmarks for landmark in required_landmarks):
                return virtual_landmarks
                
            # Safety checks for coordinate values
            for landmark_idx in [lm.value for lm in required_landmarks]:
                if landmark_idx in landmarks:
                    lm = landmarks[landmark_idx]
                    if not (0 <= lm.x <= self.SAFETY_CHECKS['coord_threshold'] and 
                            0 <= lm.y <= self.SAFETY_CHECKS['coord_threshold']):
                        logging.warning(f"Invalid required landmark coordinates: {lm.x}, {lm.y}")
                        return virtual_landmarks  # Return empty if invalid
                    
                    # Ensure minimum confidence
                    if getattr(lm, 'visibility', 0) < self.SAFETY_CHECKS['confidence_floor']:
                        logging.warning(f"Required landmark has too low confidence: {getattr(lm, 'visibility', 0)}")
                
            # Get key landmarks
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            left_knee = landmarks.get(self.mp_pose.PoseLandmark.LEFT_KNEE.value)
            right_knee = landmarks.get(self.mp_pose.PoseLandmark.RIGHT_KNEE.value)
            left_ankle = landmarks.get(self.mp_pose.PoseLandmark.LEFT_ANKLE.value)
            right_ankle = landmarks.get(self.mp_pose.PoseLandmark.RIGHT_ANKLE.value)
            
            # Calculate mid points
            mid_shoulder = self._calculate_midpoint(left_shoulder, right_shoulder)
            mid_hip = self._calculate_midpoint(left_hip, right_hip)
            
            # Create spine segments (at least 3 segments as required)
            # 1. Cervical spine (neck to mid shoulders)
            virtual_landmarks['NECK'] = self._calculate_midpoint(nose, mid_shoulder, 0.7)  # Closer to shoulders
            virtual_landmarks['CERVICAL_SPINE'] = self._calculate_midpoint(virtual_landmarks['NECK'], mid_shoulder)
            
            # 2. Thoracic spine (mid shoulders to mid torso)
            virtual_landmarks['MID_SPINE'] = self._calculate_midpoint(mid_shoulder, mid_hip)
            virtual_landmarks['THORACIC_SPINE_UPPER'] = self._calculate_midpoint(mid_shoulder, virtual_landmarks['MID_SPINE'], 0.3)
            virtual_landmarks['THORACIC_SPINE_LOWER'] = self._calculate_midpoint(mid_shoulder, virtual_landmarks['MID_SPINE'], 0.7)
            
            # 3. Lumbar spine (mid torso to mid hips)
            virtual_landmarks['LUMBAR_SPINE_UPPER'] = self._calculate_midpoint(virtual_landmarks['MID_SPINE'], mid_hip, 0.3)
            virtual_landmarks['LUMBAR_SPINE_LOWER'] = self._calculate_midpoint(virtual_landmarks['MID_SPINE'], mid_hip, 0.7)
            
            # Create reference line points
            # Vertical plumb line from top to bottom
            highest_y = min(nose.y, left_shoulder.y, right_shoulder.y)
            virtual_landmarks['PLUMB_LINE_TOP'] = (mid_shoulder.x, highest_y - 0.05, 0)
            virtual_landmarks['PLUMB_LINE_BOTTOM'] = (mid_shoulder.x, 1.5, 0)  # Extend beyond feet
            
            # Extended hip and shoulder lines
            hip_width = self._calculate_distance(left_hip, right_hip)
            virtual_landmarks['HIP_LINE_LEFT'] = (left_hip.x - hip_width * 0.2, left_hip.y, left_hip.z)
            virtual_landmarks['HIP_LINE_RIGHT'] = (right_hip.x + hip_width * 0.2, right_hip.y, right_hip.z)
            
            shoulder_width = self._calculate_distance(left_shoulder, right_shoulder)
            virtual_landmarks['SHOULDER_LINE_LEFT'] = (left_shoulder.x - shoulder_width * 0.2, left_shoulder.y, left_shoulder.z)
            virtual_landmarks['SHOULDER_LINE_RIGHT'] = (right_shoulder.x + shoulder_width * 0.2, right_shoulder.y, right_shoulder.z)
            
            # Exercise-specific reference points
            if self.config.exercise_type == 'squat' and left_knee and right_knee and left_ankle and right_ankle:
                # Target squat depth
                standing_height = self._calculate_distance(mid_shoulder, mid_hip)
                depth_factor = self.exercise_configs['squat']['depth_reference']
                virtual_landmarks['SQUAT_DEPTH'] = (mid_hip.x, mid_hip.y + standing_height * depth_factor, mid_hip.z)
                
                # Knee projection references (for tracking knees over toes)
                virtual_landmarks['LEFT_KNEE_PROJECTION'] = (left_ankle.x, left_knee.y, left_ankle.z)
                virtual_landmarks['RIGHT_KNEE_PROJECTION'] = (right_ankle.x, right_knee.y, right_ankle.z)
                
            elif self.config.exercise_type == 'pushup':
                # Extended spine line for pushups
                if left_ankle and right_ankle:
                    mid_ankle = self._calculate_midpoint(left_ankle, right_ankle)
                    virtual_landmarks['EXTENDED_SPINE_LINE'] = (
                        mid_shoulder.x + (mid_shoulder.x - mid_ankle.x) * 0.5,
                        mid_shoulder.y + (mid_shoulder.y - mid_ankle.y) * 0.5,
                        mid_shoulder.z + (mid_shoulder.z - mid_ankle.z) * 0.5
                    )
            
        except Exception as e:
            logging.error(f"Error generating virtual landmarks: {e}")
            
        return virtual_landmarks
    
    def _calculate_midpoint(self, p1, p2, ratio: float = 0.5) -> Any:
        """
        Calculate the midpoint or point at given ratio between two points.
        
        Args:
            p1: First point
            p2: Second point
            ratio: Ratio of the distance from p1 to p2 (0.5 = midpoint)
            
        Returns:
            Point object with x, y, z coordinates
        """
        # Create a simple point class to match landmark structure
        class Point:
            def __init__(self, x, y, z=0, visibility=1.0):
                self.x = x
                self.y = y
                self.z = z
                self.visibility = visibility
        
        x = p1.x + (p2.x - p1.x) * ratio
        y = p1.y + (p2.y - p1.y) * ratio
        z = getattr(p1, 'z', 0) + (getattr(p2, 'z', 0) - getattr(p1, 'z', 0)) * ratio
        visibility = min(getattr(p1, 'visibility', 1.0), getattr(p2, 'visibility', 1.0))
        
        return Point(x, y, z, visibility)
    
    def _calculate_distance(self, p1, p2) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
    
    def _get_confidence(self, landmark) -> float:
        """Get the confidence value from a landmark."""
        return getattr(landmark, 'visibility', 1.0)
    
    def _get_landmark_visibility_state(self, landmark_key, raw_confidence):
        """
        Get landmark visibility state with hysteresis and fade effects.
        
        Args:
            landmark_key: Unique key for the landmark
            raw_confidence: Raw confidence value from landmark
            
        Returns:
            Visibility state value between 0 and 1
        """
        # Initialize state if not exists
        if landmark_key not in self.landmark_visibility_state:
            self.landmark_visibility_state[landmark_key] = 0.0 if raw_confidence < self.config.confidence_threshold else 1.0
            self.fade_in_counter[landmark_key] = 0
            self.fade_out_counter[landmark_key] = 0
            
        current_state = self.landmark_visibility_state[landmark_key]
        hysteresis_range = getattr(self.config, 'hysteresis_range', 0.1)
        
        # Apply hysteresis to prevent flickering at threshold boundaries
        threshold_with_hysteresis = (
            self.config.confidence_threshold - hysteresis_range 
            if current_state >= 0.5 
            else self.config.confidence_threshold + hysteresis_range
        )
        
        # Determine target state (fully visible or fully hidden)
        target_state = 1.0 if raw_confidence >= threshold_with_hysteresis else 0.0
        
        # Implement fade-in/fade-out logic
        fade_in_frames = getattr(self.config, 'fade_in_frames', 5)
        fade_out_frames = getattr(self.config, 'fade_out_frames', 8)  # Longer fade-out for smoother disappearance
        
        if target_state > current_state:  # Fading in
            self.fade_out_counter[landmark_key] = 0  # Reset fade-out counter
            self.fade_in_counter[landmark_key] += 1
            
            # Calculate fade-in progress
            fade_in_progress = min(1.0, self.fade_in_counter[landmark_key] / fade_in_frames)
            
            # Move toward target state (with smoothed interpolation)
            # Use ease-in function: progress^2 for smoother start
            new_state = current_state + (target_state - current_state) * (fade_in_progress ** 2)
        else:  # Fading out
            self.fade_in_counter[landmark_key] = 0  # Reset fade-in counter
            self.fade_out_counter[landmark_key] += 1
            
            # Calculate fade-out progress
            fade_out_progress = min(1.0, self.fade_out_counter[landmark_key] / fade_out_frames)
            
            # Move toward target state (with smoothed interpolation)
            # Use ease-out function: 1-(1-progress)^2 for smoother end
            ease_out = 1.0 - (1.0 - fade_out_progress) ** 2
            new_state = current_state + (target_state - current_state) * ease_out
        
        # Update state
        self.landmark_visibility_state[landmark_key] = new_state
        
        return new_state
    
    def _draw_reference_lines(self, image, landmarks, virtual_landmarks):
        """
        Draw reference lines for alignment assessment.
        
        Args:
            image: Image to draw on
            landmarks: Dictionary of MediaPipe pose landmarks
            virtual_landmarks: Dictionary of virtual landmarks
        """
        if not landmarks or not virtual_landmarks or not self.config.show_reference_lines:
            return
        
        height, width = image.shape[:2]
        exercise_config = self._get_exercise_config()
        reference_lines = exercise_config.get('reference_lines', [])
        
        # Draw vertical plumb line
        if 'vertical' in reference_lines and 'PLUMB_LINE_TOP' in virtual_landmarks and 'PLUMB_LINE_BOTTOM' in virtual_landmarks:
            top = virtual_landmarks['PLUMB_LINE_TOP']
            bottom = virtual_landmarks['PLUMB_LINE_BOTTOM']
            
            pt1 = (int(top[0] * width), int(top[1] * height))
            pt2 = (int(bottom[0] * width), int(bottom[1] * height))
            
            if self.config.line_style == "outlined":
                self._draw_outlined_line(image, pt1, pt2, self.colors['reference'], 3)
            else:
                cv2.line(
                    image,
                    pt1,
                    pt2,
                    self.colors['reference'],
                    3,  # Increased thickness
                    cv2.LINE_AA
                )
        
        # Draw extended hip line
        if 'hip_line' in reference_lines and 'HIP_LINE_LEFT' in virtual_landmarks and 'HIP_LINE_RIGHT' in virtual_landmarks:
            left = virtual_landmarks['HIP_LINE_LEFT']
            right = virtual_landmarks['HIP_LINE_RIGHT']
            
            pt1 = (int(left[0] * width), int(left[1] * height))
            pt2 = (int(right[0] * width), int(right[1] * height))
            
            if self.config.line_style == "outlined":
                self._draw_outlined_line(image, pt1, pt2, self.colors['hip_line'], 3)
            else:
                cv2.line(
                    image,
                    pt1,
                    pt2,
                    self.colors['hip_line'],
                    3,  # Increased thickness
                    cv2.LINE_AA
                )
        
        # Draw exercise-specific reference lines based on the current exercise
        if self.config.exercise_type == 'squat':
            self._draw_squat_reference_lines(image, landmarks, virtual_landmarks, height, width)
        elif self.config.exercise_type == 'pushup':
            self._draw_pushup_reference_lines(image, landmarks, virtual_landmarks, height, width)
            
    def _draw_squat_reference_lines(self, image, landmarks, virtual_landmarks, height, width):
        """Draw squat-specific reference lines."""
        exercise_config = self._get_exercise_config()
        reference_lines = exercise_config.get('reference_lines', [])
        
        # Draw knee projection lines (for knees over toes analysis)
        if 'knee_projection' in reference_lines and 'LEFT_KNEE_PROJECTION' in virtual_landmarks:
            left_knee = landmarks.get(self.mp_pose.PoseLandmark.LEFT_KNEE.value)
            left_proj = virtual_landmarks['LEFT_KNEE_PROJECTION']
            
            if left_knee:
                pt1 = (int(left_knee.x * width), int(left_knee.y * height))
                pt2 = (int(left_proj.x * width), int(left_proj.y * height))
                
                if self.config.line_style == "outlined":
                    self._draw_outlined_line(image, pt1, pt2, self.colors['knee_projection'], 3)
                else:
                    cv2.line(image, pt1, pt2, self.colors['knee_projection'], 3, cv2.LINE_AA)
            
        if 'knee_projection' in reference_lines and 'RIGHT_KNEE_PROJECTION' in virtual_landmarks:
            right_knee = landmarks.get(self.mp_pose.PoseLandmark.RIGHT_KNEE.value)
            right_proj = virtual_landmarks['RIGHT_KNEE_PROJECTION']
            
            if right_knee:
                pt1 = (int(right_knee.x * width), int(right_knee.y * height))
                pt2 = (int(right_proj.x * width), int(right_proj.y * height))
                
                if self.config.line_style == "outlined":
                    self._draw_outlined_line(image, pt1, pt2, self.colors['knee_projection'], 3)
                else:
                    cv2.line(image, pt1, pt2, self.colors['knee_projection'], 3, cv2.LINE_AA)
                    
        # Draw target squat depth line if available
        if 'SQUAT_DEPTH' in virtual_landmarks:
            depth_point = virtual_landmarks['SQUAT_DEPTH']
            left_hip = landmarks.get(self.mp_pose.PoseLandmark.LEFT_HIP.value)
            right_hip = landmarks.get(self.mp_pose.PoseLandmark.RIGHT_HIP.value)
            
            if left_hip and right_hip:
                pt1 = (int(left_hip.x * width), int(depth_point.y * height))
                pt2 = (int(right_hip.x * width), int(depth_point.y * height))
                
                if self.config.line_style == "outlined":
                    self._draw_outlined_line(image, pt1, pt2, (66, 245, 200), 3)  # Light blue-green
                else:
                    cv2.line(image, pt1, pt2, (66, 245, 200), 3, cv2.LINE_AA)
                    
    def _draw_pushup_reference_lines(self, image, landmarks, virtual_landmarks, height, width):
        """Draw pushup-specific reference lines."""
        exercise_config = self._get_exercise_config()
        reference_lines = exercise_config.get('reference_lines', [])
        
        # Draw extended spine line for pushups if available
        if 'EXTENDED_SPINE_LINE' in virtual_landmarks:
            mid_shoulder = virtual_landmarks.get('MID_SPINE')
            ext_point = virtual_landmarks['EXTENDED_SPINE_LINE']
            
            if mid_shoulder:
                pt1 = (int(mid_shoulder.x * width), int(mid_shoulder.y * height))
                pt2 = (int(ext_point[0] * width), int(ext_point[1] * height))
                
                if self.config.line_style == "outlined":
                    self._draw_outlined_line(image, pt1, pt2, self.colors['spine_line'], 3)
                else:
                    cv2.line(image, pt1, pt2, self.colors['spine_line'], 3, cv2.LINE_AA)
    
    def _draw_angles(self, image, landmarks, angles: Dict):
        """
        Draw angle measurements at key joints with visual arcs.
        
        Args:
            image: Image to draw on
            landmarks: Dictionary of MediaPipe pose landmarks
            angles: Dictionary of joint angles
        """
        if not landmarks or not angles or not self.config.show_angles:
            return
            
        height, width = image.shape[:2]
        exercise_config = self._get_exercise_config()
        key_joints = exercise_config.get('key_joints', [])
        angle_thresholds = exercise_config.get('angle_thresholds', {})
        
        for joint_name, angle_data in angles.items():
            if joint_name not in key_joints:
                continue
                
            angle = angle_data.get('angle')
            confidence = angle_data.get('confidence', 0)
            
            if angle is None or confidence < self.config.confidence_threshold:
                continue
                
            # Get the landmark position
            joint_type = joint_name.split('_')[1].lower()
            pose_landmark_name = f"{joint_name.split('_')[0]}_{joint_name.split('_')[1]}"
            try:
                landmark_idx = getattr(self.mp_pose.PoseLandmark, pose_landmark_name, None)
                
                if landmark_idx is None or landmark_idx.value not in landmarks:
                    continue
                    
                landmark = landmarks[landmark_idx.value]
                pos = (int(landmark.x * width), int(landmark.y * height))
                
                # Get angle thresholds for this joint type
                min_angle, max_angle = angle_thresholds.get(joint_type, (0, 180))
                in_range = min_angle <= angle <= max_angle
                color = self.colors['feedback_good'] if in_range else self.colors['feedback_bad']
                
                # Draw angle arc with thicker line
                radius = 40  # Increased radius for better visibility
                start_angle = 0
                end_angle = angle
                
                # Create arc overlay for filled semi-transparent arc
                overlay = image.copy()
                
                # Draw filled semi-transparent arc
                cv2.ellipse(
                    overlay,
                    pos,
                    (radius, radius),
                    0,
                    0,
                    angle,
                    color,
                    -1,  # Filled
                    cv2.LINE_AA
                )
                
                # Blend the overlay with the original image
                alpha = 0.4
                cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
                
                # Draw arc outline with thicker line
                cv2.ellipse(
                    image,
                    pos,
                    (radius, radius),
                    0,
                    0,
                    angle,
                    (0, 0, 0),
                    3,  # Increased thickness for outline
                    cv2.LINE_AA
                )
                
                # Draw arc with thicker line
                cv2.ellipse(
                    image,
                    pos,
                    (radius, radius),
                    0,
                    0,
                    angle,
                    color,
                    2,  # Increased thickness
                    cv2.LINE_AA
                )
                
                # Position text slightly away from the arc
                text_angle = angle / 2 if angle < 180 else (angle + 360) / 2
                text_radius = radius + 25  # Increased distance for better readability
                text_x = int(pos[0] + text_radius * np.cos(text_angle * np.pi / 180))
                text_y = int(pos[1] - text_radius * np.sin(text_angle * np.pi / 180))
                
                # Format angle value with 1 decimal place
                angle_text = f"{angle:.1f}°"
                
                # Draw text with shadow for better visibility
                self._draw_text_with_shadow(
                    image,
                    angle_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,  # Increased font scale
                    color,
                    2  # Increased thickness
                )
            except Exception as e:
                logging.warning(f"Error drawing angle for {joint_name}: {e}")
    
    def _draw_motion_trails(self, image, landmarks):
        """
        Draw motion trails for tracking movement patterns.
        
        Args:
            image: Image to draw on
            landmarks: Dictionary of MediaPipe pose landmarks
        """
        if not landmarks or not self.config.show_motion_trails:
            return
        
        height, width = image.shape[:2]
        
        # Get key landmark indices to track
        tracked_landmarks = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            self.mp_pose.PoseLandmark.LEFT_ELBOW.value,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW.value,
            self.mp_pose.PoseLandmark.LEFT_WRIST.value,
            self.mp_pose.PoseLandmark.RIGHT_WRIST.value,
            self.mp_pose.PoseLandmark.LEFT_HIP.value,
            self.mp_pose.PoseLandmark.RIGHT_HIP.value,
            self.mp_pose.PoseLandmark.LEFT_KNEE.value,
            self.mp_pose.PoseLandmark.RIGHT_KNEE.value,
            self.mp_pose.PoseLandmark.LEFT_ANKLE.value,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE.value
        ]
        
        # Ensure we have a motion trail list
        if not hasattr(self, 'motion_trail'):
            self.motion_trail = []
        
        # Update motion trail
        current_positions = {}
        for idx in tracked_landmarks:
            if idx in landmarks:
                current_positions[idx] = (
                    int(landmarks[idx].x * width),
                    int(landmarks[idx].y * height)
                )
        
        if current_positions:
            self.motion_trail.append(current_positions)
            
            # Keep only recent positions
            max_trail_length = getattr(self, 'max_trail_length', 5)  # Reduced from 10 to 5
            if len(self.motion_trail) > max_trail_length:
                self.motion_trail.pop(0)
            
            # Draw trails with improved fading and gaps
            for i in range(len(self.motion_trail) - 1):
                # Calculate age factor (0 = newest, 1 = oldest)
                age_factor = i / (len(self.motion_trail) - 1) if len(self.motion_trail) > 1 else 0
                
                # Make trails fade out more quickly - use squared ratio for faster falloff
                # Calculate opacity - newer trails are more opaque
                alpha = 0.4 * ((1 - age_factor) ** 2)  # Square for faster falloff
                
                # Adjust thickness - newer trails are thicker
                thickness = max(1, int(2.5 * (1 - age_factor)))
                
                # Choose trail color (recent vs old)
                trail_color = self.colors['motion_trail'] if age_factor < 0.5 else self.colors['motion_trail_old']
                
                # Calculate gap size (larger for older trails)
                gap_size = int(2 + age_factor * 4)  # from 2px to 6px gap
                
                for idx in tracked_landmarks:
                    if idx in self.motion_trail[i] and idx in self.motion_trail[i + 1]:
                        start_point = self.motion_trail[i][idx]
                        end_point = self.motion_trail[i + 1][idx]
                        
                        # Calculate total line length
                        dx = end_point[0] - start_point[0]
                        dy = end_point[1] - start_point[1]
                        line_length = np.sqrt(dx**2 + dy**2)
                        
                        # Only draw if line is long enough for a gap
                        if line_length > gap_size:
                            # Calculate gap position (normalized vector)
                            if line_length > 0:
                                nx = dx / line_length
                                ny = dy / line_length
                                
                                # Create points with gap
                                gap_start = (
                                    int(end_point[0] - nx * gap_size),
                                    int(end_point[1] - ny * gap_size)
                                )
                                
                                # Draw the line with a gap near the end
                                cv2.line(
                                    image,
                                    start_point,
                                    gap_start,
                                    trail_color,
                                    thickness,
                                    cv2.LINE_AA
                                )
                        else:
                            # For very short lines, draw without gap
                            cv2.line(
                                image,
                                start_point,
                                end_point,
                                trail_color,
                                thickness,
                                cv2.LINE_AA
                            )
    
    def _draw_spine(self, image, landmarks, virtual_landmarks, height, width):
        """
        Draw the multi-segment spine with enhanced visualization.
        
        Args:
            image: Image to draw on
            landmarks: Dictionary of MediaPipe pose landmarks
            virtual_landmarks: Dictionary of virtual landmarks
            height: Image height
            width: Image width
        """
        if not landmarks:
            return
            
        # Define spine segments with proper MediaPipe pose landmark references
        spine_segments = [
            ('NECK', 'CERVICAL_SPINE'),
            ('CERVICAL_SPINE', 'THORACIC_SPINE_UPPER'),
            ('THORACIC_SPINE_UPPER', 'THORACIC_SPINE_LOWER'),
            ('THORACIC_SPINE_LOWER', 'LUMBAR_SPINE_UPPER'),
            ('LUMBAR_SPINE_UPPER', 'LUMBAR_SPINE_LOWER'),
            ('LUMBAR_SPINE_LOWER', 'MID_SPINE')
        ]
        
        # Draw central spine with special styling
        for start_name, end_name in spine_segments:
            if start_name not in virtual_landmarks or end_name not in virtual_landmarks:
                continue
                
            start_point = virtual_landmarks[start_name]
            end_point = virtual_landmarks[end_name]
            
            # Get confidence based on nearby real landmarks
            confidence = min(
                self._get_confidence(landmarks.get(self.mp_pose.PoseLandmark.NOSE.value, 1.0)),
                self._get_confidence(landmarks.get(self.mp_pose.PoseLandmark.LEFT_SHOULDER.value, 1.0)),
                self._get_confidence(landmarks.get(self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value, 1.0)),
                self._get_confidence(landmarks.get(self.mp_pose.PoseLandmark.LEFT_HIP.value, 1.0)),
                self._get_confidence(landmarks.get(self.mp_pose.PoseLandmark.RIGHT_HIP.value, 1.0))
            )
            
            # Skip if very low confidence
            if confidence < self.config.min_detection_threshold:
                continue
                
            # Calculate points positions
            x1, y1 = int(start_point.x * width), int(start_point.y * height)
            x2, y2 = int(end_point.x * width), int(end_point.y * height)
            
            # Use a distinct color for spine
            color = self.colors['spine_line']  
            
            # Calculate line thickness based on confidence
            # Central spine is thicker for emphasis
            thickness_multiplier = 0.6 + min(confidence, 1.0) * 0.6  # Scale from 0.6 to 1.2 based on visibility
            thickness = max(1, int(4 * thickness_multiplier))  # Base thickness of 4px
            
            # Draw based on line style and confidence
            if confidence > 0.85:  # High confidence spine segments
                if self.config.line_style == 'outlined':
                    # Double outlined spine for better visibility
                    # First draw a thicker black outline
                    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 0), thickness + 4, cv2.LINE_AA)
                    # Then draw a thinner white middle layer
                    cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), thickness + 2, cv2.LINE_AA)
                    # Finally draw the colored line
                    cv2.line(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
                else:
                    # For solid lines, just draw with extra thickness
                    cv2.line(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
            elif confidence > 0.7:  # Medium confidence
                if self.config.line_style == 'outlined':
                    self._draw_outlined_line(image, (x1, y1), (x2, y2), color, thickness)
                else:
                    cv2.line(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
            else:  # Low confidence
                # Use dashed lines for low confidence
                dash_length = max(5, int(10 * confidence))
                space_length = max(2, int(5 * (1 - confidence)))
                self._draw_dashed_line(image, (x1, y1), (x2, y2), color, thickness, dash_length, space_length)
                
        # Draw shoulder line (connecting shoulders)
        left_shoulder_idx = self.mp_pose.PoseLandmark.LEFT_SHOULDER.value
        right_shoulder_idx = self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        
        if (left_shoulder_idx in landmarks and right_shoulder_idx in landmarks):
            left_shoulder = landmarks[left_shoulder_idx]
            right_shoulder = landmarks[right_shoulder_idx]
            
            left_vis = getattr(left_shoulder, 'visibility', 1.0)
            right_vis = getattr(right_shoulder, 'visibility', 1.0)
            
            if left_vis >= self.config.confidence_threshold and right_vis >= self.config.confidence_threshold:
                left_pos = (int(left_shoulder.x * width), int(left_shoulder.y * height))
                right_pos = (int(right_shoulder.x * width), int(right_shoulder.y * height))
                
                # Use shoulder line color
                color = self.colors['shoulder_line']
                
                # Calculate line thickness
                min_vis = min(left_vis, right_vis)
                thickness = max(1, int(4 * (0.6 + min_vis * 0.5)))
                
                # Draw shoulder line based on style and confidence
                if self.config.line_style == 'outlined':
                    self._draw_outlined_line(image, left_pos, right_pos, color, thickness)
                else:
                    cv2.line(image, left_pos, right_pos, color, thickness, cv2.LINE_AA)
                    
        # Draw hip line (connecting hips)
        left_hip_idx = self.mp_pose.PoseLandmark.LEFT_HIP.value
        right_hip_idx = self.mp_pose.PoseLandmark.RIGHT_HIP.value
        
        if (left_hip_idx in landmarks and right_hip_idx in landmarks):
            left_hip = landmarks[left_hip_idx]
            right_hip = landmarks[right_hip_idx]
            
            left_vis = getattr(left_hip, 'visibility', 1.0)
            right_vis = getattr(right_hip, 'visibility', 1.0)
            
            if left_vis >= self.config.confidence_threshold and right_vis >= self.config.confidence_threshold:
                left_pos = (int(left_hip.x * width), int(left_hip.y * height))
                right_pos = (int(right_hip.x * width), int(right_hip.y * height))
                
                # Use hip line color
                color = self.colors['hip_line']
                
                # Calculate line thickness
                min_vis = min(left_vis, right_vis)
                thickness = max(1, int(4 * (0.6 + min_vis * 0.5)))
                
                # Draw hip line based on style and confidence
                if self.config.line_style == 'outlined':
                    self._draw_outlined_line(image, left_pos, right_pos, color, thickness)
                else:
                    cv2.line(image, left_pos, right_pos, color, thickness, cv2.LINE_AA)
    
    def _draw_limbs(self, image, landmarks, height, width):
        """
        Draw limbs connecting joint landmarks with improved visibility features.
        
        Args:
            image: Image to draw on
            landmarks: Dictionary of MediaPipe pose landmarks
            height: Image height
            width: Image width
        """
        if not landmarks:
            return
            
        # Define limb connections with proper MediaPipe pose landmark references
        limbs = [
            # Arms
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER.value, self.mp_pose.PoseLandmark.LEFT_ELBOW.value),
            (self.mp_pose.PoseLandmark.LEFT_ELBOW.value, self.mp_pose.PoseLandmark.LEFT_WRIST.value),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value, self.mp_pose.PoseLandmark.RIGHT_ELBOW.value),
            (self.mp_pose.PoseLandmark.RIGHT_ELBOW.value, self.mp_pose.PoseLandmark.RIGHT_WRIST.value),
            # Legs
            (self.mp_pose.PoseLandmark.LEFT_HIP.value, self.mp_pose.PoseLandmark.LEFT_KNEE.value),
            (self.mp_pose.PoseLandmark.LEFT_KNEE.value, self.mp_pose.PoseLandmark.LEFT_ANKLE.value),
            (self.mp_pose.PoseLandmark.RIGHT_HIP.value, self.mp_pose.PoseLandmark.RIGHT_KNEE.value),
            (self.mp_pose.PoseLandmark.RIGHT_KNEE.value, self.mp_pose.PoseLandmark.RIGHT_ANKLE.value),
            # Shoulder girdle
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER.value, self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
            # Pelvic girdle
            (self.mp_pose.PoseLandmark.LEFT_HIP.value, self.mp_pose.PoseLandmark.RIGHT_HIP.value)
        ]
        
        # Draw each limb connection
        for start_idx, end_idx in limbs:
            if start_idx not in landmarks or end_idx not in landmarks:
                continue
                
            start_point = landmarks[start_idx]
            end_point = landmarks[end_idx]
            
            # Get visibility from landmarks (if available) or default to 1.0
            start_vis = getattr(start_point, 'visibility', 1.0)
            end_vis = getattr(end_point, 'visibility', 1.0)
            
            # Calculate limb key for tracking
            limb_key = f"limb_{start_idx}_{end_idx}"
            
            # Update exponential weighted moving average of confidence
            min_vis = min(start_vis, end_vis)
            if limb_key not in self.landmark_confidence_ewma:
                self.landmark_confidence_ewma[limb_key] = min_vis
            else:
                # Apply EWMA formula: new_value = alpha * current + (1-alpha) * previous
                self.landmark_confidence_ewma[limb_key] = (
                    self.ewma_alpha * min_vis + 
                    (1 - self.ewma_alpha) * self.landmark_confidence_ewma[limb_key]
                )
            
            # Use the smoothed confidence
            smoothed_confidence = self.landmark_confidence_ewma[limb_key]
            
            # Apply hysteresis and fade effects
            visibility_state = self._get_landmark_visibility_state(limb_key, smoothed_confidence)
            
            # Skip if completely invisible
            if visibility_state <= 0.01:
                continue
                
            # Calculate points positions
            x1, y1 = int(start_point.x * width), int(start_point.y * height)
            x2, y2 = int(end_point.x * width), int(end_point.y * height)
            
            # Determine line color based on side (left/right)
            try:
                if 'LEFT' in self.mp_pose.PoseLandmark(start_idx).name:
                    base_color = self.colors['left_limb']
                else:
                    base_color = self.colors['right_limb']
            except:
                # Fallback to default skeleton color
                base_color = self.colors['skeleton']
            
            # Adjust color alpha based on visibility state for a smooth fade effect
            if len(base_color) == 3:  # RGB color
                # Convert RGB to RGBA with alpha based on visibility state
                alpha = int(255 * visibility_state)
                color = (*base_color, alpha)
            else:
                # Already has alpha
                alpha = int(base_color[3] * visibility_state)
                color = (*base_color[:3], alpha)
                
            # Calculate line thickness based on confidence and visibility state
            # Thicker for high confidence, thinner for lower confidence
            thickness_multiplier = 0.5 + min(smoothed_confidence, 1.0) * 0.7 * visibility_state
            thickness = max(1, int(4 * thickness_multiplier))  # Base thickness of 4px
                
            # Use the line style specified in config with confidence-aware rendering
            if self.config.line_style == 'solid':
                # Create a new image for the semi-transparent line
                if visibility_state < 1.0:
                    overlay = image.copy()
                    cv2.line(overlay, (x1, y1), (x2, y2), base_color, thickness, cv2.LINE_AA)
                    # Apply with alpha blending based on visibility state
                    cv2.addWeighted(overlay, visibility_state, image, 1 - visibility_state, 0, image)
                else:
                    cv2.line(image, (x1, y1), (x2, y2), base_color, thickness, cv2.LINE_AA)
            elif self.config.line_style == 'outlined':
                # For lower confidence, use simpler lines instead of outlines
                if visibility_state > 0.8:
                    self._draw_outlined_line(image, (x1, y1), (x2, y2), base_color, thickness)
                else:
                    # Create a new image for the semi-transparent line
                    overlay = image.copy()
                    cv2.line(overlay, (x1, y1), (x2, y2), base_color, thickness, cv2.LINE_AA)
                    # Apply with alpha blending based on visibility state
                    cv2.addWeighted(overlay, visibility_state, image, 1 - visibility_state, 0, image)
            elif self.config.line_style == 'dashed':
                # Calculate dash pattern based on point visibility and line length
                dash_length = max(5, int(10 * smoothed_confidence))  # Longer dashes for higher confidence
                space_length = max(2, int(5 * (1 - smoothed_confidence)))  # Shorter spaces for higher confidence
                
                # Create a new image for the semi-transparent dashed line
                if visibility_state < 1.0:
                    overlay = image.copy()
                    self._draw_dashed_line(overlay, (x1, y1), (x2, y2), base_color, thickness, dash_length, space_length)
                    # Apply with alpha blending based on visibility state
                    cv2.addWeighted(overlay, visibility_state, image, 1 - visibility_state, 0, image)
                else:
                    self._draw_dashed_line(image, (x1, y1), (x2, y2), base_color, thickness, dash_length, space_length)
    
    def _draw_joints(self, image, landmarks, height, width):
        """
        Draw joints with hierarchical sizing based on importance.
        
        Args:
            image: Image to draw on
            landmarks: Dictionary of MediaPipe pose landmarks
            height: Image height
            width: Image width
        """
        # Define primary joints (larger)
        primary_joints = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            self.mp_pose.PoseLandmark.LEFT_HIP.value,
            self.mp_pose.PoseLandmark.RIGHT_HIP.value,
            self.mp_pose.PoseLandmark.LEFT_KNEE.value,
            self.mp_pose.PoseLandmark.RIGHT_KNEE.value
        ]
        
        # Draw each joint
        for idx, landmark in landmarks.items():
            raw_confidence = self._get_confidence(landmark)
            
            # Update exponential weighted moving average of confidence
            joint_key = f"joint_{idx}"
            if joint_key not in self.landmark_confidence_ewma:
                self.landmark_confidence_ewma[joint_key] = raw_confidence
            else:
                # Apply EWMA formula: new_value = alpha * current + (1-alpha) * previous
                self.landmark_confidence_ewma[joint_key] = (
                    self.ewma_alpha * raw_confidence + 
                    (1 - self.ewma_alpha) * self.landmark_confidence_ewma[joint_key]
                )
            
            # Use the smoothed confidence
            confidence = self.landmark_confidence_ewma[joint_key]
            
            # Apply hysteresis and fade effects
            visibility_state = self._get_landmark_visibility_state(joint_key, confidence)
            
            # Skip if completely invisible
            if visibility_state <= 0.01:
                continue
                
            # Determine joint size based on importance and confidence
            base_radius = 8 if idx in primary_joints else 5  # Increased sizes
            
            # Scale radius by confidence and visibility state
            scaled_radius = base_radius * (0.7 + (confidence * 0.3)) * visibility_state
            radius = max(2, int(scaled_radius))
            
            # Get joint position
            joint_pos = (int(landmark.x * width), int(landmark.y * height))
            
            # Determine color based on visibility state for smooth transitions
            if isinstance(self.colors['joints'], tuple):
                if len(self.colors['joints']) == 3:
                    # Convert RGB to RGBA with alpha based on visibility state
                    alpha = int(255 * visibility_state)
                    color = (*self.colors['joints'], alpha)
                else:
                    # Already has alpha
                    alpha = int(self.colors['joints'][3] * visibility_state)
                    color = (*self.colors['joints'][:3], alpha)
            else:
                color = self.colors['joints']
            
            # Create a new image for the semi-transparent joint
            if visibility_state < 1.0:
                overlay = image.copy()
                
                # Draw white outline with alpha
                outline_alpha = min(255, int(255 * visibility_state))
                outline_color = (255, 255, 255, outline_alpha)
                cv2.circle(
                    overlay,
                    joint_pos,
                    radius + 2,  # Slightly larger for outline
                    outline_color,
                    2,
                    cv2.LINE_AA
                )
                
                # Draw joint circle with alpha
                cv2.circle(
                    overlay,
                    joint_pos,
                    radius,
                    color,
                    -1,
                    cv2.LINE_AA
                )
                
                # Add an inner highlight to primary joints for better visibility
                if idx in primary_joints and confidence >= self.config.confidence_threshold * 0.7:
                    # Add highlight dot in center with fade based on confidence
                    highlight_radius = max(1, radius // 3)
                    highlight_alpha = min(255, int(255 * confidence * visibility_state))
                    cv2.circle(
                        overlay,
                        joint_pos,
                        highlight_radius,
                        (255, 255, 255, highlight_alpha),
                        -1,
                        cv2.LINE_AA
                    )
                
                # Apply with alpha blending based on visibility state
                cv2.addWeighted(overlay, visibility_state, image, 1 - visibility_state, 0, image)
            else:
                # Draw white outline for all joints
                cv2.circle(
                    image,
                    joint_pos,
                    radius + 2,  # Slightly larger for outline
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
                
                # Draw joint circle
                cv2.circle(
                    image,
                    joint_pos,
                    radius,
                    self.colors['joints'],
                    -1,
                    cv2.LINE_AA
                )
                
                # Add an inner highlight to primary joints for better visibility
                if idx in primary_joints and confidence >= self.config.confidence_threshold * 0.7:
                    # Add highlight dot in center
                    highlight_radius = max(1, radius // 3)
                    cv2.circle(
                        image,
                        joint_pos,
                        highlight_radius,
                        (255, 255, 255),
                        -1,
                        cv2.LINE_AA
                    )
    
    def _draw_enhanced_skeleton(self, image, landmarks, virtual_landmarks):
        """
        Draw an anatomically improved skeleton.
        
        Args:
            image: Image to draw on
            landmarks: Dictionary of MediaPipe pose landmarks
            virtual_landmarks: Dictionary of virtual landmarks
        """
        if not landmarks:
            return
        
        height, width = image.shape[:2]
        
        # Draw multi-segment spine
        self._draw_spine(image, landmarks, virtual_landmarks, height, width)
        
        # Draw limbs with proper biomechanical connections
        self._draw_limbs(image, landmarks, height, width)
        
        # Draw joints with hierarchical sizing
        self._draw_joints(image, landmarks, height, width)
    
    def _draw_outlined_line(self, image, pt1, pt2, color, thickness=2):
        """
        Draw a line with a contrasting outline for better visibility against any background.
        
        Args:
            image: Image to draw on
            pt1: Start point (x, y)
            pt2: End point (x, y)
            color: Line color in BGR format
            thickness: Line thickness
            
        Returns:
            None, modifies image in place
        """
        # Calculate line direction vector
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        dist = max(1, math.sqrt(dx*dx + dy*dy))
        
        # Normalize direction vector
        dx, dy = dx/dist, dy/dist
        
        # Draw darker background shadow (reduced offset)
        shadow_offset = 1
        shadow_pt1 = (pt1[0] + shadow_offset, pt1[1] + shadow_offset)
        shadow_pt2 = (pt2[0] + shadow_offset, pt2[1] + shadow_offset)
        cv2.line(image, shadow_pt1, shadow_pt2, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        
        # Draw black outline (thinner than before)
        for offset in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            outline_pt1 = (pt1[0] + offset[0], pt1[1] + offset[1])
            outline_pt2 = (pt2[0] + offset[0], pt2[1] + offset[1])
            cv2.line(image, outline_pt1, outline_pt2, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        
        # Draw the main colored line on top
        cv2.line(image, pt1, pt2, color, thickness, cv2.LINE_AA)
        
        # Only add highlight for thicker lines to reduce visual noise
        if thickness >= 2:
            # Compute perpendicular vector for the highlight offset
            perp_dx, perp_dy = -dy, dx
            highlight_scale = 0.2  # Controls how far the highlight is from the center
            
            # Shift highlight slightly toward light source (top-left convention)
            highlight_pt1 = (int(pt1[0] - perp_dx * highlight_scale), int(pt1[1] - perp_dy * highlight_scale))
            highlight_pt2 = (int(pt2[0] - perp_dx * highlight_scale), int(pt2[1] - perp_dy * highlight_scale))
            
            # Draw thin white highlight
            cv2.line(image, highlight_pt1, highlight_pt2, (255, 255, 255), max(1, thickness//3), cv2.LINE_AA)
    
    def _draw_text_with_shadow(self, image, text, position, font_face, font_scale, color, thickness=1):
        """
        Draw text with shadow and outline for better visibility against any background.
        
        Args:
            image: Image to draw on
            text: Text string to draw
            position: Position (x, y) for text
            font_face: OpenCV font constant
            font_scale: Font scale factor
            color: Text color in BGR format
            thickness: Text thickness
            
        Returns:
            None, modifies image in place
        """
        # Get text size for better positioning
        text_size, baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
        
        # Draw multiple shadows for depth effect
        shadow_offsets = [(2, 2), (2, 1), (1, 2)]
        for offset in shadow_offsets:
            shadow_position = (position[0] + offset[0], position[1] + offset[1])
            cv2.putText(
                image, 
                text, 
                shadow_position, 
                font_face, 
                font_scale, 
                (0, 0, 0), 
                thickness + 1, 
                cv2.LINE_AA
            )
        
        # Draw black outline around text for better readability
        for offset in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            outline_position = (position[0] + offset[0], position[1] + offset[1])
            cv2.putText(
                image, 
                text, 
                outline_position, 
                font_face, 
                font_scale, 
                (0, 0, 0), 
                thickness, 
                cv2.LINE_AA
            )
        
        # Draw main text
        cv2.putText(
            image, 
            text, 
            position, 
            font_face, 
            font_scale, 
            color, 
            thickness, 
            cv2.LINE_AA
        )
        
        # Add subtle highlight for 3D effect
        if thickness > 1:
            highlight_position = (position[0] - 1, position[1] - 1)
            # Use a muted white for highlight to avoid overwhelming the text
            highlight_color = (min(255, color[0] + 80), min(255, color[1] + 80), min(255, color[2] + 80))
            cv2.putText(
                image, 
                text, 
                highlight_position, 
                font_face, 
                font_scale, 
                highlight_color, 
                max(1, thickness - 1), 
                cv2.LINE_AA
            )
    
    def _smooth_landmarks(self, current_landmarks, previous_landmarks):
        """
        Apply temporal smoothing to reduce flickering.
        
        Args:
            current_landmarks: Current frame landmarks
            previous_landmarks: Previous frame landmarks
            
        Returns:
            Smoothed landmarks
        """
        # If no previous landmarks or smoothing disabled, return current landmarks
        if previous_landmarks is None or self.config.smoothing_factor <= 0.01:
            return current_landmarks
        
        # Determine actual smoothing factor based on configuration
        smoothing_factor = self.config.smoothing_factor
        
        # Use different smoothing factors for different visualization modes
        if self.config.mode == VisualizationMode.ANALYSIS:
            # More smoothing for analysis mode
            smoothing_factor = min(0.8, smoothing_factor * 1.2)
        elif self.config.mode == VisualizationMode.REALTIME:
            # Less smoothing for realtime mode to prioritize responsiveness
            # But still provide stable visualization
            smoothing_factor = max(0.3, smoothing_factor * 0.8)
            
        # Further reduce smoothing if response_priority is enabled
        if getattr(self.config, 'response_priority', False):
            smoothing_factor = max(0.2, smoothing_factor * 0.6)  # Reduce to 60% but no less than 0.2
            
        smoothed = {}
        for idx, landmark in current_landmarks.items():
            # Skip very low confidence landmarks
            confidence = getattr(landmark, 'visibility', 1.0)
            
            # Use dynamic threshold based on visualization mode
            min_threshold = self.config.min_detection_threshold
            if self.config.mode == VisualizationMode.ANALYSIS:
                # Be more lenient in analysis mode
                min_threshold *= 0.8
                
            if confidence < min_threshold:
                continue
                
            if idx in previous_landmarks:
                # Apply exponential smoothing
                prev = previous_landmarks[idx]
                
                # Dynamic smoothing factor based on confidence
                # Higher confidence = less smoothing (more responsive)
                # Lower confidence = more smoothing (more stable)
                dynamic_factor = smoothing_factor * 0.8  # Fixed value for more predictable response
                
                # Create a new point with the same structure as the input landmark
                # This preserves the original landmark class structure
                try:
                    # Try to create a new instance of the same class
                    smoothed_landmark = type(landmark)()
                    
                    # Apply exponential smoothing to coordinates
                    smoothed_landmark.x = prev.x * dynamic_factor + landmark.x * (1 - dynamic_factor)
                    smoothed_landmark.y = prev.y * dynamic_factor + landmark.y * (1 - dynamic_factor)
                    
                    # Handle z coordinate if it exists
                    if hasattr(landmark, 'z') and hasattr(prev, 'z'):
                        smoothed_landmark.z = prev.z * dynamic_factor + landmark.z * (1 - dynamic_factor)
                    elif hasattr(landmark, 'z'):
                        smoothed_landmark.z = landmark.z
                        
                    # Apply minimal smoothing to visibility for smoother transitions
                    if hasattr(landmark, 'visibility') and hasattr(prev, 'visibility'):
                        vis_smooth_factor = min(0.3, dynamic_factor * 0.5)  # Use much lower factor for visibility
                        smoothed_landmark.visibility = prev.visibility * vis_smooth_factor + landmark.visibility * (1 - vis_smooth_factor)
                    elif hasattr(landmark, 'visibility'):
                        smoothed_landmark.visibility = landmark.visibility
                        
                    smoothed[idx] = smoothed_landmark
                except Exception as e:
                    # If we can't create a new instance, fall back to our custom Point class
                    logging.debug(f"Falling back to custom Point class: {e}")
                    class Point:
                        def __init__(self):
                            self.x = 0
                            self.y = 0
                            self.z = 0
                            self.visibility = 0
                    
                    smoothed_landmark = Point()
                    smoothed_landmark.x = prev.x * dynamic_factor + landmark.x * (1 - dynamic_factor)
                    smoothed_landmark.y = prev.y * dynamic_factor + landmark.y * (1 - dynamic_factor)
                    
                    if hasattr(landmark, 'z') and hasattr(prev, 'z'):
                        smoothed_landmark.z = prev.z * dynamic_factor + landmark.z * (1 - dynamic_factor)
                    elif hasattr(landmark, 'z'):
                        smoothed_landmark.z = landmark.z
                        
                    if hasattr(landmark, 'visibility'):
                        smoothed_landmark.visibility = landmark.visibility
                        
                    smoothed[idx] = smoothed_landmark
            else:
                # If no previous data, use current landmark as is
                smoothed[idx] = landmark
                
        return smoothed
    
    def render(self, image, landmarks_list, angles: Optional[Dict] = None) -> np.ndarray:
        """
        Render the enhanced skeleton visualization.
        
        Args:
            image: Input image
            landmarks_list: MediaPipe pose landmarks (list format)
            angles: Optional dictionary of joint angles
            
        Returns:
            Annotated image with skeleton visualization
        """
        # Create a completely fresh copy of the image to ensure no artifacts from previous frames
        annotated_image = np.copy(image)
        
        # Check if landmarks are present and valid
        has_valid_landmarks = landmarks_list is not None and len(landmarks_list) > 0
        using_cached_landmarks = False
        
        if has_valid_landmarks:
            # Convert landmarks to dictionary for easier access
            landmarks = {i: landmark for i, landmark in enumerate(landmarks_list)}
            
            # Validate landmarks to ensure they use image-relative coordinates
            if not self._validate_landmarks(landmarks):
                logging.warning("Invalid landmarks detected, checking if cached landmarks can be used")
                if self.previous_landmarks and self.frames_since_good_detection < 10:
                    landmarks = self.previous_landmarks
                    using_cached_landmarks = True
                    logging.info("Using cached landmarks due to invalid current landmarks")
                else:
                    logging.error("No valid landmarks available for rendering")
                    return annotated_image
            
            # Check if these landmarks have sufficient visibility/confidence
            avg_confidence = 0.0
            visible_landmarks = 0
            key_landmarks = [
                self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                self.mp_pose.PoseLandmark.LEFT_HIP.value,
                self.mp_pose.PoseLandmark.RIGHT_HIP.value
            ]
            
            for idx in key_landmarks:
                if idx in landmarks and hasattr(landmarks[idx], 'visibility'):
                    avg_confidence += landmarks[idx].visibility
                    visible_landmarks += 1
            
            # Calculate average confidence for key landmarks
            if visible_landmarks > 0:
                avg_confidence /= visible_landmarks
                
                # Update frames_since_good_detection based on confidence
                # Use lower threshold to be more inclusive
                if avg_confidence > 0.5:  # Good detection threshold
                    self.frames_since_good_detection = 0
                    self.previous_landmarks_confidence = avg_confidence
                else:
                    # Increment but with a lower rate for borderline detection
                    if avg_confidence > 0.3:
                        self.frames_since_good_detection += 0.5  # Half-count borderline frames
                    else:
                        self.frames_since_good_detection += 1
            else:
                # No visible landmarks
                self.frames_since_good_detection += 1
                avg_confidence = 0
                
            # Apply temporal smoothing to reduce flickering
            if self.previous_landmarks is not None:
                landmarks = self._smooth_landmarks(landmarks, self.previous_landmarks)
            
            # Store current landmarks for future frames
            self.previous_landmarks = landmarks.copy()
            
            # Prune history to prevent memory issues
            self._prune_history()
            
        elif self.previous_landmarks is not None and self.frames_since_good_detection < 10:
            # Use previous landmarks if they exist and it's been fewer than 10 frames
            # Increased from 3 to 10 frames for better persistence during brief occlusions
            landmarks = self.previous_landmarks
            self.frames_since_good_detection += 1
            using_cached_landmarks = True
            
            # Gradually reduce confidence for cached landmarks to implement fade-out
            self.previous_landmarks_confidence = max(0.1, self.previous_landmarks_confidence - 0.1)
            
            # Apply decay to all landmark visibilities
            # This will create a gradual fade-out effect
            for idx in self.previous_landmarks:
                if hasattr(self.previous_landmarks[idx], 'visibility'):
                    # Reduce visibility each cached frame
                    decay_rate = 0.15  # How much to fade per frame
                    new_vis = max(0.1, self.previous_landmarks[idx].visibility - decay_rate)
                    self.previous_landmarks[idx].visibility = new_vis
        else:
            # No landmarks and no valid previous landmarks
            return annotated_image
        
        # Final validation before generating virtual landmarks and rendering
        # Only proceed if we have valid landmarks with image-relative coordinates
        for idx, landmark in landmarks.items():
            if not (0 <= landmark.x <= self.SAFETY_CHECKS['coord_threshold'] and 
                    0 <= landmark.y <= self.SAFETY_CHECKS['coord_threshold']):
                logging.warning(f"Found landmark with invalid coordinates after processing: {landmark.x}, {landmark.y}")
                return annotated_image

        # Generate virtual landmarks
        virtual_landmarks = self._generate_virtual_landmarks(landmarks)
        
        # FIRST PASS: Handle motion trails if needed (background layer)
        if self.config.show_motion_trails:
            # Proceed with motion trails only if landmarks are valid
            self._render_motion_trails(annotated_image, landmarks, using_cached_landmarks)
            
        # SECOND PASS: Draw the main skeleton components on top
        self._draw_enhanced_skeleton(annotated_image, landmarks, virtual_landmarks)
        
        # THIRD PASS: Overlay angles, labels and reference lines if needed
        if self.config.show_angles and angles:
            self._draw_angles(annotated_image, landmarks, angles)
        
        if self.config.show_reference_lines:
            self._draw_reference_lines(annotated_image, landmarks, virtual_landmarks)
            
        # Apply visual feedback based on exercise type if specified
        if hasattr(self.config, 'exercise_type') and self.config.exercise_type:
            self._apply_exercise_feedback(annotated_image, landmarks, angles)
        
        return annotated_image

    def _draw_dashed_line(self, image, pt1, pt2, color, thickness, dash_length, space_length):
        """
        Draw a dashed line between two points.
        
        Args:
            image: Image to draw on
            pt1: Start point (x, y)
            pt2: End point (x, y)
            color: Line color in BGR format
            thickness: Line thickness
            dash_length: Length of each dash
            space_length: Length of each space between dashes
            
        Returns:
            None, modifies image in place
        """
        # Calculate direction vector
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Calculate distance and direction
        dx, dy = x2 - x1, y2 - y1
        dist = math.sqrt(dx * dx + dy * dy)
        
        if dist < 1:  # Avoid division by zero or very short lines
            return
            
        # Normalize direction vector
        dx, dy = dx / dist, dy / dist
        
        # Calculate number of segments
        segment_length = dash_length + space_length
        num_segments = max(1, int(dist / segment_length))
        
        # Draw dashed line
        for i in range(num_segments):
            start_dist = i * segment_length
            end_dist = min(start_dist + dash_length, dist)
            
            # Calculate points
            start_x = int(x1 + dx * start_dist)
            start_y = int(y1 + dy * start_dist)
            end_x = int(x1 + dx * end_dist)
            end_y = int(y1 + dy * end_dist)
            
            # Draw dash
            cv2.line(image, (start_x, start_y), (end_x, end_y), color, thickness, cv2.LINE_AA)

    def _render_motion_trails(self, image, landmarks, using_cached_landmarks):
        """
        Render motion trails from historical landmark positions.
        
        Args:
            image: Image to draw on
            landmarks: Current landmarks
            using_cached_landmarks: Whether we're using cached landmarks
        """
        height, width = image.shape[:2]
        
        # Update landmark history for motion trails
        if not hasattr(self, 'previous_landmark_history'):
            self.previous_landmark_history = []
        
        # Only add landmarks to history if they're not cached
        if not using_cached_landmarks:
            motion_trail_length = getattr(self.config, 'motion_trail_length', 5)
            
            if len(self.previous_landmark_history) >= motion_trail_length:
                self.previous_landmark_history.pop(0)
            self.previous_landmark_history.append(landmarks.copy())
        
        # Apply motion blur effect for analysis mode - only for history, not current frame
        if (len(self.previous_landmark_history) > 1 and 
            self.config.mode == VisualizationMode.ANALYSIS and 
            self.config.smoothing_factor > 0.3 and 
            self.config.show_motion_trails):
            
            # Draw motion trails with fading opacity (excluding the current frame)
            for i in range(len(self.previous_landmark_history) - 1, 0, -1):
                past_landmarks = self.previous_landmark_history[i]
                
                # Validate past landmarks before drawing
                valid_past_landmarks = True
                for _, lm in past_landmarks.items():
                    if not (0 <= lm.x <= self.SAFETY_CHECKS['coord_threshold'] and 
                            0 <= lm.y <= self.SAFETY_CHECKS['coord_threshold']):
                        valid_past_landmarks = False
                        break
                
                if not valid_past_landmarks:
                    continue
                
                # Improved opacity calculation for faster fading
                fade_ratio = 1 - i / len(self.previous_landmark_history)
                opacity = 0.4 * (fade_ratio * fade_ratio)  # Square the ratio for faster falloff
                
                # Only draw skeleton lines for motion trails with reduced opacity
                temp_color = self.colors.copy()
                for key in temp_color:
                    if isinstance(temp_color[key], tuple) and len(temp_color[key]) == 3:
                        # Create semi-transparent version 
                        alpha = int(255 * opacity * 0.5)
                        temp_color[key] = (*temp_color[key][:3], alpha)
                
                # Store original colors temporarily
                original_colors = self.colors
                self.colors = temp_color
                
                # Draw only skeleton for trail effect
                past_virtual = self._generate_virtual_landmarks(past_landmarks)
                self._draw_limbs(image, past_landmarks, height, width)
                
                # Restore original colors
                self.colors = original_colors

    def _apply_exercise_feedback(self, image, landmarks, angles):
        """
        Apply visual feedback based on exercise type and form quality.
        
        Args:
            image: Image to draw on
            landmarks: Current landmarks
            angles: Dictionary of joint angles
        """
        height, width = image.shape[:2]
        
        if not hasattr(self.config, 'exercise_type') or not self.config.exercise_type:
            return
            
        exercise_type = self.config.exercise_type
        
        # Add exercise type and mode label if configured
        if getattr(self.config, 'show_labels', False):
            mode_text = "Analysis Mode" if self.config.mode == VisualizationMode.ANALYSIS else f"Real-time: {exercise_type.capitalize()}"
            self._draw_text_with_shadow(
                image,
                mode_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        
        # Get exercise-specific configuration
        exercise_config = self._get_exercise_config()
        if not exercise_config:
            return
            
        # Apply exercise-specific feedback based on angles
        if not angles:
            return
            
        # Check angle thresholds for the exercise type and provide visual feedback
        if 'angle_thresholds' in exercise_config:
            for joint_type, (min_angle, max_angle) in exercise_config['angle_thresholds'].items():
                if joint_type == 'knee' and ('LEFT_KNEE' in angles or 'RIGHT_KNEE' in angles):
                    # Choose the knee with the smaller angle (deeper bend)
                    knee_key = 'LEFT_KNEE' if ('LEFT_KNEE' in angles and angles['LEFT_KNEE'].get('angle') is not None) else None
                    if not knee_key and 'RIGHT_KNEE' in angles and angles['RIGHT_KNEE'].get('angle') is not None:
                        knee_key = 'RIGHT_KNEE'
                        
                    if knee_key and angles[knee_key].get('angle') is not None:
                        angle_value = angles[knee_key]['angle']
                        
                        # Color feedback: red if outside range, green if good
                        if angle_value < min_angle or angle_value > max_angle:
                            # Outside acceptable range
                            self._highlight_joint(image, landmarks, knee_key, self.colors['feedback_bad'])
                        else:
                            # Good angle
                            self._highlight_joint(image, landmarks, knee_key, self.colors['feedback_good'])
                
                # Similar checks for other joint types (elbow, hip, etc)
                elif joint_type == 'elbow' and ('LEFT_ELBOW' in angles or 'RIGHT_ELBOW' in angles):
                    # Process elbow angles
                    elbow_key = 'LEFT_ELBOW' if ('LEFT_ELBOW' in angles and angles['LEFT_ELBOW'].get('angle') is not None) else None
                    if not elbow_key and 'RIGHT_ELBOW' in angles and angles['RIGHT_ELBOW'].get('angle') is not None:
                        elbow_key = 'RIGHT_ELBOW'
                        
                    if elbow_key and angles[elbow_key].get('angle') is not None:
                        angle_value = angles[elbow_key]['angle']
                        
                        if angle_value < min_angle or angle_value > max_angle:
                            self._highlight_joint(image, landmarks, elbow_key, self.colors['feedback_bad'])
                        else:
                            self._highlight_joint(image, landmarks, elbow_key, self.colors['feedback_good'])
                
                # Similarly for other types of joints (hip, spine, etc.)
        
        # If using cached landmarks, add an indicator
        if hasattr(self, 'frames_since_good_detection') and self.frames_since_good_detection > 0:
            # Add "Using cached data" indicator with shadow
            self._draw_text_with_shadow(
                image,
                "Using cached skeleton",
                (width - 250, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (66, 133, 244),  # Blue color
                2
            )
    
    def _highlight_joint(self, image, landmarks, joint_name, color):
        """
        Highlight a specific joint with a colored circle for feedback.
        
        Args:
            image: Image to draw on
            landmarks: Dictionary of landmarks
            joint_name: Name of the joint to highlight (e.g., 'LEFT_KNEE')
            color: Color for the highlight
        """
        height, width = image.shape[:2]
        
        # Map joint name to landmark index
        joint_map = {
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
        
        if joint_name in joint_map and joint_map[joint_name] in landmarks:
            landmark = landmarks[joint_map[joint_name]]
            
            # Validate coordinates
            if 0 <= landmark.x <= self.SAFETY_CHECKS['coord_threshold'] and 0 <= landmark.y <= self.SAFETY_CHECKS['coord_threshold']:
                # Draw a colored circle around the joint
                x, y = int(landmark.x * width), int(landmark.y * height)
                # Draw pulsing highlight circle
                cv2.circle(
                    image,
                    (x, y),
                    12,  # Larger than normal joints
                    color,
                    2,
                    cv2.LINE_AA
                )

    def _prune_history(self):
        """Limit history size to prevent memory issues."""
        # Prune motion trail
        if len(self.motion_trail) > self.motion_trail_length:
            self.motion_trail = self.motion_trail[-self.motion_trail_length:]
        
        # Prune landmark history
        if len(self.previous_landmark_history) > self.max_history_size:
            self.previous_landmark_history = self.previous_landmark_history[-self.max_history_size:]