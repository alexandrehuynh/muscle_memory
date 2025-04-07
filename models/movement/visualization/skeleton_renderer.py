import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import mediapipe as mp
from dataclasses import dataclass
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
    smoothing_factor: float = 0.7  # Controls intensity of smoothing (0.0-1.0)
    min_detection_threshold: float = 0.2  # Discard very low confidence detections
    line_style: str = "solid"  # Options: "solid", "dashed", "outlined"

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
            'spine_line': (255, 255, 255),  # White
            'shoulder_line': (255, 255, 255),  # White
            'hip_line': (255, 255, 255),  # White
            'knee_projection': (86, 237, 227)  # Brighter orange
        }
        
        # Store previous landmarks for motion trails and smoothing
        self.previous_landmarks = None
        self.motion_trail = []
        self.max_trail_length = 10
        
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
            virtual_landmarks['PLUMB_LINE_BOTTOM'] = (mid_shoulder.x, 1.1, 0)  # Extend beyond feet
            
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
        
        # Draw extended shoulder line
        if 'shoulder_line' in reference_lines and 'SHOULDER_LINE_LEFT' in virtual_landmarks and 'SHOULDER_LINE_RIGHT' in virtual_landmarks:
            left = virtual_landmarks['SHOULDER_LINE_LEFT']
            right = virtual_landmarks['SHOULDER_LINE_RIGHT']
            
            pt1 = (int(left[0] * width), int(left[1] * height))
            pt2 = (int(right[0] * width), int(right[1] * height))
            
            if self.config.line_style == "outlined":
                self._draw_outlined_line(image, pt1, pt2, self.colors['shoulder_line'], 3)
            else:
                cv2.line(
                    image,
                    pt1,
                    pt2,
                    self.colors['shoulder_line'],
                    3,  # Increased thickness
                    cv2.LINE_AA
                )
        
        # Draw squat-specific reference lines
        if self.config.exercise_type == 'squat':
            # Draw knee projection lines for tracking knees over toes
            if 'knee_projection' in reference_lines and 'LEFT_KNEE_PROJECTION' in virtual_landmarks:
                left_knee = landmarks.get(self.mp_pose.PoseLandmark.LEFT_KNEE.value)
                left_proj = virtual_landmarks['LEFT_KNEE_PROJECTION']
                
                if left_knee:
                    pt1 = (int(left_knee.x * width), int(left_knee.y * height))
                    pt2 = (int(left_proj.x * width), int(left_proj.y * height))
                    
                    if self.config.line_style == "outlined":
                        self._draw_outlined_line(image, pt1, pt2, self.colors['knee_projection'], 3)
                    else:
                        cv2.line(
                            image,
                            pt1,
                            pt2,
                            self.colors['knee_projection'],
                            3,  # Increased thickness
                            cv2.LINE_AA
                        )
            
            if 'knee_projection' in reference_lines and 'RIGHT_KNEE_PROJECTION' in virtual_landmarks:
                right_knee = landmarks.get(self.mp_pose.PoseLandmark.RIGHT_KNEE.value)
                right_proj = virtual_landmarks['RIGHT_KNEE_PROJECTION']
                
                if right_knee:
                    pt1 = (int(right_knee.x * width), int(right_knee.y * height))
                    pt2 = (int(right_proj.x * width), int(right_proj.y * height))
                    
                    if self.config.line_style == "outlined":
                        self._draw_outlined_line(image, pt1, pt2, self.colors['knee_projection'], 3)
                    else:
                        cv2.line(
                            image,
                            pt1,
                            pt2,
                            self.colors['knee_projection'],
                            3,  # Increased thickness
                            cv2.LINE_AA
                        )
            
            # Draw target squat depth line
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
                        cv2.line(
                            image,
                            pt1,
                            pt2,
                            (66, 245, 200),  # Light blue-green
                            3,  # Increased thickness
                            cv2.LINE_AA
                        )
    
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
            radius = 35  # Slightly larger radius
            start_angle = 0
            end_angle = angle
            
            # Adjust arc orientation based on joint type
            if joint_type == 'knee':
                start_angle = 180
                end_angle = 180 + angle
            elif joint_type == 'elbow':
                start_angle = 90
                end_angle = 90 + angle
                
            # Draw arc with thicker line
            cv2.ellipse(
                image,
                pos,
                (radius, radius),
                0,  # Rotation
                start_angle,
                end_angle,
                color,
                3,  # Increased thickness
                cv2.LINE_AA
            )
            
            # Draw angle text with shadow
            text_pos = (pos[0] + int(radius * 0.7 * math.cos(math.radians((start_angle + end_angle) / 2))),
                        pos[1] + int(radius * 0.7 * math.sin(math.radians((start_angle + end_angle) / 2))))
            
            self._draw_text_with_shadow(
                image,
                f"{int(angle)}°",
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,  # Larger font scale
                color,
                2  # Thicker font
            )
    
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
            if len(self.motion_trail) > self.max_trail_length:
                self.motion_trail.pop(0)
            
            # Draw trails
            for i in range(len(self.motion_trail) - 1):
                alpha = (i + 1) / len(self.motion_trail)  # Fade in newer trails
                trail_color = (*self.colors['motion_trail'], int(255 * alpha))
                
                for idx in tracked_landmarks:
                    if idx in self.motion_trail[i] and idx in self.motion_trail[i + 1]:
                        cv2.line(
                            image,
                            self.motion_trail[i][idx],
                            self.motion_trail[i + 1][idx],
                            trail_color,
                            1,
                            cv2.LINE_AA
                        )
        
        self.previous_landmarks = landmarks
    
    def _draw_spine(self, image, landmarks, virtual_landmarks, height, width):
        """
        Draw the multi-segment spine.
        
        Args:
            image: Image to draw on
            landmarks: Dictionary of MediaPipe pose landmarks
            virtual_landmarks: Dictionary of virtual landmarks
            height: Image height
            width: Image width
        """
        # Define spine segments (neck to hips with multiple segments)
        spine_segments = [
            ('NECK', 'CERVICAL_SPINE'),
            ('CERVICAL_SPINE', 'THORACIC_SPINE_UPPER'),
            ('THORACIC_SPINE_UPPER', 'THORACIC_SPINE_LOWER'),
            ('THORACIC_SPINE_LOWER', 'LUMBAR_SPINE_UPPER'),
            ('LUMBAR_SPINE_UPPER', 'LUMBAR_SPINE_LOWER'),
            ('LUMBAR_SPINE_LOWER', 'MID_SPINE')
        ]
        
        # Draw each spine segment
        for start_name, end_name in spine_segments:
            if start_name not in virtual_landmarks or end_name not in virtual_landmarks:
                continue
                
            start = virtual_landmarks[start_name]
            end = virtual_landmarks[end_name]
            
            # Get confidence based on nearby real landmarks
            confidence = min(
                self._get_confidence(landmarks.get(self.mp_pose.PoseLandmark.NOSE.value, 1.0)),
                self._get_confidence(landmarks.get(self.mp_pose.PoseLandmark.LEFT_SHOULDER.value, 1.0)),
                self._get_confidence(landmarks.get(self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value, 1.0)),
                self._get_confidence(landmarks.get(self.mp_pose.PoseLandmark.LEFT_HIP.value, 1.0)),
                self._get_confidence(landmarks.get(self.mp_pose.PoseLandmark.RIGHT_HIP.value, 1.0))
            )
            
            # Skip very low confidence segments
            if confidence < self.config.min_detection_threshold:
                continue
                
            # Prepare points
            pt1 = (int(start.x * width), int(start.y * height))
            pt2 = (int(end.x * width), int(end.y * height))
            
            line_thickness = 4  # Increased thickness
            color = self.colors['spine_line']
            
            # Determine line style based on confidence and configuration
            if confidence < self.config.confidence_threshold:
                if self.config.line_style == "dashed" or self.config.line_style == "outlined":
                    # Create dashed line effect for low confidence
                    pts = np.array([pt1, pt2], np.int32)
                    
                    # Draw dashed line manually
                    dash_length = 7  # Slightly longer dashes for visibility
                    gap_length = 4
                    distance = np.sqrt(np.sum((pts[0] - pts[1])**2))
                    
                    if distance > 0:
                        num_segments = int(distance / (dash_length + gap_length))
                        
                        for i in range(num_segments):
                            start_segment = pts[0] + (pts[1] - pts[0]) * (i * (dash_length + gap_length)) / distance
                            end_segment = pts[0] + (pts[1] - pts[0]) * (i * (dash_length + gap_length) + dash_length) / distance
                            
                            if end_segment[0] > pts[1][0]:
                                end_segment = pts[1]
                                
                            dash_pt1 = (int(start_segment[0]), int(start_segment[1]))
                            dash_pt2 = (int(end_segment[0]), int(end_segment[1]))
                            
                            if self.config.line_style == "outlined":
                                self._draw_outlined_line(image, dash_pt1, dash_pt2, color, line_thickness)
                            else:
                                cv2.line(image, dash_pt1, dash_pt2, color, line_thickness, cv2.LINE_AA)
                else:
                    # Solid line with reduced opacity
                    # Create a version of spine_line with alpha
                    spine_color_low_conf = (*color, 128)  # Add alpha channel
                    cv2.line(image, pt1, pt2, spine_color_low_conf, line_thickness, cv2.LINE_AA)
            else:
                # High confidence line
                if self.config.line_style == "outlined":
                    self._draw_outlined_line(image, pt1, pt2, color, line_thickness)
                else:
                    cv2.line(image, pt1, pt2, color, line_thickness, cv2.LINE_AA)
    
    def _draw_limbs(self, image, landmarks, height, width):
        """
        Draw limbs with proper biomechanical connections.
        
        Args:
            image: Image to draw on
            landmarks: Dictionary of MediaPipe pose landmarks
            height: Image height
            width: Image width
        """
        # Define limb connections
        connections = [
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
        
        # Draw each connection
        for start_idx, end_idx in connections:
            if start_idx not in landmarks or end_idx not in landmarks:
                continue
                
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            
            # Get confidence
            confidence = min(self._get_confidence(start), self._get_confidence(end))
            
            # Skip very low confidence connections
            if confidence < self.config.min_detection_threshold:
                continue
                
            # Prepare points
            pt1 = (int(start.x * width), int(start.y * height))
            pt2 = (int(end.x * width), int(end.y * height))
            
            line_thickness = 4  # Increased thickness
            
            # Determine line style based on confidence and configuration
            if confidence < self.config.confidence_threshold:
                if self.config.line_style == "dashed" or self.config.line_style == "outlined":
                    # Create dashed line effect for low confidence
                    pts = np.array([pt1, pt2], np.int32)
                    
                    # Draw dashed line manually
                    dash_length = 7  # Slightly longer dashes for visibility
                    gap_length = 4
                    distance = np.sqrt(np.sum((pts[0] - pts[1])**2))
                    
                    if distance > 0:
                        num_segments = int(distance / (dash_length + gap_length))
                        
                        for i in range(num_segments):
                            start_segment = pts[0] + (pts[1] - pts[0]) * (i * (dash_length + gap_length)) / distance
                            end_segment = pts[0] + (pts[1] - pts[0]) * (i * (dash_length + gap_length) + dash_length) / distance
                            
                            if end_segment[0] > pts[1][0]:
                                end_segment = pts[1]
                                
                            dash_pt1 = (int(start_segment[0]), int(start_segment[1]))
                            dash_pt2 = (int(end_segment[0]), int(end_segment[1]))
                            
                            if self.config.line_style == "outlined":
                                self._draw_outlined_line(image, dash_pt1, dash_pt2, self.colors['skeleton'], line_thickness)
                            else:
                                cv2.line(image, dash_pt1, dash_pt2, self.colors['skeleton'], line_thickness, cv2.LINE_AA)
                else:
                    # Solid line with reduced opacity
                    cv2.line(image, pt1, pt2, self.colors['skeleton_low_conf'], line_thickness, cv2.LINE_AA)
            else:
                # High confidence line
                if self.config.line_style == "outlined":
                    self._draw_outlined_line(image, pt1, pt2, self.colors['skeleton'], line_thickness)
                else:
                    cv2.line(image, pt1, pt2, self.colors['skeleton'], line_thickness, cv2.LINE_AA)
    
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
            confidence = self._get_confidence(landmark)
            
            if confidence < self.config.min_detection_threshold:
                continue  # Skip very low confidence landmarks
                
            # Determine joint size based on importance
            radius = 8 if idx in primary_joints else 5  # Increased sizes
            color = self.colors['joints']
            
            # Adjust based on confidence
            if confidence < self.config.confidence_threshold:
                color = self.colors['joints_low_conf']
                radius = max(3, radius - 2)
            
            # Draw joint circle
            cv2.circle(
                image,
                (int(landmark.x * width), int(landmark.y * height)),
                radius,
                color,
                -1,
                cv2.LINE_AA
            )
            
            # Add a white border to primary joints for better visibility
            if idx in primary_joints and confidence >= self.config.confidence_threshold:
                cv2.circle(
                    image,
                    (int(landmark.x * width), int(landmark.y * height)),
                    radius + 1,
                    (255, 255, 255),
                    1,
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
    
    def _draw_outlined_line(self, image, pt1, pt2, color, thickness=4):
        """Draw a line with a contrasting outline for better visibility."""
        # Draw slightly thicker black outline
        cv2.line(image, pt1, pt2, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        # Draw the colored line on top
        cv2.line(image, pt1, pt2, color, thickness, cv2.LINE_AA)
    
    def _draw_text_with_shadow(self, image, text, position, font_face, font_scale, color, thickness=1):
        """Draw text with shadow for better visibility against any background."""
        shadow_position = (position[0] + 2, position[1] + 2)
        cv2.putText(image, text, shadow_position, font_face, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(image, text, position, font_face, font_scale, color, thickness, cv2.LINE_AA)
    
    def _smooth_landmarks(self, current_landmarks, previous_landmarks):
        """
        Apply temporal smoothing to reduce flickering.
        
        Args:
            current_landmarks: Current frame landmarks
            previous_landmarks: Previous frame landmarks
            
        Returns:
            Smoothed landmarks
        """
        if previous_landmarks is None:
            return current_landmarks
            
        smoothed = {}
        for idx, landmark in current_landmarks.items():
            # Skip very low confidence landmarks
            if hasattr(landmark, 'visibility') and landmark.visibility < self.config.min_detection_threshold:
                continue
                
            if idx in previous_landmarks:
                # Apply exponential smoothing
                prev = previous_landmarks[idx]
                
                # Create a new point with the same structure
                class Point:
                    def __init__(self):
                        self.x = 0
                        self.y = 0
                        self.z = 0
                        self.visibility = 0
                
                smoothed[idx] = Point()
                
                # Apply exponential smoothing to coordinates
                smoothed[idx].x = prev.x * self.config.smoothing_factor + landmark.x * (1 - self.config.smoothing_factor)
                smoothed[idx].y = prev.y * self.config.smoothing_factor + landmark.y * (1 - self.config.smoothing_factor)
                smoothed[idx].z = getattr(prev, 'z', 0) * self.config.smoothing_factor + getattr(landmark, 'z', 0) * (1 - self.config.smoothing_factor)
                smoothed[idx].visibility = landmark.visibility  # Don't smooth visibility
            else:
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
        if landmarks_list is None or len(landmarks_list) == 0:
            return image
        
        # Create a copy of the image to draw on
        annotated_image = image.copy()
        
        # Convert landmarks to dictionary for easier access
        landmarks = {i: landmark for i, landmark in enumerate(landmarks_list)}
        
        # Apply temporal smoothing to reduce flickering
        landmarks = self._smooth_landmarks(landmarks, self.previous_landmarks)
        self.previous_landmarks = landmarks.copy()  # Store for next frame
        
        # Generate virtual landmarks
        virtual_landmarks = self._generate_virtual_landmarks(landmarks)
        
        # Draw different visualization elements based on mode
        if self.config.mode == VisualizationMode.ANALYSIS:
            # Draw detailed visualization
            self._draw_enhanced_skeleton(annotated_image, landmarks, virtual_landmarks)
            self._draw_reference_lines(annotated_image, landmarks, virtual_landmarks)
            if angles:
                self._draw_angles(annotated_image, landmarks, angles)
            self._draw_motion_trails(annotated_image, landmarks)
            
            # Add visualization mode indicator with shadow
            self._draw_text_with_shadow(
                annotated_image,
                "Analysis Mode",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        else:
            # Draw simplified visualization for real-time mode
            self._draw_enhanced_skeleton(annotated_image, landmarks, virtual_landmarks)
            if angles:
                self._draw_angles(annotated_image, landmarks, angles)
                
            # Add exercise type and visualization mode indicator with shadow
            if self.config.exercise_type:
                self._draw_text_with_shadow(
                    annotated_image,
                    f"Real-time: {self.config.exercise_type.capitalize()}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
        
        return annotated_image 