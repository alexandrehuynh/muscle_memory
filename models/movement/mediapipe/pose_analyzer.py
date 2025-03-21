import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Generator, Callable
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
                 min_tracking_confidence: float = 0.7, enable_segmentation: bool = True,
                 smooth_landmarks: bool = True, smooth_segmentation: bool = True,
                 process_every_n_frames: int = 2, use_tasks_api: bool = False):
        """
        Initialize the PoseAnalyzer with MediaPipe configuration.
        
        Args:
            model_complexity: MediaPipe model complexity (0, 1, or 2)
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
            enable_segmentation: Whether to enable body segmentation
            smooth_landmarks: Whether to apply temporal filtering to reduce jitter
            smooth_segmentation: Whether to apply temporal filtering to segmentation mask
            process_every_n_frames: Process every N frames (skip frames for efficiency)
            use_tasks_api: Whether to use the newer MediaPipe Tasks API
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
            'smooth_landmarks': smooth_landmarks,
            'smooth_segmentation': smooth_segmentation,
            'process_every_n_frames': process_every_n_frames,
            'use_tasks_api': use_tasks_api
        }
        
        # Flag for API type
        self.use_tasks_api = use_tasks_api
        
        # Initialize variables to store angles data
        self.angles_data = []
        self.detected_joints = set()
        
        # Define joint mappings for angle calculations
        self.joint_mappings = self._create_joint_mappings()
        
        # Initialize pose detection model only for legacy API
        # For Tasks API, we'll initialize per usage mode
        if not use_tasks_api:
            self.pose = self.configure_mediapipe()
    
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
            static_image_mode=False,  # Set to False for video processing
            model_complexity=self.config['model_complexity'],
            min_detection_confidence=self.config['min_detection_confidence'],
            min_tracking_confidence=self.config['min_tracking_confidence'],
            enable_segmentation=self.config['enable_segmentation'],
            smooth_landmarks=self.config['smooth_landmarks'],
            smooth_segmentation=self.config['smooth_segmentation']
        )
    
    def configure_tasks_api(self, mode='VIDEO'):
        """Configure and return a MediaPipe PoseLandmarker instance using Tasks API."""
        try:
            # Import Tasks API components
            BaseOptions = mp.tasks.BaseOptions
            PoseLandmarker = mp.tasks.vision.PoseLandmarker
            PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode
            
            # Set model path (you may need to adjust this)
            model_path = 'pose_landmarker.task'
            
            # Create options based on mode
            if mode == 'IMAGE':
                options = PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=model_path),
                    running_mode=VisionRunningMode.IMAGE,
                    min_pose_detection_confidence=self.config['min_detection_confidence'],
                    min_pose_presence_confidence=self.config['min_tracking_confidence'],
                    min_tracking_confidence=self.config['min_tracking_confidence'],
                    output_segmentation_masks=self.config['enable_segmentation']
                )
            elif mode == 'VIDEO':
                options = PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=model_path),
                    running_mode=VisionRunningMode.VIDEO,
                    min_pose_detection_confidence=self.config['min_detection_confidence'],
                    min_pose_presence_confidence=self.config['min_tracking_confidence'],
                    min_tracking_confidence=self.config['min_tracking_confidence'],
                    output_segmentation_masks=self.config['enable_segmentation']
                )
            elif mode == 'LIVE_STREAM':
                # For live stream mode, we need to provide a callback
                def result_callback(result, output_image, timestamp_ms):
                    # Process results in real-time
                    if hasattr(result, 'pose_landmarks') and result.pose_landmarks:
                        # Calculate angles and store them
                        self._process_live_results(result, timestamp_ms)
                
                options = PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=model_path),
                    running_mode=VisionRunningMode.LIVE_STREAM,
                    min_pose_detection_confidence=self.config['min_detection_confidence'],
                    min_pose_presence_confidence=self.config['min_tracking_confidence'],
                    min_tracking_confidence=self.config['min_tracking_confidence'],
                    output_segmentation_masks=self.config['enable_segmentation'],
                    result_callback=result_callback
                )
            
            return PoseLandmarker.create_from_options(options)
        
        except (AttributeError, ImportError) as e:
            print(f"MediaPipe Tasks API not available: {str(e)}")
            print("Falling back to legacy MediaPipe API")
            self.use_tasks_api = False
            return self.configure_mediapipe()
    
    def _process_live_results(self, result, timestamp_ms):
        """Process live stream results from the callback."""
        # This method would handle real-time results from the Tasks API
        if hasattr(result, 'pose_landmarks') and result.pose_landmarks:
            # Similar to calculate_angles but for Tasks API results
            angles = {}
            
            # Process the landmarks
            # Implement this based on how Tasks API structures its results
            
            # Store the angles data with timestamp
            self.angles_data.append({
                'timestamp_ms': timestamp_ms,
                'angles': angles
            })
            
            # Update detected joints
            self.detected_joints.update(angles.keys())
        
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
            if self.use_tasks_api:
                # Use Tasks API for processing
                with self.configure_tasks_api(mode='VIDEO') as pose_landmarker:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        frame_count += 1
                        if frame_count % self.config['process_every_n_frames'] != 0:
                            continue
                        
                        # Convert to RGB and create MediaPipe Image
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                        
                        # Calculate timestamp for video (assuming constant FPS)
                        timestamp_ms = int((frame_count / fps) * 1000)
                        
                        # Process the frame
                        results = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
                        
                        if results.pose_landmarks:
                            # Draw landmarks if output is required
                            if out:
                                annotated_image = draw_landmarks_on_image(frame, results)
                                out.write(annotated_image)
                            
                            # Calculate joint angles (implement Tasks API version)
                            angles = self.calculate_angles_tasks(results, joints_to_process)
                            self.angles_data.append(angles)
                            self.detected_joints.update(angles.keys())
                        elif out:
                            # Write the original frame if no landmarks detected
                            out.write(frame)
                        
                        # Update progress
                        yield int((frame_count / total_frames) * 100)
            else:
                # Use legacy API for processing
                with self.configure_mediapipe() as pose:
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
    
    def calculate_angles_tasks(self, results, joints_to_process: List[str]) -> Dict:
        """Calculate angles for specified joints from pose landmarks using Tasks API results."""
        angles = {}
        
        # Implementation for Tasks API results structure
        # This would need to be adjusted based on exact Tasks API output structure
        
        return angles
    
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
        if self.use_tasks_api:
            # Use Tasks API
            with self.configure_tasks_api(mode='IMAGE') as pose_landmarker:
                # Convert to RGB and create MediaPipe Image
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
                
                # Process the image
                results = pose_landmarker.detect(mp_image)
                
                if results.pose_landmarks:
                    # Draw landmarks if output is required
                    if output_path:
                        annotated_image = draw_landmarks_on_image(image, results)
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        cv2.imwrite(output_path, annotated_image)
                    
                    # Calculate joint angles
                    angles = self.calculate_angles_tasks(results, joints_to_process)
                    self.angles_data.append(angles)
                    self.detected_joints.update(angles.keys())
                    
                    return True, f"Successfully processed image"
                else:
                    return False, "No pose landmarks detected in the image"
        else:
            # Use legacy API
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            with self.configure_mediapipe() as pose:
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
    
    def process_live_video(self, camera_id: int = 0, callback: Callable = None, 
                          joints_to_process: List[str] = None, display: bool = True) -> None:
        """
        Process live video from camera for real-time pose analysis.
        
        Args:
            camera_id: Camera ID to use
            callback: Optional callback function for real-time data processing
            joints_to_process: List of joint names to analyze
            display: Whether to display the video with landmarks
        """
        # Reset data
        self.angles_data = []
        self.detected_joints.clear()
        
        # Set default joints if none provided
        if joints_to_process is None:
            joints_to_process = list(self.joint_mappings.keys())
        
        # Open camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Unable to open camera {camera_id}")
            return
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Default to 30fps if not available
        
        frame_count = 0
        
        try:
            if self.use_tasks_api:
                # Use MediaPipe Tasks API with LIVE_STREAM mode
                def result_processor(result, output_image, timestamp_ms):
                    nonlocal frame_count
                    
                    if result.pose_landmarks:
                        # Calculate joint angles
                        angles = self.calculate_angles_tasks(result, joints_to_process)
                        self.angles_data.append(angles)
                        self.detected_joints.update(angles.keys())
                        
                        # Call user callback if provided
                        if callback:
                            callback(angles, timestamp_ms)
                    
                    frame_count += 1
                
                with self.configure_tasks_api(mode='LIVE_STREAM') as pose_landmarker:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        if frame_count % self.config['process_every_n_frames'] != 0:
                            frame_count += 1
                            continue
                        
                        # Convert to RGB and create MediaPipe Image
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                        
                        # Calculate timestamp
                        timestamp_ms = int(frame_count * (1000 / fps))
                        
                        # Process the frame asynchronously
                        pose_landmarker.detect_async(mp_image, timestamp_ms)
                        
                        # Display the frame with landmarks if requested
                        if display:
                            if hasattr(pose_landmarker, 'result') and pose_landmarker.result:
                                annotated_image = draw_landmarks_on_image(frame, pose_landmarker.result)
                                cv2.imshow('MediaPipe Pose', annotated_image)
                            else:
                                cv2.imshow('MediaPipe Pose', frame)
                        
                        # Break on ESC key
                        if cv2.waitKey(5) & 0xFF == 27:
                            break
            else:
                # Use legacy MediaPipe API
                with self.configure_mediapipe() as pose:
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
                        
                        timestamp_ms = int(frame_count * (1000 / fps))
                        
                        if results.pose_landmarks:
                            # Draw pose landmarks if display is enabled
                            if display:
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
                            
                            # Call user callback if provided
                            if callback:
                                callback(angles, timestamp_ms)
                        
                        # Display the frame with landmarks if requested
                        if display:
                            cv2.imshow('MediaPipe Pose', image)
                        
                        # Break on ESC key
                        if cv2.waitKey(5) & 0xFF == 27:
                            break
        
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
    
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

# Utility function for drawing landmarks with Tasks API
def draw_landmarks_on_image(image, results):
    """Draw pose landmarks on an image using Tasks API results structure."""
    # This implementation would need to be adjusted based on Tasks API output
    # Current implementation is a placeholder
    annotated_image = image.copy()
    
    # Draw landmarks based on Tasks API structure
    # This is a placeholder that would need to be implemented
    
    return annotated_image