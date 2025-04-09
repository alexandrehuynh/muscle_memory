import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Generator, Callable
import os
import json
import logging
import math
from datetime import datetime
from ..visualization.skeleton_renderer import SkeletonRenderer, VisualizationMode, VisualizationConfig

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
                 process_every_n_frames: int = 1, use_tasks_api: bool = False,
                 analysis_type: str = None):
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
            use_tasks_api: Whether to use the newer MediaPipe Tasks API (ignored, always False)
            analysis_type: Type of analysis to perform (affects segmentation needs)
        """
        # Initialize MediaPipe drawing and pose solutions
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        
        # Initialize skeleton renderer with enhanced configuration
        is_detailed = analysis_type in ['detailed_analysis', 'form_visualization']
        self.skeleton_renderer = SkeletonRenderer(
            VisualizationConfig(
                mode=VisualizationMode.ANALYSIS if is_detailed else VisualizationMode.REALTIME,
                show_reference_lines=is_detailed,
                show_angles=True,
                show_motion_trails=is_detailed,
                confidence_threshold=min_detection_confidence,
                smoothing_factor=0.7 if smooth_landmarks else 0.0,  # Use smoothing factor if enabled
                min_detection_threshold=min_detection_confidence * 0.4,  # Set threshold relative to detection confidence
                line_style="outlined" if is_detailed else "solid"  # Use outlined style for detailed analysis
            )
        )
        
        # Store configuration
        self.config = {
            'model_complexity': model_complexity,
            'min_detection_confidence': min_detection_confidence,
            'min_tracking_confidence': min_tracking_confidence,
            'enable_segmentation': enable_segmentation,
            'smooth_landmarks': smooth_landmarks,
            'smooth_segmentation': smooth_segmentation,
            'process_every_n_frames': process_every_n_frames,
            'use_tasks_api': False,  # Always use legacy API
            'analysis_type': analysis_type,
            'confidence_threshold': 0.6,  # Threshold for pose caching
            'landmark_history_size': 10,  # Store history of 10 frames for each landmark
            'confidence_decay_rate': 0.85  # Rate at which confidence decays over time
        }
        
        # Flag for API type - always use legacy API
        self.use_tasks_api = False
        
        # Initialize variables to store angles data
        self.angles_data = []
        self.detected_joints = set()
        
        # Define joint mappings for angle calculations
        self.joint_mappings = self._create_joint_mappings()
        
        # Initialize pose detection model only for legacy API
        self.pose = None  # Will be initialized when needed
            
        # Enhanced landmark persistence system
        self.previous_landmarks = None  # Keep for backward compatibility
        self.landmark_history = {}  # Dictionary to store history of landmarks
        self.landmark_confidence_history = {}  # Dictionary to store confidence history
        self.frames_since_detection = {}  # Track frames since last good detection per landmark
        self.current_mode = None  # Track current mode for cross-mode consistency
    
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
        
    def configure_mediapipe(self, exercise_type=None):
        """
        Configure and return a MediaPipe Pose instance.
        
        Args:
            exercise_type: Optional exercise type to adjust model complexity
        """
        # Determine model complexity based on exercise type
        if exercise_type is not None:
            if exercise_type in ['yoga', 'balance_poses']:
                complexity = 2  # Highest precision for poses requiring balance/stability
            elif exercise_type in ['squat', 'lunge', 'pushup']:
                complexity = 1  # Medium precision for standard exercises
            else:
                complexity = 0  # Lower precision for simpler tracking needs
        else:
            complexity = self.config['model_complexity']
        
        # Make segmentation optional based on analysis type
        enable_seg = True
        if self.config.get('analysis_type') in ['basic_tracking', 'angle_only', 'quick_analysis']:
            enable_seg = False
        elif self.config.get('analysis_type') in ['form_visualization', 'detailed_analysis']:
            enable_seg = True
        else:
            enable_seg = self.config['enable_segmentation']
        
        return self.mp_pose.Pose(
            static_image_mode=False,  # Set to False for video processing
            model_complexity=complexity,
            min_detection_confidence=self.config['min_detection_confidence'],
            min_tracking_confidence=self.config['min_tracking_confidence'],
            enable_segmentation=enable_seg,
            smooth_landmarks=self.config['smooth_landmarks'],
            smooth_segmentation=self.config['smooth_segmentation']
        )
        
    def configure_tasks_api(self, mode='VIDEO', exercise_type=None):
        """Configure and return a MediaPipe PoseLandmarker instance using Tasks API."""
        try:
            # Import Tasks API components
            BaseOptions = mp.tasks.BaseOptions
            PoseLandmarker = mp.tasks.vision.PoseLandmarker
            PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode
            
            # Determine model complexity based on exercise type
            if exercise_type is not None:
                if exercise_type in ['yoga', 'balance_poses']:
                    model_path = 'pose_landmarker_heavy.task'
                elif exercise_type in ['squat', 'lunge', 'pushup']:
                    model_path = 'pose_landmarker.task'
                else:
                    model_path = 'pose_landmarker_lite.task'
            else:
                # Default model path based on complexity
                if self.config['model_complexity'] == 2:
                    model_path = 'pose_landmarker_heavy.task'
                elif self.config['model_complexity'] == 1:
                    model_path = 'pose_landmarker.task'
                else:
                    model_path = 'pose_landmarker_lite.task'
            
            # Ensure the model exists, otherwise fall back to legacy API
            model_asset_path = model_path
            if not os.path.exists(model_asset_path):
                logging.warning(f"Model file {model_asset_path} not found. Using legacy MediaPipe API instead.")
                self.use_tasks_api = False
                return self.configure_mediapipe(exercise_type)
                
            # Create options based on mode
            options = None
            
            if mode == 'IMAGE':
                options = PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=model_asset_path),
                    running_mode=VisionRunningMode.IMAGE,
                    num_poses=1,  # Just detect one person
                    min_pose_detection_confidence=self.config['min_detection_confidence'],
                    min_pose_presence_confidence=self.config['min_tracking_confidence'],
                    min_tracking_confidence=self.config['min_tracking_confidence'],
                    output_segmentation_masks=self.config['enable_segmentation']
                )
            elif mode == 'VIDEO':
                options = PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=model_asset_path),
                    running_mode=VisionRunningMode.VIDEO,
                    num_poses=1,
                    min_pose_detection_confidence=self.config['min_detection_confidence'],
                    min_pose_presence_confidence=self.config['min_tracking_confidence'],
                    min_tracking_confidence=self.config['min_tracking_confidence'],
                    output_segmentation_masks=self.config['enable_segmentation']
                )
            
            return PoseLandmarker.create_from_options(options)
            
        except Exception as e:
            logging.error(f"Failed to configure MediaPipe Tasks API: {e}")
            logging.info("Falling back to legacy MediaPipe API")
            self.use_tasks_api = False
            return self.configure_mediapipe(exercise_type)
    
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
        
    def calculate_angle(self, a: List[float], b: List[float], c: List[float]) -> Optional[float]:
        """
        Calculate the angle between three points in 3D space.
        
        Args:
            a: First point coordinates [x, y, z]
            b: Second point (vertex) coordinates [x, y, z]
            c: Third point coordinates [x, y, z]
            
        Returns:
            Angle in degrees or None if calculation fails
        """
        try:
            a = np.array(a)
            b = np.array(b)
            c = np.array(c)
            
            # Check for NaN values
            if np.isnan(a).any() or np.isnan(b).any() or np.isnan(c).any():
                logging.warning(f"NaN values detected in coordinates: a={a}, b={b}, c={c}")
                return None
                
            # Calculate vectors
            ba = a - b
            bc = c - b
            
            # Check for zero-length vectors
            ba_norm = np.linalg.norm(ba)
            bc_norm = np.linalg.norm(bc)
            
            if ba_norm < 1e-10 or bc_norm < 1e-10:
                logging.warning(f"Zero-length vector detected: |ba|={ba_norm}, |bc|={bc_norm}")
                return None
                
            # Calculate angle using dot product
            cosine_angle = np.dot(ba, bc) / (ba_norm * bc_norm)
            
            # Ensure value is within valid range for arccos
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            
            angle = np.degrees(np.arccos(cosine_angle))
            
            # Final sanity check
            if np.isnan(angle) or np.isinf(angle):
                logging.warning(f"Invalid angle result: {angle}")
                return None
                
            return float(angle)  # Convert to native Python float
            
        except Exception as e:
            logging.error(f"Error in calculate_angle: {e}")
            return None
        
    def get_landmark_coords(self, landmark) -> List[float]:
        """
        Extract coordinates from a landmark safely.
        
        Args:
            landmark: Landmark object that should have x, y, z attributes
            
        Returns:
            List of [x, y, z] coordinates
        """
        try:
            # Try to access coordinates with robust error handling
            x = getattr(landmark, 'x', 0.0)
            y = getattr(landmark, 'y', 0.0)
            
            # Z might not always be present
            try:
                z = getattr(landmark, 'z', 0.0)
            except Exception:
                z = 0.0
                
            return [x, y, z]
        except Exception as e:
            logging.warning(f"Error extracting landmark coordinates: {e}")
            return [0.0, 0.0, 0.0]  # Return origin as fallback
    
    def process_video_batch(self, frames, exercise_type=None, batch_size=10):
        """
        Process multiple frames in a batch for performance.
        
        Args:
            frames: List of frames to process
            exercise_type: Optional exercise type to adjust model complexity
            batch_size: Number of frames to process in each batch
            
        Returns:
            List of processed results
        """
        results = []
        
        # Initialize pose if not already done
        if self.pose is None:
            self.pose = self.configure_mediapipe(exercise_type)
        
        with self.pose as pose:
            for i in range(0, len(frames), batch_size):
                batch = frames[i:i+batch_size]
                # Preprocess all frames in the batch
                preprocessed = []
                for frame in batch:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb_frame.flags.writeable = False
                    preprocessed.append(rgb_frame)
                
                # Process each frame in the batch
                batch_results = []
                for frame in preprocessed:
                    result = pose.process(frame)
                    
                    # Apply smart caching for stability if needed
                    if result.pose_landmarks:
                        landmark_visibility = [lm.visibility for lm in result.pose_landmarks.landmark]
                        avg_visibility = sum(landmark_visibility) / len(landmark_visibility)
                        
                        if avg_visibility < self.config['confidence_threshold'] and self.previous_landmarks:
                            # Use previous landmarks if current detection is poor
                            result.pose_landmarks = self.previous_landmarks
                        else:
                            # Update previous landmarks if current detection is good
                            self.previous_landmarks = result.pose_landmarks
                    
                    batch_results.append(result)
                
                results.extend(batch_results)
        
        return results
    
    def clear_history(self):
        """Clear all history data to prevent memory issues."""
        self.previous_landmarks = None
        self.landmark_history = {}
        self.landmark_confidence_history = {}
        self.frames_since_detection = {}
        self.angles_data = []
        self.detected_joints.clear()

    def process_video(self, video_path: str, output_path: str = None, 
                      joints_to_process: List[str] = None, exercise_type: str = None, 
                      analysis_type: str = None) -> Generator[int, None, Tuple[bool, str]]:
        """
        Process a video file to detect and analyze poses.
        
        Args:
            video_path: Path to the input video file
            output_path: Path to save the output video (optional)
            joints_to_process: List of joint names to analyze
            exercise_type: Type of exercise for model complexity adjustment
            analysis_type: Type of analysis to perform (affects processing options)
            
        Returns:
            Generator yielding progress percentage
            Final tuple of (success, message)
        """
        # Reset data
        self.clear_history()
        
        # Update skeleton renderer configuration
        self.skeleton_renderer.config.exercise_type = exercise_type
        
        # Apply analysis_type if provided
        if analysis_type is not None:
            self.config['analysis_type'] = analysis_type
        
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
            # New output naming system: [year][month][day][rest_of_datetime][original_video_name]
            if output_path == os.path.join("output", f"analyzed_{os.path.basename(video_path)}"):
                # Only modify default naming pattern
                now = datetime.now()
                date_prefix = now.strftime("%Y%m%d_%H%M%S")
                original_name = os.path.basename(video_path)
                new_output_name = f"{date_prefix}_{original_name}"
                output_dir = os.path.dirname(output_path)
                output_path = os.path.join(output_dir, new_output_name)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        else:
            out = None
        
        frame_count = 0
        
        try:
            # Use legacy API for processing
            with self.configure_mediapipe(exercise_type) as pose:
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
                    # Add image dimensions explicitly
                    image_height, image_width = image.shape[:2]
                    results = pose.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                    # Apply smart caching for pose stability
                    if results.pose_landmarks:
                        landmark_visibility = [lm.visibility for lm in results.pose_landmarks.landmark]
                        avg_visibility = sum(landmark_visibility) / len(landmark_visibility)
                        
                        # Update landmark history regardless of visibility
                        self._update_landmark_history(results.pose_landmarks, {})
                        
                        # Keep backward compatibility with previous_landmarks
                        if avg_visibility >= self.config['confidence_threshold']:
                            self.previous_landmarks = results.pose_landmarks
                    
                    # Track current mode for cross-mode consistency
                    if hasattr(results, 'current_mode'):
                        # Handle mode transition
                        if self.current_mode != results.current_mode:
                            # We're transitioning modes, keep using history more heavily
                            old_decay_rate = self.config['confidence_decay_rate']
                            self.config['confidence_decay_rate'] = min(0.95, old_decay_rate * 1.2)
                            
                            # Restore normal decay rate after a few frames
                            def restore_decay_rate():
                                self.config['confidence_decay_rate'] = old_decay_rate
                            # (In a real implementation, you'd need a timer or frame counter to restore)
                            
                        self.current_mode = results.current_mode
                    # Add handling for direct mode change from MovementAnalyzer
                    elif hasattr(self, 'current_mode') and self.current_mode is not None:
                        # This case handles mode changes set directly by MovementAnalyzer
                        pass  # The current_mode is already set, nothing more to do
                    
                    # Draw pose landmarks if output is required
                    if out:
                        # Calculate joint angles
                        angles = self.calculate_angles(results, joints_to_process)
                        
                        # Use skeleton renderer for visualization
                        annotated_image = self.skeleton_renderer.render(
                            image,
                            results.pose_landmarks.landmark,
                            angles
                        )
                        out.write(annotated_image)
                        
                    # Make sure we calculate and store angles data regardless of output
                    if results.pose_landmarks:
                        # Calculate joint angles and store the data
                        angles = self.calculate_angles(results, joints_to_process)
                        self.angles_data.append(angles)
                        self.detected_joints.update(angles.keys())
                    
                    # Write the annotated frame to output video
                    if out and not out.isOpened():
                        logging.warning("Output video writer is not open, trying to reopen")
                        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
                        if out.isOpened() and results.pose_landmarks:
                            out.write(annotated_image)
                    
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
                      joints_to_process: List[str] = None, exercise_type: str = None) -> Tuple[bool, str]:
        """
        Process a single image to detect and analyze pose.
        
        Args:
            image_path: Path to the input image
            output_path: Path to save the annotated output image
            joints_to_process: List of joint names to analyze
            exercise_type: Type of exercise for model complexity adjustment
            
        Returns:
            Tuple of (success, message)
        """
        # Reset data
        self.clear_history()
        
        # Update skeleton renderer configuration
        self.skeleton_renderer.config.exercise_type = exercise_type
        
        # Set default joints if none provided
        if joints_to_process is None:
            joints_to_process = list(self.joint_mappings.keys())
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return False, f"Error: Unable to read image file {image_path}"
        
        # Process image using legacy API
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        with self.configure_mediapipe(exercise_type) as pose:
            results = pose.process(image_rgb)
            
            if results.pose_landmarks:
                # Draw landmarks if output is required
                if output_path:
                    # Calculate joint angles
                    angles = self.calculate_angles(results, joints_to_process)
                    
                    # Use skeleton renderer for visualization
                    annotated_image = self.skeleton_renderer.render(
                        image,
                        results.pose_landmarks.landmark,
                        angles
                    )
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    cv2.imwrite(output_path, annotated_image)
                
                # Calculate joint angles
                angles = self.calculate_angles(results, joints_to_process)
                self.angles_data.append(angles)
                self.detected_joints.update(angles.keys())
                
                return True, f"Successfully processed image"
            else:
                return False, "No pose landmarks detected in the image"
    
    def process_live_video(self, camera_id: int = 0, callback: Callable = None, 
                          joints_to_process: List[str] = None, display: bool = True,
                          exercise_type: str = None) -> None:
        """
        Process live video from camera for real-time pose analysis.
        
        Args:
            camera_id: Camera ID to use
            callback: Optional callback function for real-time data processing
            joints_to_process: List of joint names to analyze
            display: Whether to display the video with landmarks
            exercise_type: Type of exercise for model complexity adjustment
        """
        # Reset data
        self.clear_history()
        
        # Update skeleton renderer configuration
        self.skeleton_renderer.config.exercise_type = exercise_type
        self.skeleton_renderer.config.mode = VisualizationMode.REALTIME
        
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
            # Use legacy MediaPipe API
            with self.configure_mediapipe(exercise_type) as pose:
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
                    
                    # Apply smart caching for pose stability
                    if results.pose_landmarks:
                        landmark_visibility = [lm.visibility for lm in results.pose_landmarks.landmark]
                        avg_visibility = sum(landmark_visibility) / len(landmark_visibility)
                        
                        # Update landmark history regardless of visibility
                        self._update_landmark_history(results.pose_landmarks, {})
                        
                        # Keep backward compatibility with previous_landmarks
                        if avg_visibility >= self.config['confidence_threshold']:
                            self.previous_landmarks = results.pose_landmarks
                    
                    # Track current mode for cross-mode consistency
                    if hasattr(results, 'current_mode'):
                        # Handle mode transition
                        if self.current_mode != results.current_mode:
                            # We're transitioning modes, keep using history more heavily
                            old_decay_rate = self.config['confidence_decay_rate']
                            self.config['confidence_decay_rate'] = min(0.95, old_decay_rate * 1.2)
                            
                            # Restore normal decay rate after a few frames
                            def restore_decay_rate():
                                self.config['confidence_decay_rate'] = old_decay_rate
                            # (In a real implementation, you'd need a timer or frame counter to restore)
                            
                        self.current_mode = results.current_mode
                    # Add handling for direct mode change from MovementAnalyzer
                    elif hasattr(self, 'current_mode') and self.current_mode is not None:
                        # This case handles mode changes set directly by MovementAnalyzer
                        pass  # The current_mode is already set, nothing more to do
                    
                    # Draw pose landmarks if display is enabled
                    if display:
                        # Calculate joint angles
                        angles = self.calculate_angles(results, joints_to_process)
                        
                        # Use skeleton renderer for visualization
                        annotated_image = self.skeleton_renderer.render(
                            image,
                            results.pose_landmarks.landmark,
                            angles
                        )
                        cv2.imshow('MediaPipe Pose', annotated_image)
                    
                    # Call user callback if provided
                    if callback:
                        callback(angles, timestamp_ms)
                    
                    # Break on ESC key
                    if cv2.waitKey(5) & 0xFF == 27:
                        break
        
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
    
    def calculate_angles(self, results, joints_to_process: List[str]) -> Dict:
        """Calculate angles for specified joints from pose landmarks."""
        angles = {}
        
        try:
            # Use ONLY pose_landmarks (image-relative coordinates) for visualization consistency
            # NOT pose_world_landmarks which uses a different coordinate system
            landmarks = results.pose_landmarks
            
            # Check if we have landmarks before proceeding
            if landmarks is None or not hasattr(landmarks, 'landmark') or len(landmarks.landmark) == 0:
                # Return empty angles dictionary with default values to avoid errors
                # This ensures detected_joints gets populated even with low confidence data
                return {joint: {'angle': None, 'confidence': 0.1, 'raw_confidence': 0.1} for joint in joints_to_process}
            
            for joint in joints_to_process:
                if joint in self.joint_mappings:
                    try:
                        p1, p2, p3 = self.joint_mappings[joint]
                        
                        # Make sure all required landmarks exist
                        if not all(p.value < len(landmarks.landmark) for p in [p1, p2, p3]):
                            # Add a placeholder entry with low confidence
                            angles[joint] = {
                                'angle': None,
                                'confidence': 0.1,
                                'raw_confidence': 0.1,
                                'error': 'Missing landmark'
                            }
                            continue
                            
                        coords1 = self.get_landmark_coords(landmarks.landmark[p1.value])
                        coords2 = self.get_landmark_coords(landmarks.landmark[p2.value])
                        coords3 = self.get_landmark_coords(landmarks.landmark[p3.value])
                        
                        # Safely get visibility with fallback value
                        def get_visibility(landmark):
                            try:
                                return getattr(landmark, 'visibility', 0.3)
                            except Exception:
                                return 0.3  # Default mid-confidence
                        
                        # Get visibility with gradual decay instead of hard threshold
                        raw_visibility = min(
                            get_visibility(landmarks.landmark[p1.value]),
                            get_visibility(landmarks.landmark[p2.value]),
                            get_visibility(landmarks.landmark[p3.value])
                        )
                        
                        # Apply hysteresis to visibility
                        # If we've had good visibility recently, be more lenient with current frame
                        effective_visibility = max(0.25, raw_visibility)  # Minimum visibility floor to ensure data is captured
                        landmark_indices = [p1.value, p2.value, p3.value]
                        
                        for idx in landmark_indices:
                            if idx in self.frames_since_detection and self.frames_since_detection[idx] < 5:
                                # Boost confidence if we've seen this landmark recently
                                confidence_boost = max(0, 0.15 - (self.frames_since_detection[idx] * 0.03))
                                effective_visibility = max(effective_visibility, raw_visibility + confidence_boost)
                        
                        # Calculate angle even with lower visibility
                        angle = self.calculate_angle(coords1, coords2, coords3)
                        
                        # Check for invalid values
                        if angle is None or math.isnan(angle) or math.isinf(angle):
                            logging.warning(f"Invalid angle calculated for {joint}: {angle}")
                            angles[joint] = {
                                'angle': None,
                                'confidence': effective_visibility,
                                'raw_confidence': raw_visibility,
                                'error': 'Invalid angle calculation'
                            }
                        else:
                            angles[joint] = {
                                'angle': angle,
                                'confidence': effective_visibility,
                                'raw_confidence': raw_visibility
                            }
                    except Exception as e:
                        logging.error(f"Error calculating angle for {joint}: {e}")
                        # Still add to angles dict with error to make sure joint is registered
                        angles[joint] = {
                            'angle': None,
                            'confidence': 0.1,
                            'error': str(e)
                        }
        except Exception as e:
            logging.error(f"Error in calculate_angles: {e}")
            # Return a fallback with default values to avoid empty result
            return {joint: {'angle': None, 'confidence': 0.1, 'error': 'General error'} for joint in joints_to_process}
        
        return angles

    def get_angles_data(self) -> List[Dict]:
        """Get the collected angles data."""
        return self.angles_data
    
    def get_detected_joints(self) -> set:
        """Get the set of detected joints."""
        return self.detected_joints

    def set_mode(self, new_mode: str, previous_mode: str = None):
        """
        Set the current analysis mode and handle transition effects.
        
        Args:
            new_mode: The new analysis mode to set
            previous_mode: Optional previous mode for tracking transitions
        """
        # Only handle transitions when there's a change
        if new_mode != self.current_mode:
            old_decay_rate = self.config.get('confidence_decay_rate', 0.85)
            
            # We're transitioning modes, keep using history more heavily for stability
            self.config['confidence_decay_rate'] = min(0.95, old_decay_rate * 1.2)
            
            # Update the mode
            self.current_mode = new_mode
            
            # In a real implementation, we would need a timer or frame counter
            # to restore the original decay rate after a few frames
            # For now, we'll leave it at the higher rate for mode transitions

    def _update_landmark_history(self, landmarks, confidence_values):
        """
        Update the landmark history with new landmarks and their confidence values.
        Implements confidence-weighted decay for more stable landmark tracking.
        
        Args:
            landmarks: Current frame landmarks
            confidence_values: Dictionary of confidence values for each landmark
        """
        if landmarks is None:
            return
        
        # Check if this is actually landmarks with the expected structure
        if not hasattr(landmarks, 'landmark'):
            # Skip processing for objects that don't have landmarks
            return
            
        try:
            # Get all landmark indices
            landmark_indices = range(len(landmarks.landmark))
            
            for idx in landmark_indices:
                try:
                    landmark = landmarks.landmark[idx]
                    # Safely get visibility with fallback
                    try:
                        confidence = getattr(landmark, 'visibility', 0.5)
                    except:
                        confidence = 0.5  # Default confidence if attribute error
                    
                    # Initialize history for this landmark if it doesn't exist
                    if idx not in self.landmark_history:
                        self.landmark_history[idx] = []
                        self.landmark_confidence_history[idx] = []
                        self.frames_since_detection[idx] = 0
                    
                    # Only add to history if confidence is above minimum threshold
                    # Lower the threshold to be more inclusive and prevent empty detections
                    min_threshold = max(0.2, self.config['min_detection_confidence'] * 0.3)
                    
                    if confidence >= min_threshold:
                        # Add current landmark to history
                        self.landmark_history[idx].append(landmark)
                        self.landmark_confidence_history[idx].append(confidence)
                        self.frames_since_detection[idx] = 0
                        
                        # Limit history size
                        history_size = self.config['landmark_history_size']
                        if len(self.landmark_history[idx]) > history_size:
                            self.landmark_history[idx] = self.landmark_history[idx][-history_size:]
                            self.landmark_confidence_history[idx] = self.landmark_confidence_history[idx][-history_size:]
                    else:
                        # Increment frames since detection
                        self.frames_since_detection[idx] += 1
                except Exception as e:
                    logging.warning(f"Error processing landmark {idx}: {e}")
                    continue
        except Exception as e:
            logging.error(f"Error in _update_landmark_history: {e}")

    def _get_stabilized_landmarks(self, original_landmarks):
        """
        Get stabilized landmarks using history and confidence-weighted averaging.
        
        Args:
            original_landmarks: Original landmarks from pose detection
            
        Returns:
            Stabilized landmarks object
        """
        if original_landmarks is None:
            return None
        
        try:
            # Update history with new landmarks
            self._update_landmark_history(original_landmarks, {})
            
            # For safety, in case we have no history, just return original
            if not self.landmark_history:
                return original_landmarks
                
            # Create a proper copy of the original landmarks
            # We'll need to use a different approach since we can't directly assign to the original
            import copy
            
            # Create a deep copy of the original landmarks
            try:
                stabilized_landmarks = copy.deepcopy(original_landmarks)
                
                # Replace each landmark with a stabilized version in the copy
                for i in range(len(stabilized_landmarks.landmark)):
                    try:
                        stabilized = self._get_stabilized_landmark(i)
                        if stabilized:
                            # Copy the stabilized values into the existing landmark object
                            # This avoids assignment errors if the landmarks are immutable
                            stabilized_landmarks.landmark[i].x = stabilized.x
                            stabilized_landmarks.landmark[i].y = stabilized.y
                            stabilized_landmarks.landmark[i].z = stabilized.z
                            stabilized_landmarks.landmark[i].visibility = stabilized.visibility
                    except:
                        # If we can't update this landmark, keep the original
                        continue
                
                return stabilized_landmarks
            except Exception as e:
                logging.warning(f"Error modifying landmarks: {e}. Using original landmarks.")
                return original_landmarks
        except Exception as e:
            logging.error(f"Error in _get_stabilized_landmarks: {e}")
            return original_landmarks

# Utility function for drawing landmarks with Tasks API
def draw_landmarks_on_image(image, results):
    """Draw pose landmarks on an image using Tasks API results structure."""
    # This implementation would need to be adjusted based on Tasks API output
    # Current implementation is a placeholder
    annotated_image = image.copy()
    
    # Draw landmarks based on Tasks API structure
    # This is a placeholder that would need to be implemented
    
    return annotated_image