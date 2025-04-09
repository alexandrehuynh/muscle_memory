import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Generator, Any
import os
from dataclasses import dataclass
import logging
from datetime import datetime

# Import our core modules
from .mediapipe.pose_estimator import PoseEstimator, PoseEstimatorConfig
from .joint_calculator import JointCalculator
from .visualization.skeleton_visualizer import SkeletonVisualizer, VisualizationConfig, VisualizationMode
from .movement_analyzer import MovementAnalyzer, MovementConfig
from utils.video.processor import VideoProcessor

@dataclass
class PipelineConfig:
    """Configuration for the movement analysis pipeline."""
    # Video processing options
    process_every_n_frames: int = 1
    output_fps: float = 30.0
    
    # Analysis options
    analysis_type: str = "detailed"  # "quick", "detailed", "realtime"
    
    # Pose estimation options
    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    
    # Visualization options
    show_angles: bool = True
    show_reference_lines: bool = True
    
    # Exercise recognition
    auto_detect_exercise: bool = True
    exercise_type: Optional[str] = None

class MovementPipeline:
    """
    MovementPipeline connects all modules together for video analysis.
    
    This class orchestrates the process:
    1. VideoProcessor handles video I/O
    2. PoseEstimator detects poses in frames
    3. JointCalculator calculates joint angles
    4. SkeletonVisualizer renders the skeleton with angles
    5. MovementAnalyzer analyzes the movement patterns
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the movement analysis pipeline.
        
        Args:
            config: Configuration for the pipeline
        """
        self.config = config or PipelineConfig()
        
        # Initialize all modules
        self._init_modules()
        
        # Data storage
        self.angles_data = []
        self.detected_joints = set()
    
    def _init_modules(self):
        """Initialize all component modules."""
        # Video processor is stateless, no need to initialize
        
        # Initialize pose estimator
        pose_config = PoseEstimatorConfig(
            model_complexity=self.config.model_complexity,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            enable_segmentation=self.config.analysis_type == "detailed"
        )
        self.pose_estimator = PoseEstimator(pose_config)
        
        # Initialize joint calculator
        self.joint_calculator = JointCalculator()
        
        # Initialize skeleton visualizer
        viz_mode = VisualizationMode.DETAILED if self.config.analysis_type == "detailed" else VisualizationMode.SIMPLE
        viz_config = VisualizationConfig(
            mode=viz_mode,
            show_angles=self.config.show_angles,
            show_reference_lines=self.config.show_reference_lines
        )
        self.visualizer = SkeletonVisualizer(viz_config)
        
        # Initialize movement analyzer
        movement_config = MovementConfig()
        self.movement_analyzer = MovementAnalyzer(movement_config)
    
    def process_video(self, video_path: str, output_path: Optional[str] = None, 
                     joints_to_process: Optional[List[str]] = None) -> Tuple[bool, str]:
        """
        Process a video to analyze movement.
        
        Args:
            video_path: Path to the input video
            output_path: Path to save the output video (optional)
            joints_to_process: List of joint names to analyze
            
        Returns:
            Tuple of (success, message)
        """
        # Create default output path if not provided
        if output_path is None and video_path:
            base_name = os.path.basename(video_path)
            now = datetime.now()
            date_prefix = now.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join("output", "videos", f"{date_prefix}_{base_name}")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            # Get video info for output writing
            video_info = VideoProcessor.get_video_info(video_path)
            if not video_info:
                return False, "Failed to read video information"
            
            # Create video writer
            width = video_info['width']
            height = video_info['height']
            fps = video_info.get('fps', self.config.output_fps)
            
            video_writer = None
            if output_path:
                video_writer = VideoProcessor.create_video_writer(
                    output_path, width, height, fps
                )
            
            # Process frames
            frame_count = 0
            processed_frames = 0
            
            # Track progress
            total_frames = video_info.get('frame_count', 0)
            logging.info(f"Processing video with {total_frames} frames")
            
            # Clear previous data
            self.angles_data = []
            self.detected_joints = set()
            
            for frame, frame_idx in VideoProcessor.read_video_frames(
                video_path, self.config.process_every_n_frames
            ):
                # Report progress every 100 frames
                if frame_idx % 100 == 0:
                    progress = int(frame_idx / total_frames * 100) if total_frames > 0 else 0
                    logging.info(f"Processing: {progress}% complete")
                
                # Detect landmarks
                landmarks, confidences, success = self.pose_estimator.detect_landmarks(frame)
                
                # If detection successful, calculate angles and visualize
                if success:
                    # Get stabilized landmarks
                    stable_landmarks = self.pose_estimator.get_stabilized_landmarks()
                    
                    # Calculate joint angles
                    angles = self.joint_calculator.calculate_joint_angles(stable_landmarks, joints_to_process)
                    
                    # Store angles data
                    self.angles_data.append(angles)
                    
                    # Update detected joints
                    self.detected_joints.update(angles.keys())
                    
                    # Visualize on frame
                    frame = self.visualizer.visualize(frame, stable_landmarks, angles)
                
                # Write frame to output video
                if video_writer:
                    video_writer.write(frame)
                
                frame_count += 1
                processed_frames += 1
            
            # Release resources
            if video_writer:
                video_writer.release()
            
            # Log completion
            logging.info(f"Completed processing {processed_frames} frames")
            
            # Check if any data was collected
            if not self.angles_data:
                return False, "No pose data could be detected in the video"
            
            return True, f"Successfully processed video with {processed_frames} frames"
            
        except Exception as e:
            logging.error(f"Error processing video: {e}")
            # Make sure to release resources on error
            if 'video_writer' in locals() and video_writer:
                video_writer.release()
            return False, f"Error processing video: {str(e)}"
        finally:
            # Clean up resources
            if 'video_writer' in locals() and video_writer:
                video_writer.release()
    
    def analyze_movement(self, selected_joints: Optional[List[str]] = None) -> Tuple[bool, Dict, str]:
        """
        Analyze the movement data collected during video processing.
        
        Args:
            selected_joints: List of joint names to include in analysis
            
        Returns:
            Tuple of (success, analysis_results, message)
        """
        if not self.angles_data:
            return False, {}, "No data to analyze. Please process a video first."
        
        # Filter to selected joints if specified
        if selected_joints:
            selected_joints = [j for j in selected_joints if j in self.detected_joints]
        else:
            selected_joints = list(self.detected_joints)
        
        if not selected_joints:
            return False, {}, "No valid joints selected for analysis"
        
        try:
            # Get exercise type if specified, otherwise auto-detect
            exercise_type = self.config.exercise_type
            
            # Analyze movement
            analysis_results = self.movement_analyzer.analyze_movement(
                self.angles_data, exercise_type
            )
            
            return True, analysis_results, "Analysis completed successfully"
            
        except Exception as e:
            logging.error(f"Error analyzing movement: {e}")
            return False, {}, f"Error analyzing movement: {str(e)}"
    
    def process_image(self, image_path: str, output_path: Optional[str] = None,
                     joints_to_process: Optional[List[str]] = None) -> Tuple[bool, str]:
        """
        Process a single image for pose analysis.
        
        Args:
            image_path: Path to the input image
            output_path: Path to save the output image (optional)
            joints_to_process: List of joint names to analyze
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                return False, f"Failed to read image from {image_path}"
            
            # Create default output path if not provided
            if output_path is None and image_path:
                base_name = os.path.basename(image_path)
                now = datetime.now()
                date_prefix = now.strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join("output", "images", f"{date_prefix}_{base_name}")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Detect landmarks
            landmarks, confidences, success = self.pose_estimator.detect_landmarks(image)
            
            # If detection successful, calculate angles and visualize
            if success:
                # Calculate joint angles
                angles = self.joint_calculator.calculate_joint_angles(landmarks, joints_to_process)
                
                # Store angles data
                self.angles_data = [angles]
                
                # Update detected joints
                self.detected_joints = set(angles.keys())
                
                # Visualize on image
                image = self.visualizer.visualize(image, landmarks, angles)
                
                # Save the output image
                if output_path:
                    cv2.imwrite(output_path, image)
                
                return True, f"Successfully processed image"
            else:
                return False, "No pose could be detected in the image"
                
        except Exception as e:
            logging.error(f"Error processing image: {e}")
            return False, f"Error processing image: {str(e)}"
    
    def process_webcam(self, camera_id: int = 0, display: bool = True,
                      record_output: bool = False, output_path: Optional[str] = None,
                      joints_to_process: Optional[List[str]] = None) -> None:
        """
        Process webcam feed for real-time pose analysis.
        
        Args:
            camera_id: Camera device ID
            display: Whether to display the output
            record_output: Whether to record the output video
            output_path: Path to save the output video (optional)
            joints_to_process: List of joint names to analyze
        """
        try:
            # Open webcam
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                logging.error(f"Could not open webcam {camera_id}")
                return
            
            # Get webcam properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Create video writer if recording
            video_writer = None
            if record_output:
                if output_path is None:
                    now = datetime.now()
                    date_prefix = now.strftime("%Y%m%d_%H%M%S")
                    output_path = os.path.join("output", "videos", f"webcam_{date_prefix}.mp4")
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                video_writer = VideoProcessor.create_video_writer(
                    output_path, width, height, fps
                )
            
            # Clear previous data
            self.angles_data = []
            self.detected_joints = set()
            
            frame_count = 0
            
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames based on config
                if frame_count % self.config.process_every_n_frames != 0:
                    frame_count += 1
                    continue
                
                # Detect landmarks
                landmarks, confidences, success = self.pose_estimator.detect_landmarks(frame)
                
                # If detection successful, calculate angles and visualize
                if success:
                    # Get stabilized landmarks
                    stable_landmarks = self.pose_estimator.get_stabilized_landmarks()
                    
                    # Calculate joint angles
                    angles = self.joint_calculator.calculate_joint_angles(stable_landmarks, joints_to_process)
                    
                    # Store angles data (keep only the most recent 30 frames for analysis)
                    self.angles_data.append(angles)
                    if len(self.angles_data) > 30:
                        self.angles_data.pop(0)
                    
                    # Update detected joints
                    self.detected_joints.update(angles.keys())
                    
                    # Visualize on frame
                    frame = self.visualizer.visualize(frame, stable_landmarks, angles)
                
                # Display the frame
                if display:
                    cv2.imshow('Pose Analysis (Press ESC to exit)', frame)
                
                # Write frame to output video
                if video_writer:
                    video_writer.write(frame)
                
                # Check for exit key
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                    break
                
                frame_count += 1
            
            # Release resources
            cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            logging.error(f"Error processing webcam feed: {e}")
            # Make sure to release resources on error
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            if 'video_writer' in locals() and video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
    
    def release(self):
        """Release resources held by the pipeline."""
        self.pose_estimator.release()
        self.visualizer.release()
        # Reset data
        self.angles_data = []
        self.detected_joints = set() 