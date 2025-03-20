import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Optional, Tuple, Union
import json
from datetime import datetime
from .mediapipe.pose_analyzer import PoseAnalyzer

class MovementAnalyzer:
    """
    High-level class for analyzing movements using PoseAnalyzer.
    This class handles:
    - Processing videos/images
    - Analyzing joint angles
    - Generating statistics and visualizations
    - Detecting exercise types
    """
    
    def __init__(self, model_complexity: int = 2):
        """
        Initialize the MovementAnalyzer.
        
        Args:
            model_complexity: MediaPipe model complexity (0, 1, or 2)
        """
        self.pose_analyzer = PoseAnalyzer(model_complexity=model_complexity)
        self.exercise_types = {
            'squat': self._detect_squat,
            'lunge': self._detect_lunge,
            'pushup': self._detect_pushup
        }
    
    def process_video(self, video_path: str, output_path: str = None, 
                      joints_to_process: List[str] = None) -> Tuple[bool, str]:
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
            output_path = os.path.join("output", f"analyzed_{base_name}")
            os.makedirs("output", exist_ok=True)
        
        # Process video and collect progress updates
        progress_generator = self.pose_analyzer.process_video(video_path, output_path, joints_to_process)
        
        # Process all frames (in a real application, you might want to handle progress differently)
        progress = 0
        for progress in progress_generator:
            pass  # In a real app, update UI with progress
        
        # Check if any joints were detected
        if not self.pose_analyzer.detected_joints:
            return False, "No joints were reliably detected in the video"
        
        return True, f"Successfully processed video. {len(self.pose_analyzer.detected_joints)} joints detected."
    
    def analyze_movement(self, selected_joints: List[str] = None) -> Tuple[bool, Dict, str]:
        """
        Analyze the movement data collected by the pose analyzer.
        
        Args:
            selected_joints: List of joint names to include in analysis
            
        Returns:
            Tuple of (success, analysis_results, message)
        """
        if not self.pose_analyzer.angles_data:
            return False, {}, "No data to analyze. Please process a video first."
        
        # Convert data to DataFrame
        angles_data = self.pose_analyzer.angles_data
        detected_joints = self.pose_analyzer.detected_joints
        
        # Filter selected joints
        if selected_joints:
            selected_joints = [j for j in selected_joints if j in detected_joints]
        else:
            selected_joints = list(detected_joints)
        
        if not selected_joints:
            return False, {}, "No valid joints selected for analysis"
        
        # Create DataFrames for angle data and confidence
        angle_values = {}
        confidence_values = {}
        
        for joint in selected_joints:
            angle_values[joint] = [d.get(joint, {}).get('angle') for d in angles_data]
            confidence_values[joint] = [d.get(joint, {}).get('confidence', 0) for d in angles_data]
        
        angle_df = pd.DataFrame(angle_values)
        confidence_df = pd.DataFrame(confidence_values)
        
        # Filter out low confidence values
        filtered_angle_df = angle_df.where(confidence_df > 0.5)
        
        # Calculate statistics
        stats = filtered_angle_df.agg(['mean', 'min', 'max', 'std']).round(2)
        
        # Detect exercise type
        exercise_type = self._detect_exercise_type(filtered_angle_df)
        
        # Calculate movement metrics
        metrics = self._calculate_movement_metrics(filtered_angle_df, exercise_type)
        
        # Combine results
        analysis_results = {
            'statistics': stats.to_dict(),
            'exercise_type': exercise_type,
            'metrics': metrics,
            'joint_angles': {
                joint: filtered_angle_df[joint].dropna().tolist() for joint in selected_joints
            }
        }
        
        return True, analysis_results, "Analysis completed successfully"
    
    def _detect_exercise_type(self, angle_df: pd.DataFrame) -> str:
        """
        Detect the type of exercise from joint angle patterns.
        
        Args:
            angle_df: DataFrame of joint angles
            
        Returns:
            Exercise type name or 'unknown'
        """
        for exercise_name, detector_func in self.exercise_types.items():
            if detector_func(angle_df):
                return exercise_name
        
        return "unknown"
    
    def _detect_squat(self, angle_df: pd.DataFrame) -> bool:
        """Detect if the movement is a squat."""
        # Check if we have knee angle data
        if 'LEFT_KNEE' not in angle_df.columns and 'RIGHT_KNEE' not in angle_df.columns:
            return False
        
        # Use either knee depending on what's available
        knee_col = 'LEFT_KNEE' if 'LEFT_KNEE' in angle_df.columns else 'RIGHT_KNEE'
        
        knee_angles = angle_df[knee_col].dropna()
        if knee_angles.empty:
            return False
        
        # Check for characteristic squat pattern
        # 1. Starting with mostly straight legs
        # 2. Significant bend in the middle
        # 3. Return to straight
        if len(knee_angles) < 10:
            return False
        
        max_angle = knee_angles.max()
        min_angle = knee_angles.min()
        
        # Check for a significant bend (at least 30 degrees)
        angle_range = max_angle - min_angle
        
        return angle_range > 30 and min_angle < 120
    
    def _detect_lunge(self, angle_df: pd.DataFrame) -> bool:
        """Detect if the movement is a lunge."""
        # Simplified lunge detection
        # A proper implementation would look at the pattern of both knees
        if 'LEFT_KNEE' not in angle_df.columns or 'RIGHT_KNEE' not in angle_df.columns:
            return False
        
        left_knee = angle_df['LEFT_KNEE'].dropna()
        right_knee = angle_df['RIGHT_KNEE'].dropna()
        
        if left_knee.empty or right_knee.empty:
            return False
        
        # In a lunge, typically one knee bends deeply while the other less so
        left_min = left_knee.min()
        right_min = right_knee.min()
        
        angle_diff = abs(left_min - right_min)
        
        return angle_diff > 20 and (left_min < 120 or right_min < 120)
    
    def _detect_pushup(self, angle_df: pd.DataFrame) -> bool:
        """Detect if the movement is a push-up."""
        if 'LEFT_ELBOW' not in angle_df.columns and 'RIGHT_ELBOW' not in angle_df.columns:
            return False
        
        elbow_col = 'LEFT_ELBOW' if 'LEFT_ELBOW' in angle_df.columns else 'RIGHT_ELBOW'
        
        elbow_angles = angle_df[elbow_col].dropna()
        if elbow_angles.empty:
            return False
        
        # Check for characteristic pushup pattern:
        # 1. Starting with extended arms
        # 2. Significant bend in elbows
        # 3. Return to extended
        if len(elbow_angles) < 10:
            return False
        
        max_angle = elbow_angles.max()
        min_angle = elbow_angles.min()
        
        # Significant elbow flexion should occur
        angle_range = max_angle - min_angle
        
        return angle_range > 40 and min_angle < 100
    
    def _calculate_movement_metrics(self, angle_df: pd.DataFrame, exercise_type: str) -> Dict:
        """
        Calculate metrics specific to the detected exercise type.
        
        Args:
            angle_df: DataFrame of joint angles
            exercise_type: Detected exercise type
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'symmetry_score': self._calculate_symmetry(angle_df),
            'smoothness_score': self._calculate_smoothness(angle_df),
            'range_of_motion_score': self._calculate_range_of_motion(angle_df, exercise_type),
        }
        
        # Add exercise-specific metrics
        if exercise_type == 'squat':
            metrics.update(self._calculate_squat_metrics(angle_df))
        elif exercise_type == 'lunge':
            metrics.update(self._calculate_lunge_metrics(angle_df))
        elif exercise_type == 'pushup':
            metrics.update(self._calculate_pushup_metrics(angle_df))
        
        return metrics
    
    def _calculate_symmetry(self, angle_df: pd.DataFrame) -> float:
        """Calculate bilateral symmetry score."""
        symmetry_pairs = [
            ('LEFT_SHOULDER', 'RIGHT_SHOULDER'),
            ('LEFT_ELBOW', 'RIGHT_ELBOW'),
            ('LEFT_HIP', 'RIGHT_HIP'),
            ('LEFT_KNEE', 'RIGHT_KNEE'),
            ('LEFT_ANKLE', 'RIGHT_ANKLE')
        ]
        
        symmetry_scores = []
        
        for left, right in symmetry_pairs:
            if left in angle_df.columns and right in angle_df.columns:
                left_data = angle_df[left].dropna()
                right_data = angle_df[right].dropna()
                
                if not left_data.empty and not right_data.empty:
                    # Interpolate to make lengths match
                    if len(left_data) != len(right_data):
                        # Create a common index range
                        common_length = min(len(left_data), len(right_data))
                        left_data = left_data.iloc[:common_length]
                        right_data = right_data.iloc[:common_length]
                    
                    # Calculate difference between left and right
                    diff = abs(left_data - right_data)
                    avg_diff = diff.mean()
                    
                    # Convert to a 0-100 score (0 = completely asymmetrical, 100 = perfect symmetry)
                    # Assuming a difference of 30 degrees or more is major asymmetry
                    pair_score = max(0, 100 - (avg_diff * 100 / 30))
                    symmetry_scores.append(pair_score)
        
        # Return average symmetry score, or 0 if no pairs were analyzed
        return round(np.mean(symmetry_scores)) if symmetry_scores else 0
    
    def _calculate_smoothness(self, angle_df: pd.DataFrame) -> float:
        """Calculate movement smoothness score based on angle variations."""
        smoothness_scores = []
        
        for joint in angle_df.columns:
            data = angle_df[joint].dropna()
            if len(data) > 10:  # Need enough data points
                # Calculate the second derivative (acceleration) of the angle
                # First, calculate first differences (velocity)
                velocity = data.diff().dropna()
                # Then calculate second differences (acceleration)
                acceleration = velocity.diff().dropna().abs()
                
                # Smoothness is inversely related to acceleration peaks
                # Calculate the 90th percentile of acceleration
                acc_threshold = acceleration.quantile(0.9)
                
                # Scale to a 0-100 score (lower acceleration = smoother)
                # A threshold of 10 degrees/frame² is considered high jerkiness
                smoothness = max(0, 100 - (acc_threshold * 100 / 10))
                
                smoothness_scores.append(smoothness)
        
        # Return average smoothness score, or 0 if no joints were analyzed
        return round(np.mean(smoothness_scores)) if smoothness_scores else 0
    
    def _calculate_range_of_motion(self, angle_df: pd.DataFrame, exercise_type: str) -> float:
        """Calculate range of motion score based on exercise type."""
        if exercise_type == 'squat':
            # For squats, focus on knee angles
            key_joints = ['LEFT_KNEE', 'RIGHT_KNEE']
        elif exercise_type == 'lunge':
            # For lunges, focus on both knees
            key_joints = ['LEFT_KNEE', 'RIGHT_KNEE']
        elif exercise_type == 'pushup':
            # For pushups, focus on elbow angles
            key_joints = ['LEFT_ELBOW', 'RIGHT_ELBOW']
        else:
            # For unknown exercises, consider all major joints
            key_joints = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
                          'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE']
        
        # Keep only available joints
        key_joints = [j for j in key_joints if j in angle_df.columns]
        
        if not key_joints:
            return 0
        
        rom_scores = []
        for joint in key_joints:
            data = angle_df[joint].dropna()
            if not data.empty:
                # Calculate the range of motion
                angle_range = data.max() - data.min()
                
                # Define expected ranges for different joints and exercises
                if 'KNEE' in joint and exercise_type == 'squat':
                    # For squats, expect 80-120 degree ROM in knees
                    expected_range = 100
                    rom_score = min(100, (angle_range / expected_range) * 100)
                elif 'KNEE' in joint and exercise_type == 'lunge':
                    # For lunges, expect 60-90 degree ROM in knees
                    expected_range = 75
                    rom_score = min(100, (angle_range / expected_range) * 100)
                elif 'ELBOW' in joint and exercise_type == 'pushup':
                    # For pushups, expect 60-90 degree ROM in elbows
                    expected_range = 75
                    rom_score = min(100, (angle_range / expected_range) * 100)
                else:
                    # Generic scoring for other joints
                    expected_range = 60
                    rom_score = min(100, (angle_range / expected_range) * 100)
                
                rom_scores.append(rom_score)
        
        return round(np.mean(rom_scores)) if rom_scores else 0
    
    def _calculate_squat_metrics(self, angle_df: pd.DataFrame) -> Dict:
        """Calculate squat-specific metrics."""
        metrics = {}
        
        # Check for knee and hip angles
        knee_cols = [col for col in angle_df.columns if 'KNEE' in col]
        hip_cols = [col for col in angle_df.columns if 'HIP' in col]
        
        if knee_cols and hip_cols:
            # Get the min angle for knees (deepest point of squat)
            knee_min = min(angle_df[col].min() for col in knee_cols if not angle_df[col].dropna().empty)
            
            # Check if knees go too low (potential strain)
            if knee_min < 70:
                metrics['knee_safety'] = 'Warning: Knee angle too acute at bottom of squat'
            else:
                metrics['knee_safety'] = 'Good knee angle maintained'
            
            # Calculate depth score (how deep the squat is)
            # A full squat would have knee angle around 70-90 degrees
            if knee_min > 100:
                metrics['depth_score'] = 'Partial'
            elif knee_min > 90:
                metrics['depth_score'] = 'Moderate'
            else:
                metrics['depth_score'] = 'Deep'
        
        return metrics
    
    def _calculate_lunge_metrics(self, angle_df: pd.DataFrame) -> Dict:
        """Calculate lunge-specific metrics."""
        metrics = {}
        
        # Check for knee angles
        left_knee = 'LEFT_KNEE' if 'LEFT_KNEE' in angle_df.columns else None
        right_knee = 'RIGHT_KNEE' if 'RIGHT_KNEE' in angle_df.columns else None
        
        if left_knee and right_knee:
            left_min = angle_df[left_knee].dropna().min() if not angle_df[left_knee].dropna().empty else 180
            right_min = angle_df[right_knee].dropna().min() if not angle_df[right_knee].dropna().empty else 180
            
            # Determine which leg is forward
            if left_min < right_min:
                forward_leg = 'left'
                forward_knee_angle = left_min
            else:
                forward_leg = 'right'
                forward_knee_angle = right_min
            
            metrics['forward_leg'] = forward_leg
            
            # Check forward knee angle (should be around 90 degrees)
            angle_diff = abs(forward_knee_angle - 90)
            if angle_diff < 15:
                metrics['form_quality'] = 'Good'
            elif angle_diff < 30:
                metrics['form_quality'] = 'Moderate'
            else:
                metrics['form_quality'] = 'Needs improvement'
        
        return metrics
    
    def _calculate_pushup_metrics(self, angle_df: pd.DataFrame) -> Dict:
        """Calculate pushup-specific metrics."""
        metrics = {}
        
        # Check for elbow angles
        elbow_cols = [col for col in angle_df.columns if 'ELBOW' in col]
        
        if elbow_cols:
            # Get the min angle for elbows (lowest point of pushup)
            elbow_min = min(angle_df[col].min() for col in elbow_cols if not angle_df[col].dropna().empty)
            
            # Calculate depth score
            # A full pushup would have elbow angle around 70-90 degrees
            if elbow_min > 100:
                metrics['depth_score'] = 'Partial'
            elif elbow_min > 90:
                metrics['depth_score'] = 'Moderate'
            else:
                metrics['depth_score'] = 'Deep'
        
        return metrics
    
    def generate_plots(self, output_dir: str = 'output/plots') -> List[str]:
        """
        Generate plots visualizing the movement data.
        
        Args:
            output_dir: Directory to save generated plots
            
        Returns:
            List of paths to generated plot files
        """
        if not self.pose_analyzer.angles_data:
            return []
        
        os.makedirs(output_dir, exist_ok=True)
        plot_paths = []
        
        # Convert data to DataFrame
        angles_data = self.pose_analyzer.angles_data
        detected_joints = self.pose_analyzer.detected_joints
        
        # Extract angle values from the complex dictionary structure
        angle_values = {}
        for joint in detected_joints:
            angle_values[joint] = [d.get(joint, {}).get('angle') for d in angles_data]
        
        angle_df = pd.DataFrame(angle_values)
        
        # Generate time series plot
        try:
            plt.figure(figsize=(12, 6))
            for column in angle_df.columns:
                plt.plot(angle_df[column].rolling(window=5).mean(), label=column)
            
            plt.legend()
            plt.title("Joint Angle Variations Over Time (5-frame moving average)")
            plt.xlabel("Frame")
            plt.ylabel("Angle (degrees)")
            plt.grid(True, alpha=0.3)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(output_dir, f"angle_plot_{timestamp}.png")
            plt.savefig(plot_path)
            plt.close()
            
            plot_paths.append(plot_path)
        except Exception as e:
            print(f"Error creating time series plot: {e}")
        
        # Generate box plot for range distribution
        try:
            plt.figure(figsize=(12, 6))
            angle_df.boxplot()
            plt.title("Joint Angle Distributions")
            plt.ylabel("Angle (degrees)")
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            plot_path = os.path.join(output_dir, f"angle_distribution_{timestamp}.png")
            plt.savefig(plot_path)
            plt.close()
            
            plot_paths.append(plot_path)
        except Exception as e:
            print(f"Error creating distribution plot: {e}")
        
        return plot_paths