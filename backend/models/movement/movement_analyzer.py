import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import math
from dataclasses import dataclass
import logging

@dataclass
class MovementConfig:
    """Configuration for movement analysis."""
    min_confidence: float = 0.5
    min_range_of_motion: float = 20.0
    min_repetitions: int = 3
    smoothness_window: int = 5  # Window size for smoothness calculation
    symmetry_threshold: float = 15.0  # Maximum angle difference for good symmetry

class MovementAnalyzer:
    """
    MovementAnalyzer handles ONLY the analysis of movement patterns and exercise classification.
    
    Responsibilities:
    - Exercise classification
    - Range of motion analysis
    - Symmetry analysis
    - Movement quality assessment
    
    This class does NOT handle:
    - Pose detection
    - Joint angle calculations
    - Video processing
    - Visualization
    """
    
    def __init__(self, config: Optional[MovementConfig] = None):
        """
        Initialize the movement analyzer.
        
        Args:
            config: Configuration for movement analysis
        """
        self.config = config or MovementConfig()
        
        # Dictionary of exercise detection functions
        self.exercise_detectors = {
            'squat': self._detect_squat,
            'lunge': self._detect_lunge,
            'pushup': self._detect_pushup
        }
        
        # Dictionary of exercise-specific analysis functions
        self.exercise_analyzers = {
            'squat': self._analyze_squat,
            'lunge': self._analyze_lunge,
            'pushup': self._analyze_pushup
        }
    
    def classify_exercise(self, angles_df: pd.DataFrame) -> str:
        """
        Classify the type of exercise based on joint angle patterns.
        
        Args:
            angles_df: DataFrame of joint angles over time
            
        Returns:
            Exercise type or 'unknown'
        """
        # Check each exercise type
        for exercise_name, detector_func in self.exercise_detectors.items():
            if detector_func(angles_df):
                return exercise_name
        
        return 'unknown'
    
    def _detect_squat(self, angles_df: pd.DataFrame) -> bool:
        """
        Detect if the movement is a squat.
        
        Args:
            angles_df: DataFrame of joint angles over time
            
        Returns:
            True if the movement is a squat, False otherwise
        """
        # Squats primarily involve knee flexion and extension
        if 'LEFT_KNEE' not in angles_df.columns and 'RIGHT_KNEE' not in angles_df.columns:
            return False
            
        # Calculate range of motion for knees
        knee_rom = 0
        
        if 'LEFT_KNEE' in angles_df.columns:
            left_knee = angles_df['LEFT_KNEE'].dropna()
            if len(left_knee) > 10:  # Need enough data points
                left_rom = left_knee.max() - left_knee.min()
                knee_rom = max(knee_rom, left_rom)
                
        if 'RIGHT_KNEE' in angles_df.columns:
            right_knee = angles_df['RIGHT_KNEE'].dropna()
            if len(right_knee) > 10:  # Need enough data points
                right_rom = right_knee.max() - right_knee.min()
                knee_rom = max(knee_rom, right_rom)
        
        # Also check hip angles
        hip_rom = 0
        
        if 'LEFT_HIP' in angles_df.columns:
            left_hip = angles_df['LEFT_HIP'].dropna()
            if len(left_hip) > 10:
                left_rom = left_hip.max() - left_hip.min()
                hip_rom = max(hip_rom, left_rom)
                
        if 'RIGHT_HIP' in angles_df.columns:
            right_hip = angles_df['RIGHT_HIP'].dropna()
            if len(right_hip) > 10:
                right_rom = right_hip.max() - right_hip.min()
                hip_rom = max(hip_rom, right_rom)
        
        # Criteria for squat:
        # 1. Significant knee flexion and extension (ROM > 30 degrees)
        # 2. Significant hip flexion and extension
        # 3. Vertical movement pattern
        
        knee_criterion = knee_rom > 30
        hip_criterion = hip_rom > 30
        
        return knee_criterion and hip_criterion
    
    def _detect_lunge(self, angles_df: pd.DataFrame) -> bool:
        """
        Detect if the movement is a lunge.
        
        Args:
            angles_df: DataFrame of joint angles over time
            
        Returns:
            True if the movement is a lunge, False otherwise
        """
        # Lunges involve knee flexion with asymmetry between legs
        if 'LEFT_KNEE' not in angles_df.columns or 'RIGHT_KNEE' not in angles_df.columns:
            return False
            
        left_knee = angles_df['LEFT_KNEE'].dropna()
        right_knee = angles_df['RIGHT_KNEE'].dropna()
        
        if len(left_knee) < 10 or len(right_knee) < 10:
            return False
            
        # Calculate range of motion
        left_rom = left_knee.max() - left_knee.min()
        right_rom = right_knee.max() - right_knee.min()
        
        # Calculate asymmetry between legs
        avg_left = left_knee.mean()
        avg_right = right_knee.mean()
        asymmetry = abs(avg_left - avg_right)
        
        # Criteria for lunge:
        # 1. Significant knee flexion in at least one leg
        # 2. Asymmetry between legs
        rom_criterion = left_rom > 40 or right_rom > 40
        asymmetry_criterion = asymmetry > 15
        
        return rom_criterion and asymmetry_criterion
    
    def _detect_pushup(self, angles_df: pd.DataFrame) -> bool:
        """
        Detect if the movement is a pushup.
        
        Args:
            angles_df: DataFrame of joint angles over time
            
        Returns:
            True if the movement is a pushup, False otherwise
        """
        # Pushups primarily involve elbow flexion and extension
        if 'LEFT_ELBOW' not in angles_df.columns and 'RIGHT_ELBOW' not in angles_df.columns:
            return False
            
        # Calculate range of motion for elbows
        elbow_rom = 0
        
        if 'LEFT_ELBOW' in angles_df.columns:
            left_elbow = angles_df['LEFT_ELBOW'].dropna()
            if len(left_elbow) > 10:
                left_rom = left_elbow.max() - left_elbow.min()
                elbow_rom = max(elbow_rom, left_rom)
                
        if 'RIGHT_ELBOW' in angles_df.columns:
            right_elbow = angles_df['RIGHT_ELBOW'].dropna()
            if len(right_elbow) > 10:
                right_rom = right_elbow.max() - right_elbow.min()
                elbow_rom = max(elbow_rom, right_rom)
        
        # Criteria for pushup:
        # 1. Significant elbow flexion and extension (ROM > 45 degrees)
        # 2. Limited movement in lower body joints
        
        elbow_criterion = elbow_rom > 45
        
        # Check that lower body joints aren't moving much
        lower_body_movement = 0
        
        for joint in ['LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE']:
            if joint in angles_df.columns:
                joint_data = angles_df[joint].dropna()
                if len(joint_data) > 10:
                    joint_rom = joint_data.max() - joint_data.min()
                    lower_body_movement = max(lower_body_movement, joint_rom)
        
        lower_body_criterion = lower_body_movement < 20
        
        return elbow_criterion and lower_body_criterion
    
    def analyze_movement(self, angles_data: List[Dict], exercise_type: Optional[str] = None) -> Dict:
        """
        Analyze movement patterns in the joint angle data.
        
        Args:
            angles_data: List of dictionaries containing joint angles
            exercise_type: Optional exercise type (if already known)
            
        Returns:
            Dictionary of analysis results
        """
        # Convert angles_data list to pandas DataFrame
        joint_names = set()
        for frame_data in angles_data:
            joint_names.update(frame_data.keys())
        
        # Create DataFrames for angle data and confidence
        angle_values = {}
        confidence_values = {}
        
        for joint in joint_names:
            angle_values[joint] = []
            confidence_values[joint] = []
            
            for frame_data in angles_data:
                if joint in frame_data:
                    angle_values[joint].append(frame_data[joint].get('angle', None))
                    confidence_values[joint].append(frame_data[joint].get('confidence', 0))
                else:
                    angle_values[joint].append(None)
                    confidence_values[joint].append(0)
        
        angles_df = pd.DataFrame(angle_values)
        confidence_df = pd.DataFrame(confidence_values)
        
        # Filter out low confidence values
        filtered_df = angles_df.copy()
        for joint in angles_df.columns:
            mask = confidence_df[joint] < self.config.min_confidence
            filtered_df.loc[mask, joint] = np.nan
        
        # Detect exercise type if not provided
        if exercise_type is None or exercise_type == 'unknown':
            exercise_type = self.classify_exercise(filtered_df)
        
        # Calculate general metrics
        results = {
            'exercise_type': exercise_type,
            'metrics': {
                'symmetry': self._calculate_symmetry(filtered_df),
                'smoothness': self._calculate_smoothness(filtered_df),
                'range_of_motion': self._calculate_range_of_motion(filtered_df, exercise_type),
                'repetition_count': self._count_repetitions(filtered_df, exercise_type)
            },
            'statistics': self._calculate_statistics(filtered_df)
        }
        
        # Calculate exercise-specific metrics
        if exercise_type in self.exercise_analyzers:
            results['metrics'].update(self.exercise_analyzers[exercise_type](filtered_df))
        
        return results
    
    def _calculate_statistics(self, angles_df: pd.DataFrame) -> Dict:
        """
        Calculate statistics for each joint.
        
        Args:
            angles_df: DataFrame of joint angles
            
        Returns:
            Dictionary of statistics
        """
        # Calculate mean, min, max, and standard deviation for each joint
        stats = {}
        
        for joint in angles_df.columns:
            joint_data = angles_df[joint].dropna()
            if len(joint_data) > 0:
                stats[joint] = {
                    'mean': float(joint_data.mean()),
                    'min': float(joint_data.min()),
                    'max': float(joint_data.max()),
                    'std': float(joint_data.std()),
                    'range': float(joint_data.max() - joint_data.min())
                }
        
        return stats
    
    def _calculate_symmetry(self, angles_df: pd.DataFrame) -> float:
        """
        Calculate symmetry between left and right sides of the body.
        
        Args:
            angles_df: DataFrame of joint angles
            
        Returns:
            Symmetry score (0-100)
        """
        # Identify corresponding left/right joint pairs
        joint_pairs = [
            ('LEFT_SHOULDER', 'RIGHT_SHOULDER'),
            ('LEFT_ELBOW', 'RIGHT_ELBOW'),
            ('LEFT_HIP', 'RIGHT_HIP'),
            ('LEFT_KNEE', 'RIGHT_KNEE'),
            ('LEFT_ANKLE', 'RIGHT_ANKLE')
        ]
        
        symmetry_scores = []
        
        for left_joint, right_joint in joint_pairs:
            if left_joint in angles_df.columns and right_joint in angles_df.columns:
                left_data = angles_df[left_joint].dropna()
                right_data = angles_df[right_joint].dropna()
                
                if len(left_data) > 10 and len(right_data) > 10:
                    # Calculate differences in patterns
                    left_mean = left_data.mean()
                    right_mean = right_data.mean()
                    left_range = left_data.max() - left_data.min()
                    right_range = right_data.max() - right_data.min()
                    
                    # Calculate symmetry scores for this joint pair
                    mean_diff = abs(left_mean - right_mean)
                    range_diff = abs(left_range - right_range)
                    
                    # Convert to a 0-100 score (100 is perfect symmetry)
                    mean_score = max(0, 100 - (mean_diff * 5))  # Deduct 5 points per degree of difference
                    range_score = max(0, 100 - (range_diff * 3))  # Deduct 3 points per degree of ROM difference
                    
                    # Average the scores
                    joint_symmetry = (mean_score + range_score) / 2
                    symmetry_scores.append(joint_symmetry)
        
        # Return overall symmetry score
        if symmetry_scores:
            return float(sum(symmetry_scores) / len(symmetry_scores))
        else:
            return 0.0
    
    def _calculate_smoothness(self, angles_df: pd.DataFrame) -> float:
        """
        Calculate movement smoothness based on angle velocity variation.
        
        Args:
            angles_df: DataFrame of joint angles
            
        Returns:
            Smoothness score (0-100)
        """
        smoothness_scores = []
        
        for joint in angles_df.columns:
            joint_data = angles_df[joint].dropna()
            
            if len(joint_data) > self.config.smoothness_window:
                # Calculate first derivative (velocity)
                velocity = joint_data.diff().fillna(0)
                
                # Calculate standard deviation of velocity
                velocity_std = velocity.std()
                
                # Calculate jerk (rate of change of velocity)
                jerk = velocity.diff().fillna(0)
                jerk_magnitude = jerk.abs().mean()
                
                # Convert to smoothness score (100 is perfectly smooth)
                # Lower jerk and velocity variation = smoother movement
                if velocity_std > 0:
                    jerk_score = max(0, 100 - (jerk_magnitude * 10))
                    var_score = max(0, 100 - (velocity_std * 5))
                    
                    joint_smoothness = (jerk_score + var_score) / 2
                    smoothness_scores.append(joint_smoothness)
        
        # Return overall smoothness score
        if smoothness_scores:
            return float(sum(smoothness_scores) / len(smoothness_scores))
        else:
            return 0.0
    
    def _calculate_range_of_motion(self, angles_df: pd.DataFrame, exercise_type: str) -> float:
        """
        Calculate range of motion score for the primary joints in the exercise.
        
        Args:
            angles_df: DataFrame of joint angles
            exercise_type: Type of exercise
            
        Returns:
            Range of motion score (0-100)
        """
        # Define primary joints for each exercise type
        primary_joints = {
            'squat': ['LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_HIP', 'RIGHT_HIP'],
            'lunge': ['LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_HIP', 'RIGHT_HIP'],
            'pushup': ['LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_SHOULDER', 'RIGHT_SHOULDER']
        }
        
        # Get primary joints for this exercise
        joints_to_check = primary_joints.get(exercise_type, angles_df.columns)
        
        rom_values = []
        
        for joint in joints_to_check:
            if joint in angles_df.columns:
                joint_data = angles_df[joint].dropna()
                if len(joint_data) > 10:
                    # Calculate ROM
                    joint_rom = joint_data.max() - joint_data.min()
                    rom_values.append(joint_rom)
        
        # If no ROM values, return 0
        if not rom_values:
            return 0.0
            
        # Calculate average ROM
        avg_rom = sum(rom_values) / len(rom_values)
        
        # Convert to a score (0-100)
        # Calibrate based on expected ROM for the exercise
        expected_rom = {
            'squat': 90,   # Expect 90 degrees ROM for squats
            'lunge': 80,   # Expect 80 degrees ROM for lunges
            'pushup': 70,  # Expect 70 degrees ROM for pushups
            'unknown': 45  # Default expected ROM
        }
        
        target_rom = expected_rom.get(exercise_type, 45)
        
        # Score based on how close to target ROM
        if avg_rom >= target_rom:
            rom_score = 100  # Full score if at or above target
        else:
            # Proportional score based on percentage of target achieved
            rom_score = (avg_rom / target_rom) * 100
        
        return float(rom_score)
    
    def _count_repetitions(self, angles_df: pd.DataFrame, exercise_type: str) -> int:
        """
        Count the number of repetitions in the movement.
        
        Args:
            angles_df: DataFrame of joint angles
            exercise_type: Type of exercise
            
        Returns:
            Number of repetitions
        """
        # Define primary joints for counting repetitions
        primary_joints = {
            'squat': ['LEFT_KNEE', 'RIGHT_KNEE'],
            'lunge': ['LEFT_KNEE', 'RIGHT_KNEE'],
            'pushup': ['LEFT_ELBOW', 'RIGHT_ELBOW']
        }
        
        # Get primary joints for this exercise
        joints_to_check = primary_joints.get(exercise_type, [])
        
        max_reps = 0
        
        for joint in joints_to_check:
            if joint in angles_df.columns:
                joint_data = angles_df[joint].dropna()
                
                if len(joint_data) > 20:  # Need enough data points
                    # Smooth data to reduce noise
                    smoothed = joint_data.rolling(window=5, center=True).mean().fillna(joint_data)
                    
                    # Find peaks and valleys to count repetitions
                    # A repetition is a transition from peak to valley and back to peak
                    
                    # Simple peak detection
                    peak_threshold = (smoothed.max() - smoothed.min()) * 0.3
                    peaks = []
                    valleys = []
                    
                    # Collect peaks and valleys
                    for i in range(2, len(smoothed) - 2):
                        if (smoothed.iloc[i] > smoothed.iloc[i-1] and 
                            smoothed.iloc[i] > smoothed.iloc[i-2] and
                            smoothed.iloc[i] > smoothed.iloc[i+1] and 
                            smoothed.iloc[i] > smoothed.iloc[i+2]):
                            peaks.append(i)
                        elif (smoothed.iloc[i] < smoothed.iloc[i-1] and 
                              smoothed.iloc[i] < smoothed.iloc[i-2] and
                              smoothed.iloc[i] < smoothed.iloc[i+1] and 
                              smoothed.iloc[i] < smoothed.iloc[i+2]):
                            valleys.append(i)
                    
                    # Count valid repetitions (peak-valley-peak sequences)
                    valid_reps = 0
                    peak_valley_pairs = min(len(peaks), len(valleys))
                    
                    if peak_valley_pairs > 0:
                        # Make sure peaks and valleys alternate
                        first_peak_idx = peaks[0]
                        first_valley_idx = valleys[0]
                        
                        if first_peak_idx < first_valley_idx:
                            # Starts with peak -> valley -> peak
                            valid_reps = peak_valley_pairs
                        else:
                            # Starts with valley -> peak -> valley
                            valid_reps = peak_valley_pairs - 1
                    
                    max_reps = max(max_reps, valid_reps)
        
        return max_reps
    
    def _analyze_squat(self, angles_df: pd.DataFrame) -> Dict:
        """
        Analyze squat-specific movement patterns.
        
        Args:
            angles_df: DataFrame of joint angles
            
        Returns:
            Dictionary of squat-specific metrics
        """
        results = {}
        
        # Check knee angles for depth
        knee_min = float('inf')
        for joint in ['LEFT_KNEE', 'RIGHT_KNEE']:
            if joint in angles_df.columns:
                joint_data = angles_df[joint].dropna()
                if len(joint_data) > 0:
                    knee_min = min(knee_min, joint_data.min())
        
        if knee_min != float('inf'):
            # Score squat depth (lower is better, typically 90 degrees is good)
            if knee_min <= 90:
                depth_score = 100
            else:
                # Reduce score as knee angle increases (less depth)
                depth_score = max(0, 100 - ((knee_min - 90) * 2))
            results['depth_score'] = float(depth_score)
        
        # Check if knees are tracking over toes
        # This would require 3D data or specific joint calculations
        # Simplified approximation for now
        knee_alignment_score = None
        if 'LEFT_KNEE' in angles_df.columns and 'LEFT_ANKLE' in angles_df.columns:
            # This is a simplified check that would need to be improved with actual knee-over-toe analysis
            knee_alignment_score = 70  # Placeholder score
        results['knee_alignment_score'] = knee_alignment_score
        
        return results
    
    def _analyze_lunge(self, angles_df: pd.DataFrame) -> Dict:
        """
        Analyze lunge-specific movement patterns.
        
        Args:
            angles_df: DataFrame of joint angles
            
        Returns:
            Dictionary of lunge-specific metrics
        """
        results = {}
        
        # Check front knee angle (should be close to 90 degrees at bottom)
        front_knee_score = None
        for joint in ['LEFT_KNEE', 'RIGHT_KNEE']:
            if joint in angles_df.columns:
                joint_data = angles_df[joint].dropna()
                if len(joint_data) > 0:
                    min_angle = joint_data.min()
                    # Score how close to 90 degrees
                    deviation = abs(min_angle - 90)
                    knee_score = max(0, 100 - (deviation * 2))
                    
                    if front_knee_score is None or knee_score > front_knee_score:
                        front_knee_score = knee_score
        
        results['front_knee_score'] = float(front_knee_score) if front_knee_score is not None else None
        
        # Analyze balance and stability
        # This would ideally use center of mass calculations
        # Simplified version looks at variation in upper body angles
        stability_score = None
        if 'TORSO' in angles_df.columns:
            torso_data = angles_df['TORSO'].dropna()
            if len(torso_data) > 0:
                # Less variation = more stable
                torso_std = torso_data.std()
                stability_score = max(0, 100 - (torso_std * 5))
        
        results['stability_score'] = float(stability_score) if stability_score is not None else None
        
        return results
    
    def _analyze_pushup(self, angles_df: pd.DataFrame) -> Dict:
        """
        Analyze pushup-specific movement patterns.
        
        Args:
            angles_df: DataFrame of joint angles
            
        Returns:
            Dictionary of pushup-specific metrics
        """
        results = {}
        
        # Check elbow angles for depth
        elbow_min = float('inf')
        for joint in ['LEFT_ELBOW', 'RIGHT_ELBOW']:
            if joint in angles_df.columns:
                joint_data = angles_df[joint].dropna()
                if len(joint_data) > 0:
                    elbow_min = min(elbow_min, joint_data.min())
        
        if elbow_min != float('inf'):
            # Score pushup depth (lower is better, typically 90 degrees is good)
            if elbow_min <= 90:
                depth_score = 100
            else:
                # Reduce score as elbow angle increases (less depth)
                depth_score = max(0, 100 - ((elbow_min - 90) * 2))
            results['depth_score'] = float(depth_score)
        
        # Check body alignment (should be straight line)
        alignment_score = None
        if 'TORSO' in angles_df.columns:
            torso_data = angles_df['TORSO'].dropna()
            if len(torso_data) > 0:
                # Closer to 180 degrees is better (straight line)
                avg_angle = torso_data.mean()
                deviation = abs(180 - avg_angle)
                alignment_score = max(0, 100 - (deviation * 2))
        
        results['body_alignment_score'] = float(alignment_score) if alignment_score is not None else None
        
        return results 