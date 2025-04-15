import cv2
import numpy as np
import mediapipe as mp
import os
import sys
import time

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.movement.visualization.skeleton_visualizer import SkeletonVisualizer, VisualizationConfig

def test_skeleton_visualizer():
    """Test the SkeletonVisualizer implementation."""
    # Create configuration
    config = VisualizationConfig(
        confidence_threshold=0.5,
        visibility_threshold=0.5,
        connection_thickness=2,
        landmark_radius=5,
        landmark_thickness=2,
        simplified_torso=True,
        hide_face_landmarks=True
    )
    
    # Create visualizer
    visualizer = SkeletonVisualizer(config)
    
    # Create a sample frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Create sample landmarks (x, y, visibility)
    landmarks = []
    
    # Add landmarks for a simple stick figure
    # Format: [x (normalized), y (normalized), visibility]
    
    # Head (nose)
    landmarks.append([0.5, 0.2, 0.9])
    
    # Eyes
    landmarks.append([0.48, 0.19, 0.9])  # LEFT_EYE_INNER
    landmarks.append([0.47, 0.19, 0.9])  # LEFT_EYE
    landmarks.append([0.46, 0.19, 0.9])  # LEFT_EYE_OUTER
    landmarks.append([0.52, 0.19, 0.9])  # RIGHT_EYE_INNER
    landmarks.append([0.53, 0.19, 0.9])  # RIGHT_EYE
    landmarks.append([0.54, 0.19, 0.9])  # RIGHT_EYE_OUTER
    
    # Ears
    landmarks.append([0.45, 0.2, 0.9])  # LEFT_EAR
    landmarks.append([0.55, 0.2, 0.9])  # RIGHT_EAR
    
    # Mouth
    landmarks.append([0.48, 0.22, 0.9])  # MOUTH_LEFT
    landmarks.append([0.52, 0.22, 0.9])  # MOUTH_RIGHT
    
    # Add other body landmarks
    # Shoulders
    landmarks.append([0.4, 0.3, 0.9])  # LEFT_SHOULDER
    landmarks.append([0.6, 0.3, 0.9])  # RIGHT_SHOULDER
    
    # Elbows
    landmarks.append([0.3, 0.4, 0.9])  # LEFT_ELBOW
    landmarks.append([0.7, 0.4, 0.9])  # RIGHT_ELBOW
    
    # Wrists
    landmarks.append([0.25, 0.5, 0.9])  # LEFT_WRIST
    landmarks.append([0.75, 0.5, 0.9])  # RIGHT_WRIST
    
    # Hips
    landmarks.append([0.45, 0.5, 0.9])  # LEFT_HIP
    landmarks.append([0.55, 0.5, 0.9])  # RIGHT_HIP
    
    # Knees
    landmarks.append([0.45, 0.7, 0.9])  # LEFT_KNEE
    landmarks.append([0.55, 0.7, 0.9])  # RIGHT_KNEE
    
    # Ankles
    landmarks.append([0.45, 0.9, 0.9])  # LEFT_ANKLE
    landmarks.append([0.55, 0.9, 0.9])  # RIGHT_ANKLE
    
    # Fill in any remaining landmarks needed by MediaPipe
    # Each landmark follows the format [x, y, visibility]
    while len(landmarks) < 33:  # MediaPipe Pose has 33 landmarks
        landmarks.append([0.5, 0.5, 0.0])  # Low visibility for unused landmarks
    
    # Visualize with face landmarks hidden
    result_hidden_face = visualizer.visualize(frame, landmarks)
    
    # Show the result
    cv2.imshow("Skeleton (Face Hidden)", result_hidden_face)
    
    # Change config to show face landmarks
    visualizer.config.hide_face_landmarks = False
    
    # Visualize with face landmarks shown
    result_shown_face = visualizer.visualize(frame, landmarks)
    
    # Show the result
    cv2.imshow("Skeleton (Face Shown)", result_shown_face)
    
    # Wait for key press
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Test completed successfully!")

if __name__ == "__main__":
    test_skeleton_visualizer() 