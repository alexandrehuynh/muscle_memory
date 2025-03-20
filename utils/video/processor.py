import cv2
import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Generator
from datetime import datetime

class VideoProcessor:
    """
    Utility class for handling video processing operations.
    """
    
    @staticmethod
    def create_output_dirs():
        """Create necessary output directories."""
        os.makedirs("output", exist_ok=True)
        os.makedirs("temp/videos", exist_ok=True)
        os.makedirs("input", exist_ok=True)
    
    @staticmethod
    def get_video_info(video_path: str) -> Dict:
        """
        Get information about a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with video properties
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {}
        
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(cap.get(cv2.CAP_PROP_FPS)),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / int(cap.get(cv2.CAP_PROP_FPS)),
            'codec': int(cap.get(cv2.CAP_PROP_FOURCC))
        }
        
        cap.release()
        return info
    
    @staticmethod
    def record_from_webcam(output_path: str = None, record_time: int = None) -> str:
        """
        Record video from webcam.
        
        Args:
            output_path: Path to save the recorded video
            record_time: Maximum recording time in seconds
            
        Returns:
            Path to the saved video file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("input", exist_ok=True)
            output_path = os.path.join("input", f"webcam_recording_{timestamp}.mp4")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Unable to access webcam")
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 20
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        start_time = datetime.now()
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)  # Mirror the frame for user-friendly view
                
                # Add recording indicator
                cv2.putText(frame, "RECORDING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Display timestamp
                elapsed = (datetime.now() - start_time).seconds
                cv2.putText(frame, f"Time: {elapsed}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Write frame to output video
                out.write(frame)
                frame_count += 1
                
                # Display the frame to the user
                cv2.imshow('Recording (Press ESC to stop)', frame)
                
                # Check for user exit or time limit
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                    break
                
                if record_time and elapsed >= record_time:
                    break
        
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
        
        if frame_count == 0:
            return None
        
        return output_path
    
    @staticmethod
    def extract_frames(video_path: str, output_dir: str, frame_interval: int = 1) -> List[str]:
        """
        Extract frames from a video at specified intervals.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save extracted frames
            frame_interval: Extract every Nth frame
            
        Returns:
            List of paths to extracted frame images
        """
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        
        frame_paths = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
            
            frame_count += 1
        
        cap.release()
        return frame_paths