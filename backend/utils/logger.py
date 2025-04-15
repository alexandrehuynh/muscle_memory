import logging
import os
from datetime import datetime

def setup_logging():
    """Configure logging for the application."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(log_dir, f"muscle_memory_{today}.log")
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Set more verbose logging for specific modules
    logging.getLogger('models.movement').setLevel(logging.DEBUG)
    logging.getLogger('models.mediapipe').setLevel(logging.DEBUG)
    
    return logging.getLogger(__name__)