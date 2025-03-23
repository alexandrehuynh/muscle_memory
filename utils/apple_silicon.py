# ================================================================
# Apple Silicon Optimization for Muscle Memory Fitness Analysis
# ================================================================

"""
This module contains optimizations for Apple Silicon (M-series) chips.
Import this at the beginning of your application to enable hardware
acceleration for MediaPipe and OpenCV on Apple Silicon.
"""

import os
import platform
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def enable_apple_silicon_optimizations():
    """
    Configure environment for optimal performance on Apple Silicon.
    
    This function:
    1. Detects Apple Silicon processors
    2. Enables hardware acceleration for OpenCV
    3. Sets up TensorFlow for Metal (Apple's GPU framework)
    4. Configures threading for optimal performance
    
    Returns:
        bool: True if Apple Silicon optimizations were applied, False otherwise
    """
    try:
        # Check if running on macOS
        if platform.system() != 'Darwin':
            logger.info("Not running on macOS, skipping Apple Silicon optimizations")
            return False
        
        # Check if running on Apple Silicon
        is_apple_silicon = False
        machine = platform.machine()
        
        if machine == 'arm64':
            is_apple_silicon = True
            logger.info("Detected Apple Silicon processor")
        else:
            logger.info(f"Not running on Apple Silicon (detected: {machine})")
            return False
        
        if is_apple_silicon:
            # 1. OpenCV optimizations
            try:
                # Enable OpenCV hardware acceleration
                cv2.setUseOptimized(True)
                
                # Try enabling OpenCL (may not be available on all versions)
                try:
                    cv2.ocl.setUseOpenCL(True)
                    logger.info(f"OpenCL enabled: {cv2.ocl.useOpenCL()}")
                except:
                    logger.warning("Failed to enable OpenCL")
                
                logger.info(f"OpenCV optimizations enabled: {cv2.useOptimized()}")
            except Exception as e:
                logger.warning(f"Failed to enable OpenCV optimizations: {str(e)}")
            
            # 2. TensorFlow/MediaPipe optimizations
            try:
                # Enable Metal support for TensorFlow
                os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
                
                # Set up hardware acceleration for video processing
                os.environ['OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS'] = '1'
                
                # Try to import tensorflow and configure for Metal
                try:
                    import tensorflow as tf
                    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
                    
                    # Check for Metal plugin
                    metal_devices = tf.config.list_physical_devices('GPU')
                    if metal_devices:
                        logger.info(f"Metal GPU devices available: {metal_devices}")
                        # Configure TensorFlow to use the Metal device
                        for device in metal_devices:
                            tf.config.experimental.set_memory_growth(device, True)
                    else:
                        logger.warning("No Metal GPU devices found for TensorFlow")
                        
                    logger.info("TensorFlow configured for Metal acceleration")
                except ImportError:
                    logger.info("TensorFlow not installed, skipping Metal configuration")
                except Exception as e:
                    logger.warning(f"Failed to configure TensorFlow for Metal: {str(e)}")
            except Exception as e:
                logger.warning(f"Failed to set TensorFlow environment variables: {str(e)}")
            
            # 3. Threading optimizations
            try:
                # Set number of threads for optimal performance
                # On Apple Silicon, use 1 thread per performance core (usually 4)
                performance_cores = 4  # M1/M2/M3 typically have 4 performance cores
                
                # For OpenCV
                cv2.setNumThreads(performance_cores)
                
                # For NumPy (if using MKL backend)
                try:
                    import numpy as np
                    np.__config__.show()  # Print NumPy config for debugging
                    
                    # Try to set Accelerate framework for NumPy
                    try:
                        # Install libblas with accelerate backend
                        import subprocess
                        print("Installing Apple Accelerate framework for optimal performance...")
                        subprocess.run(["conda", "install", "-y", "libblas=*=*accelerate"], check=True)
                        print("Successfully installed Accelerate-optimized BLAS")
                    except Exception as e:
                        logger.warning(f"Failed to install Accelerate framework: {str(e)}")
                        logger.warning("NumPy may not be optimized for Apple Silicon")
                except Exception as e:
                    logger.warning(f"Failed to configure NumPy threading: {str(e)}")
                
                logger.info(f"Threading configured for {performance_cores} performance cores")
            except Exception as e:
                logger.warning(f"Failed to configure threading: {str(e)}")
            
            logger.info("Apple Silicon optimizations applied successfully")
            return True
    except Exception as e:
        logger.error(f"Error during Apple Silicon optimization: {str(e)}")
        return False
    
    return False

def check_mediapipe_compatibility():
    """
    Check if MediaPipe is properly configured for Apple Silicon.
    
    Returns:
        dict: Compatibility status and recommendations
    """
    results = {
        "compatible": False,
        "recommendations": [],
        "environment": {}
    }
    
    try:
        import mediapipe as mp
        
        # Record MediaPipe version
        results["environment"]["mediapipe_version"] = mp.__version__
        
        # Check Python version
        import sys
        results["environment"]["python_version"] = sys.version
        
        # Check if we're on Apple Silicon
        if platform.machine() == 'arm64' and platform.system() == 'Darwin':
            results["environment"]["platform"] = "Apple Silicon"
            
            # Check if using Miniforge/Conda
            is_conda = os.path.exists(os.path.join(sys.prefix, 'conda-meta'))
            results["environment"]["using_conda"] = is_conda
            
            if not is_conda:
                results["recommendations"].append(
                    "Install Miniforge with Conda environment for better Apple Silicon compatibility"
                )
            
            # Test a basic MediaPipe function
            try:
                test_image = np.zeros((100, 100, 3), dtype=np.uint8)
                mp_drawing = mp.solutions.drawing_utils
                mp_pose = mp.solutions.pose
                with mp_pose.Pose(
                    static_image_mode=True,
                    model_complexity=0,
                    min_detection_confidence=0.5
                ) as pose:
                    # Just to test if it runs without errors
                    pose.process(test_image)
                results["compatible"] = True
            except Exception as e:
                results["compatible"] = False
                results["error"] = str(e)
                results["recommendations"].append(
                    f"MediaPipe test failed. Consider reinstalling with Miniforge: {str(e)}"
                )
        else:
            results["environment"]["platform"] = f"{platform.system()} {platform.machine()}"
            results["recommendations"].append(
                "Not running on Apple Silicon. No specific optimizations needed."
            )
            results["compatible"] = True  # Assume compatible on non-Apple Silicon
    except ImportError:
        results["error"] = "MediaPipe not installed"
        results["recommendations"].append(
            "Install MediaPipe using Miniforge with Conda environment"
        )
    except Exception as e:
        results["error"] = str(e)
    
    return results

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run optimization
    result = enable_apple_silicon_optimizations()
    print(f"Optimizations applied: {result}")
    
    # Check MediaPipe compatibility
    compat = check_mediapipe_compatibility()
    print(f"MediaPipe compatibility: {compat['compatible']}")
    if not compat['compatible']:
        print(f"Error: {compat.get('error', 'Unknown error')}")
    
    for rec in compat['recommendations']:
        print(f"Recommendation: {rec}")