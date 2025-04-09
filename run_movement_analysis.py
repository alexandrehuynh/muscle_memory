#!/usr/bin/env python
"""
Movement Analysis CLI

This script demonstrates the use of the refactored movement analysis pipeline.
It allows analyzing videos from the command line.
"""

import argparse
import os
import logging
from datetime import datetime
from models.movement import MovementPipeline, PipelineConfig

def setup_logging():
    """Set up logging."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    return logging.getLogger(__name__)

def main():
    """Main entry point."""
    logger = setup_logging()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Movement Analysis CLI")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--output", "-o", help="Path to save the annotated video (optional)")
    parser.add_argument("--model-complexity", "-m", type=int, choices=[0, 1, 2], default=1,
                        help="MediaPipe model complexity (0-2)")
    parser.add_argument("--analysis-type", "-a", choices=["detailed", "quick", "realtime"], 
                        default="detailed", help="Type of analysis to perform")
    parser.add_argument("--exercise", "-e", choices=["squat", "lunge", "pushup"], 
                        help="Exercise type (auto-detected if not specified)")
    parser.add_argument("--joints", "-j", nargs="+", 
                        help="Specific joints to analyze (all if not specified)")
    parser.add_argument("--show-angles", action="store_true", default=True,
                        help="Show joint angles on the annotated video")
    parser.add_argument("--show-reference-lines", action="store_true", default=True,
                        help="Show reference lines on the annotated video")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.isfile(args.video_path):
        logger.error(f"Input file does not exist: {args.video_path}")
        return 1
    
    # Create default output path if not provided
    output_path = args.output
    if output_path is None:
        base_name = os.path.basename(args.video_path)
        date_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join("output", "videos", f"{date_prefix}_analyzed_{base_name}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    logger.info(f"Processing video: {args.video_path}")
    logger.info(f"Output will be saved to: {output_path}")
    
    # Create pipeline configuration
    pipeline_config = PipelineConfig(
        model_complexity=args.model_complexity,
        analysis_type=args.analysis_type,
        show_angles=args.show_angles,
        show_reference_lines=args.show_reference_lines,
        auto_detect_exercise=args.exercise is None,
        exercise_type=args.exercise
    )
    
    # Create pipeline and process video
    pipeline = MovementPipeline(pipeline_config)
    success, message = pipeline.process_video(args.video_path, output_path, args.joints)
    
    if not success:
        logger.error(f"Error processing video: {message}")
        return 1
    
    logger.info(message)
    
    # Analyze movement
    analysis_success, analysis_results, analysis_message = pipeline.analyze_movement(args.joints)
    
    if not analysis_success:
        logger.error(f"Error analyzing movement: {analysis_message}")
        return 1
    
    # Print analysis results
    logger.info(analysis_message)
    logger.info(f"Exercise type: {analysis_results.get('exercise_type', 'unknown')}")
    
    metrics = analysis_results.get('metrics', {})
    logger.info("Movement metrics:")
    logger.info(f"  Symmetry score: {metrics.get('symmetry', 0):.2f}")
    logger.info(f"  Smoothness score: {metrics.get('smoothness', 0):.2f}")
    logger.info(f"  Range of motion score: {metrics.get('range_of_motion', 0):.2f}")
    logger.info(f"  Repetition count: {metrics.get('repetition_count', 0)}")
    
    # Print exercise-specific metrics
    exercise_type = analysis_results.get('exercise_type', 'unknown')
    logger.info(f"{exercise_type.capitalize()} specific metrics:")
    
    if exercise_type == 'squat':
        logger.info(f"  Depth score: {metrics.get('depth_score', 0):.2f}")
        logger.info(f"  Knee alignment score: {metrics.get('knee_alignment_score', 0):.2f}")
    elif exercise_type == 'lunge':
        logger.info(f"  Front knee score: {metrics.get('front_knee_score', 0):.2f}")
        logger.info(f"  Stability score: {metrics.get('stability_score', 0):.2f}")
    elif exercise_type == 'pushup':
        logger.info(f"  Depth score: {metrics.get('depth_score', 0):.2f}")
        logger.info(f"  Body alignment score: {metrics.get('body_alignment_score', 0):.2f}")
    
    logger.info(f"Annotated video saved to: {output_path}")
    
    # Clean up resources
    pipeline.release()
    
    return 0

if __name__ == "__main__":
    exit(main()) 