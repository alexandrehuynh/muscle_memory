from enum import Enum, auto

class AnalysisType(str, Enum):
    """
    Types of movement analysis that can be performed.
    
    Options:
    - DETAILED_ANALYSIS: Full detailed analysis with metrics and visualizations
    - FORM_VISUALIZATION: Focus on form visualization with reference lines and angles
    - BASIC_TRACKING: Simple tracking with minimal processing
    - ANGLE_ONLY: Only calculate joint angles without additional metrics
    - QUICK_ANALYSIS: Simplified analysis optimized for speed
    """
    DETAILED_ANALYSIS = "detailed_analysis"
    FORM_VISUALIZATION = "form_visualization"
    BASIC_TRACKING = "basic_tracking"
    ANGLE_ONLY = "angle_only"
    QUICK_ANALYSIS = "quick_analysis" 