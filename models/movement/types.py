from enum import Enum

class AnalysisType(str, Enum):
    """
    Types of movement analysis that can be performed.
    
    Options:
    - DETAILED: Full detailed analysis with metrics and visualizations
    - QUICK: Simplified analysis optimized for speed
    - REALTIME: Analysis optimized for real-time tracking
    """
    DETAILED = "detailed"
    QUICK = "quick"
    REALTIME = "realtime" 