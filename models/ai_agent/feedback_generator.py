# muscle_memory/models/ai_agent/feedback_generator.py

from typing import Dict, List, Optional, Any
import json
import os

class FeedbackGenerator:
    """
    Generates coaching feedback from movement analysis results.
    This is a placeholder class that will later integrate with GPT-4 Mini.
    """
    
    def __init__(self):
        """Initialize the feedback generator."""
        self.exercise_templates = self._load_exercise_templates()
    
    def _load_exercise_templates(self) -> Dict:
        """Load exercise-specific feedback templates."""
        # In a real implementation, these would be loaded from a file
        return {
            "squat": {
                "form_cues": [
                    "Keep your back straight",
                    "Push your knees outward in line with your toes",
                    "Keep your weight in your heels",
                    "Brace your core throughout the movement"
                ],
                "depth_feedback": {
                    "Deep": "Great depth on your squat! You're achieving full range of motion.",
                    "Moderate": "You're getting to a moderate depth. Try going a bit deeper for full benefits.",
                    "Partial": "You're doing partial squats. Try to lower your hips until your thighs are parallel to the ground or lower."
                },
                "knee_safety": {
                    "Good knee angle maintained": "You're maintaining safe knee alignment throughout the movement.",
                    "Warning: Knee angle too acute at bottom of squat": "Be careful with how deep you're squatting. Your knees are bending more than recommended, which could cause strain."
                }
            },
            "lunge": {
                "form_cues": [
                    "Keep your torso upright",
                    "Step far enough forward to create proper angles",
                    "Keep your front knee aligned over your ankle",
                    "Push through your heel to return to standing"
                ],
                "form_quality": {
                    "Good": "Your lunge form is excellent with proper knee alignment.",
                    "Moderate": "Your lunge form is decent, but your knee position could be improved.",
                    "Needs improvement": "Your lunge technique needs work - focus on creating a 90-degree angle with your front knee."
                }
            },
            "pushup": {
                "form_cues": [
                    "Keep your body in a straight line",
                    "Position your hands slightly wider than shoulder-width",
                    "Lower your chest to the ground",
                    "Keep your elbows at about 45 degrees from your body"
                ],
                "depth_score": {
                    "Deep": "Great depth on your push-ups! You're achieving full range of motion.",
                    "Moderate": "You're going to a moderate depth. Try lowering a bit more for full benefits.",
                    "Partial": "You're doing partial push-ups. Focus on lowering your chest closer to the ground."
                }
            },
            "unknown": {
                "form_cues": [
                    "Maintain proper posture throughout the exercise",
                    "Move in a controlled manner",
                    "Focus on the muscles you're targeting",
                    "Breathe steadily throughout the movement"
                ]
            }
        }
    
    def generate_feedback(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate coaching feedback based on movement analysis results.
        
        Args:
            analysis_results: Dictionary containing movement analysis data
            
        Returns:
            Dictionary with feedback and recommendations
        """
        exercise_type = analysis_results.get("exercise_type", "unknown")
        metrics = analysis_results.get("metrics", {})
        statistics = analysis_results.get("statistics", {})
        
        # Get template for this exercise type
        template = self.exercise_templates.get(exercise_type, self.exercise_templates["unknown"])
        
        # Generate general feedback based on symmetry score
        symmetry_score = metrics.get("symmetry_score", 0)
        symmetry_feedback = self._generate_symmetry_feedback(symmetry_score)
        
        # Generate smoothness feedback
        smoothness_score = metrics.get("smoothness_score", 0)
        smoothness_feedback = self._generate_smoothness_feedback(smoothness_score)
        
        # Generate range of motion feedback
        rom_score = metrics.get("range_of_motion_score", 0)
        rom_feedback = self._generate_rom_feedback(rom_score)
        
        # Get exercise-specific feedback
        exercise_feedback = self._generate_exercise_specific_feedback(exercise_type, metrics, template)
        
        # Combine all feedback
        main_feedback = exercise_feedback + " " + symmetry_feedback + " " + rom_feedback
        
        # Generate recommendations
        recommendations = self._generate_recommendations(exercise_type, metrics, statistics, template)
        
        return {
            "feedback": main_feedback,
            "recommendations": recommendations,
            "form_cues": template.get("form_cues", []),
            "scores": {
                "symmetry": symmetry_score,
                "smoothness": smoothness_score,
                "range_of_motion": rom_score
            }
        }
    
    def _generate_symmetry_feedback(self, symmetry_score: float) -> str:
        """Generate feedback about movement symmetry."""
        if symmetry_score >= 90:
            return "Your movement is very well balanced between left and right sides."
        elif symmetry_score >= 75:
            return "Your movement shows good balance between sides, with only minor asymmetries."
        elif symmetry_score >= 60:
            return "There is moderate asymmetry in your movement. Try to engage both sides equally."
        else:
            return "Your movement shows significant asymmetry. Focus on balancing the work between both sides."
    
    def _generate_smoothness_feedback(self, smoothness_score: float) -> str:
        """Generate feedback about movement smoothness."""
        if smoothness_score >= 90:
            return "Your movement is very smooth and controlled."
        elif smoothness_score >= 75:
            return "Your movement shows good control with only minor jerky moments."
        elif smoothness_score >= 60:
            return "Work on maintaining more control throughout the movement to reduce jerkiness."
        else:
            return "Focus on slow, controlled movements rather than quick, jerky actions."
    
    def _generate_rom_feedback(self, rom_score: float) -> str:
        """Generate feedback about range of motion."""
        if rom_score >= 90:
            return "You're achieving excellent range of motion in this exercise."
        elif rom_score >= 75:
            return "Your range of motion is good, but could be improved slightly for maximum benefit."
        elif rom_score >= 60:
            return "You could benefit from increasing your range of motion in this exercise."
        else:
            return "Work on increasing your mobility to achieve a greater range of motion."
    
    def _generate_exercise_specific_feedback(self, exercise_type: str, metrics: Dict, template: Dict) -> str:
        """Generate exercise-specific feedback."""
        if exercise_type == "squat":
            depth = metrics.get("depth_score", "Moderate")
            knee_safety = metrics.get("knee_safety", "Good knee angle maintained")
            
            depth_feedback = template.get("depth_feedback", {}).get(depth, "")
            knee_feedback = template.get("knee_safety", {}).get(knee_safety, "")
            
            return f"{depth_feedback} {knee_feedback}"
        
        elif exercise_type == "lunge":
            form_quality = metrics.get("form_quality", "Moderate")
            forward_leg = metrics.get("forward_leg", "left")
            
            form_feedback = template.get("form_quality", {}).get(form_quality, "")
            
            return f"{form_feedback} You're primarily lunging with your {forward_leg} leg forward."
        
        elif exercise_type == "pushup":
            depth = metrics.get("depth_score", "Moderate")
            depth_feedback = template.get("depth_score", {}).get(depth, "")
            
            return depth_feedback
        
        else:
            return "Your movement has been analyzed, but it doesn't match our defined exercise patterns."
    
    def _generate_recommendations(self, exercise_type: str, metrics: Dict, statistics: Dict, template: Dict) -> List[str]:
        """Generate exercise-specific recommendations."""
        recommendations = []
        
        # Generic recommendations based on scores
        symmetry_score = metrics.get("symmetry_score", 0)
        if symmetry_score < 75:
            recommendations.append("Work on balancing your effort between left and right sides.")
        
        smoothness_score = metrics.get("smoothness_score", 0)
        if smoothness_score < 75:
            recommendations.append("Practice slower, more controlled movements to improve smoothness.")
        
        rom_score = metrics.get("range_of_motion_score", 0)
        if rom_score < 75:
            recommendations.append("Consider mobility exercises to increase your range of motion.")
        
        # Exercise-specific recommendations
        if exercise_type == "squat":
            depth = metrics.get("depth_score", "Moderate")
            if depth in ["Partial", "Moderate"]:
                recommendations.append("Try box squats to improve your depth awareness.")
                recommendations.append("Focus on hip mobility exercises to improve squat depth.")
            
            knee_safety = metrics.get("knee_safety", "")
            if "Warning" in knee_safety:
                recommendations.append("Work on ankle mobility to improve knee alignment during squats.")
        
        elif exercise_type == "lunge":
            form_quality = metrics.get("form_quality", "")
            if form_quality in ["Needs improvement", "Moderate"]:
                recommendations.append("Practice split squats to improve lunge stability.")
                recommendations.append("Focus on keeping your front knee aligned with your ankle.")
        
        elif exercise_type == "pushup":
            depth = metrics.get("depth_score", "")
            if depth in ["Partial", "Moderate"]:
                recommendations.append("Try incline push-ups to build strength through the full range of motion.")
                recommendations.append("Focus on chest-to-ground movement, even if it means doing fewer repetitions.")
        
        return recommendations