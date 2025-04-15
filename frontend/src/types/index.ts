export interface User {
  id: string
  name: string
  email: string
  profileImage?: string
  joinedDate: string
  stats: {
    totalWorkouts: number
    totalMinutes: number
    streak: number
  }
}

export interface Exercise {
  id: string
  name: string
  category: "strength" | "cardio" | "flexibility" | "balance"
  difficulty: "beginner" | "intermediate" | "advanced"
  targetMuscles: string[]
  description: string
  thumbnailUrl: string
  videoUrl?: string
}

export interface Analysis {
  id: string
  userId: string
  exerciseId: string
  exercise: Exercise
  date: string
  videoUrl: string
  duration: number
  skeletonData: SkeletonFrame[]
  feedback: Feedback[]
  score: number
}

export interface SkeletonFrame {
  timestamp: number
  keypoints: Keypoint[]
  angles: Angle[]
}

export interface Keypoint {
  name: string
  x: number
  y: number
  confidence: number
}

export interface Angle {
  name: string
  joints: [string, string, string]
  value: number
  ideal: number
  deviation: number
}

export interface Feedback {
  timestamp: number
  message: string
  severity: "info" | "warning" | "error"
  relatedJoints?: string[]
}

export type RootStackParamList = {
  Main: undefined
  RecordVideo: undefined
  UploadVideo: undefined
  ExerciseDetail: { exerciseId: string }
  AnalysisResult: { analysisId: string }
  Settings: undefined
}

export type MainTabParamList = {
  Home: undefined
  Library: undefined
  Profile: undefined
}
