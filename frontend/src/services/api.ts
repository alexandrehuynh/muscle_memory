import type { Analysis, Exercise, User, Feedback, SkeletonFrame } from "../types"

// Mock user data
const mockUser: User = {
  id: "user1",
  name: "Alex Johnson",
  email: "alex@example.com",
  profileImage: "https://i.pravatar.cc/300",
  joinedDate: "2023-01-15",
  stats: {
    totalWorkouts: 48,
    totalMinutes: 1240,
    streak: 5,
  },
}

// Mock exercises data
const mockExercises: Exercise[] = [
  {
    id: "ex1",
    name: "Squat",
    category: "strength",
    difficulty: "beginner",
    targetMuscles: ["quadriceps", "hamstrings", "glutes", "core"],
    description: "A compound exercise that targets multiple muscle groups in the lower body.",
    thumbnailUrl: "/placeholder.svg?height=200&width=300",
  },
  {
    id: "ex2",
    name: "Push-up",
    category: "strength",
    difficulty: "intermediate",
    targetMuscles: ["chest", "shoulders", "triceps", "core"],
    description: "A classic upper body exercise that builds strength in the chest, shoulders, and arms.",
    thumbnailUrl: "/placeholder.svg?height=200&width=300",
  },
  {
    id: "ex3",
    name: "Deadlift",
    category: "strength",
    difficulty: "advanced",
    targetMuscles: ["lower back", "glutes", "hamstrings", "core"],
    description: "A compound exercise that targets the posterior chain and builds overall strength.",
    thumbnailUrl: "/placeholder.svg?height=200&width=300",
  },
  {
    id: "ex4",
    name: "Plank",
    category: "strength",
    difficulty: "beginner",
    targetMuscles: ["core", "shoulders", "back"],
    description: "An isometric core exercise that improves stability and posture.",
    thumbnailUrl: "/placeholder.svg?height=200&width=300",
  },
  {
    id: "ex5",
    name: "Jumping Jacks",
    category: "cardio",
    difficulty: "beginner",
    targetMuscles: ["full body"],
    description: "A simple cardio exercise that elevates heart rate and improves coordination.",
    thumbnailUrl: "/placeholder.svg?height=200&width=300",
  },
  {
    id: "ex6",
    name: "Downward Dog",
    category: "flexibility",
    difficulty: "beginner",
    targetMuscles: ["shoulders", "hamstrings", "calves", "back"],
    description: "A yoga pose that stretches and strengthens the entire body.",
    thumbnailUrl: "/placeholder.svg?height=200&width=300",
  },
]

// Generate mock skeleton data
const generateMockSkeletonFrames = (duration: number): SkeletonFrame[] => {
  const frames: SkeletonFrame[] = []
  const frameCount = Math.floor(duration * 30) // Assuming 30fps

  for (let i = 0; i < frameCount; i += 5) {
    // Generate every 5th frame for mock data
    const timestamp = i / 30
    frames.push({
      timestamp,
      keypoints: [
        { name: "nose", x: 0.5 + Math.sin(timestamp) * 0.01, y: 0.2 + Math.cos(timestamp) * 0.01, confidence: 0.95 },
        {
          name: "left_shoulder",
          x: 0.4 + Math.sin(timestamp) * 0.02,
          y: 0.3 + Math.cos(timestamp) * 0.01,
          confidence: 0.9,
        },
        {
          name: "right_shoulder",
          x: 0.6 + Math.sin(timestamp) * 0.02,
          y: 0.3 + Math.cos(timestamp) * 0.01,
          confidence: 0.9,
        },
        {
          name: "left_elbow",
          x: 0.3 + Math.sin(timestamp) * 0.03,
          y: 0.4 + Math.cos(timestamp) * 0.02,
          confidence: 0.85,
        },
        {
          name: "right_elbow",
          x: 0.7 + Math.sin(timestamp) * 0.03,
          y: 0.4 + Math.cos(timestamp) * 0.02,
          confidence: 0.85,
        },
        {
          name: "left_wrist",
          x: 0.25 + Math.sin(timestamp) * 0.04,
          y: 0.5 + Math.cos(timestamp) * 0.03,
          confidence: 0.8,
        },
        {
          name: "right_wrist",
          x: 0.75 + Math.sin(timestamp) * 0.04,
          y: 0.5 + Math.cos(timestamp) * 0.03,
          confidence: 0.8,
        },
        {
          name: "left_hip",
          x: 0.45 + Math.sin(timestamp) * 0.01,
          y: 0.6 + Math.cos(timestamp) * 0.01,
          confidence: 0.9,
        },
        {
          name: "right_hip",
          x: 0.55 + Math.sin(timestamp) * 0.01,
          y: 0.6 + Math.cos(timestamp) * 0.01,
          confidence: 0.9,
        },
        {
          name: "left_knee",
          x: 0.4 + Math.sin(timestamp) * 0.02,
          y: 0.75 + Math.cos(timestamp) * 0.02,
          confidence: 0.85,
        },
        {
          name: "right_knee",
          x: 0.6 + Math.sin(timestamp) * 0.02,
          y: 0.75 + Math.cos(timestamp) * 0.02,
          confidence: 0.85,
        },
        {
          name: "left_ankle",
          x: 0.4 + Math.sin(timestamp) * 0.01,
          y: 0.9 + Math.cos(timestamp) * 0.01,
          confidence: 0.8,
        },
        {
          name: "right_ankle",
          x: 0.6 + Math.sin(timestamp) * 0.01,
          y: 0.9 + Math.cos(timestamp) * 0.01,
          confidence: 0.8,
        },
      ],
      angles: [
        {
          name: "left_elbow",
          joints: ["left_shoulder", "left_elbow", "left_wrist"],
          value: 160 + Math.sin(timestamp) * 20,
          ideal: 170,
          deviation: 10,
        },
        {
          name: "right_elbow",
          joints: ["right_shoulder", "right_elbow", "right_wrist"],
          value: 160 + Math.sin(timestamp) * 20,
          ideal: 170,
          deviation: 10,
        },
        {
          name: "left_shoulder",
          joints: ["left_elbow", "left_shoulder", "left_hip"],
          value: 80 + Math.sin(timestamp) * 10,
          ideal: 90,
          deviation: 10,
        },
        {
          name: "right_shoulder",
          joints: ["right_elbow", "right_shoulder", "right_hip"],
          value: 80 + Math.sin(timestamp) * 10,
          ideal: 90,
          deviation: 10,
        },
        {
          name: "left_hip",
          joints: ["left_shoulder", "left_hip", "left_knee"],
          value: 170 + Math.sin(timestamp) * 10,
          ideal: 180,
          deviation: 10,
        },
        {
          name: "right_hip",
          joints: ["right_shoulder", "right_hip", "right_knee"],
          value: 170 + Math.sin(timestamp) * 10,
          ideal: 180,
          deviation: 10,
        },
        {
          name: "left_knee",
          joints: ["left_hip", "left_knee", "left_ankle"],
          value: 170 + Math.sin(timestamp) * 20,
          ideal: 180,
          deviation: 10,
        },
        {
          name: "right_knee",
          joints: ["right_hip", "right_knee", "right_ankle"],
          value: 170 + Math.sin(timestamp) * 20,
          ideal: 180,
          deviation: 10,
        },
      ],
    })
  }

  return frames
}

// Generate mock feedback
const generateMockFeedback = (): Feedback[] => {
  return [
    {
      timestamp: 1.5,
      message: "Keep your back straight",
      severity: "warning",
      relatedJoints: ["left_shoulder", "left_hip", "right_shoulder", "right_hip"],
    },
    {
      timestamp: 3.2,
      message: "Bend your knees more",
      severity: "error",
      relatedJoints: ["left_knee", "right_knee"],
    },
    {
      timestamp: 5.7,
      message: "Good form on the upward movement",
      severity: "info",
    },
    {
      timestamp: 8.3,
      message: "Keep your weight on your heels",
      severity: "warning",
      relatedJoints: ["left_ankle", "right_ankle"],
    },
  ]
}

// Mock analyses data
const mockAnalyses: Analysis[] = [
  {
    id: "an1",
    userId: "user1",
    exerciseId: "ex1",
    exercise: mockExercises[0],
    date: "2023-05-10T14:30:00Z",
    videoUrl: "/placeholder.svg?height=400&width=300",
    duration: 15,
    skeletonData: generateMockSkeletonFrames(15),
    feedback: generateMockFeedback(),
    score: 85,
  },
  {
    id: "an2",
    userId: "user1",
    exerciseId: "ex2",
    exercise: mockExercises[1],
    date: "2023-05-08T09:15:00Z",
    videoUrl: "/placeholder.svg?height=400&width=300",
    duration: 12,
    skeletonData: generateMockSkeletonFrames(12),
    feedback: generateMockFeedback(),
    score: 78,
  },
  {
    id: "an3",
    userId: "user1",
    exerciseId: "ex4",
    exercise: mockExercises[3],
    date: "2023-05-05T16:45:00Z",
    videoUrl: "/placeholder.svg?height=400&width=300",
    duration: 10,
    skeletonData: generateMockSkeletonFrames(10),
    feedback: generateMockFeedback(),
    score: 92,
  },
]

// API service
export const api = {
  // User endpoints
  getUser: async (): Promise<User> => {
    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 500))
    return mockUser
  },

  // Exercise endpoints
  getExercises: async (): Promise<Exercise[]> => {
    await new Promise((resolve) => setTimeout(resolve, 700))
    return mockExercises
  },

  getExercise: async (id: string): Promise<Exercise | undefined> => {
    await new Promise((resolve) => setTimeout(resolve, 300))
    return mockExercises.find((ex) => ex.id === id)
  },

  searchExercises: async (query: string, filters?: { category?: string; difficulty?: string }): Promise<Exercise[]> => {
    await new Promise((resolve) => setTimeout(resolve, 500))
    let filtered = mockExercises

    if (query) {
      const lowerQuery = query.toLowerCase()
      filtered = filtered.filter(
        (ex) =>
          ex.name.toLowerCase().includes(lowerQuery) ||
          ex.description.toLowerCase().includes(lowerQuery) ||
          ex.targetMuscles.some((muscle) => muscle.toLowerCase().includes(lowerQuery)),
      )
    }

    if (filters?.category) {
      filtered = filtered.filter((ex) => ex.category === filters.category)
    }

    if (filters?.difficulty) {
      filtered = filtered.filter((ex) => ex.difficulty === filters.difficulty)
    }

    return filtered
  },

  // Analysis endpoints
  getAnalyses: async (userId: string): Promise<Analysis[]> => {
    await new Promise((resolve) => setTimeout(resolve, 800))
    return mockAnalyses.filter((analysis) => analysis.userId === userId)
  },

  getAnalysis: async (id: string): Promise<Analysis | undefined> => {
    await new Promise((resolve) => setTimeout(resolve, 400))
    return mockAnalyses.find((analysis) => analysis.id === id)
  },

  uploadVideo: async (exerciseId: string, videoUri: string): Promise<Analysis> => {
    // Simulate video processing and analysis
    await new Promise((resolve) => setTimeout(resolve, 2000))

    const exercise = mockExercises.find((ex) => ex.id === exerciseId)
    if (!exercise) {
      throw new Error("Exercise not found")
    }

    const newAnalysis: Analysis = {
      id: `an${Date.now()}`,
      userId: "user1",
      exerciseId,
      exercise,
      date: new Date().toISOString(),
      videoUrl: videoUri,
      duration: Math.floor(Math.random() * 10) + 10, // Random duration between 10-20 seconds
      skeletonData: generateMockSkeletonFrames(15),
      feedback: generateMockFeedback(),
      score: Math.floor(Math.random() * 30) + 70, // Random score between 70-100
    }

    // Add to mock data
    mockAnalyses.push(newAnalysis)

    return newAnalysis
  },
}
