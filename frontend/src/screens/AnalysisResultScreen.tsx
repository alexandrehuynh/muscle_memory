"use client"

import { useState, useEffect, useRef } from "react"
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  SafeAreaView,
  StatusBar,
  ActivityIndicator,
  Dimensions,
  TouchableOpacity,
} from "react-native"
import { useRoute, type RouteProp } from "@react-navigation/native"
import { Video } from "expo-av"
import Slider from "@react-native-community/slider"
import { Play, Pause, RotateCcw, MessageCircle, Award, Share2 } from "lucide-react-native"

import type { RootStackParamList, Analysis, Feedback, SkeletonFrame } from "../types"
import { colors, spacing, fontSizes, fontWeights, borderRadius } from "../theme"
import { api } from "../services/api"
import { Card } from "../components/Card"
import { SkeletonRenderer } from "../components/SkeletonRenderer"

type AnalysisResultScreenRouteProp = RouteProp<RootStackParamList, "AnalysisResult">

const { width: screenWidth } = Dimensions.get("window")
const videoHeight = (screenWidth * 16) / 9 // 16:9 aspect ratio

const AnalysisResultScreen = () => {
  const route = useRoute<AnalysisResultScreenRouteProp>()
  const { analysisId } = route.params

  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [loading, setLoading] = useState(true)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [currentFrame, setCurrentFrame] = useState<SkeletonFrame | null>(null)
  const [activeFeedback, setActiveFeedback] = useState<Feedback | null>(null)
  const [highlightedJoints, setHighlightedJoints] = useState<string[]>([])

  const videoRef = useRef<Video>(null)

  useEffect(() => {
    const fetchAnalysis = async () => {
      try {
        setLoading(true)
        const data = await api.getAnalysis(analysisId)
        if (data) {
          setAnalysis(data)
        }
      } catch (error) {
        console.error("Error fetching analysis:", error)
      } finally {
        setLoading(false)
      }
    }

    fetchAnalysis()
  }, [analysisId])

  useEffect(() => {
    if (analysis && currentTime > 0) {
      // Find the closest frame to the current time
      const closestFrame = analysis.skeletonData.reduce((prev, curr) => {
        return Math.abs(curr.timestamp - currentTime) < Math.abs(prev.timestamp - currentTime) ? curr : prev
      })

      setCurrentFrame(closestFrame)

      // Find any feedback that should be shown at this time
      const relevantFeedback = analysis.feedback.find(
        (f) => Math.abs(f.timestamp - currentTime) < 0.5, // Within half a second
      )

      if (relevantFeedback) {
        setActiveFeedback(relevantFeedback)
        setHighlightedJoints(relevantFeedback.relatedJoints || [])
      } else {
        setActiveFeedback(null)
        setHighlightedJoints([])
      }
    }
  }, [analysis, currentTime])

  const handlePlaybackStatusUpdate = (status: any) => {
    if (status.isLoaded) {
      setIsPlaying(status.isPlaying)
      setCurrentTime(status.positionMillis / 1000) // Convert to seconds
      setDuration(status.durationMillis / 1000) // Convert to seconds
    }
  }

  const handlePlayPause = async () => {
    if (videoRef.current) {
      if (isPlaying) {
        await videoRef.current.pauseAsync()
      } else {
        await videoRef.current.playAsync()
      }
    }
  }

  const handleRestart = async () => {
    if (videoRef.current) {
      await videoRef.current.setPositionAsync(0)
      await videoRef.current.playAsync()
    }
  }

  const handleSliderChange = async (value: number) => {
    if (videoRef.current) {
      await videoRef.current.setPositionAsync(value * 1000) // Convert to milliseconds
    }
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs < 10 ? "0" : ""}${secs}`
  }

  const getFeedbackColor = (severity: Feedback["severity"]) => {
    switch (severity) {
      case "error":
        return colors.error
      case "warning":
        return colors.warning
      case "info":
        return colors.primary[500]
      default:
        return colors.gray[500]
    }
  }

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color={colors.primary[500]} />
        <Text style={styles.loadingText}>Loading analysis results...</Text>
      </View>
    )
  }

  if (!analysis) {
    return (
      <View style={styles.errorContainer}>
        <Text style={styles.errorText}>Analysis not found</Text>
      </View>
    )
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar barStyle="dark-content" backgroundColor="black" />
      <ScrollView style={styles.container}>
        <View style={styles.videoContainer}>
          <Video
            ref={videoRef}
            source={{ uri: analysis.videoUrl }}
            style={styles.video}
            resizeMode="contain"
            isLooping
            shouldPlay={false}
            onPlaybackStatusUpdate={handlePlaybackStatusUpdate}
          />

          {currentFrame && (
            <SkeletonRenderer
              frame={currentFrame}
              width={screenWidth}
              height={videoHeight}
              highlightJoints={highlightedJoints}
            />
          )}

          {activeFeedback && (
            <View style={styles.feedbackOverlay}>
              <View style={[styles.feedbackBubble, { borderColor: getFeedbackColor(activeFeedback.severity) }]}>
                <Text style={styles.feedbackText}>{activeFeedback.message}</Text>
              </View>
            </View>
          )}
        </View>

        <View style={styles.videoControls}>
          <View style={styles.timeContainer}>
            <Text style={styles.timeText}>{formatTime(currentTime)}</Text>
            <Text style={styles.timeText}>{formatTime(duration)}</Text>
          </View>

          <Slider
            style={styles.slider}
            minimumValue={0}
            maximumValue={duration}
            value={currentTime}
            onValueChange={handleSliderChange}
            minimumTrackTintColor={colors.primary[500]}
            maximumTrackTintColor={colors.gray[300]}
            thumbTintColor={colors.primary[500]}
          />

          <View style={styles.buttonContainer}>
            <TouchableOpacity style={styles.controlButton} onPress={handleRestart}>
              <RotateCcw size={24} color={colors.gray[700]} />
            </TouchableOpacity>

            <TouchableOpacity style={[styles.controlButton, styles.playButton]} onPress={handlePlayPause}>
              {isPlaying ? <Pause size={28} color="white" /> : <Play size={28} color="white" />}
            </TouchableOpacity>

            <View style={styles.controlButton} />
          </View>
        </View>

        <View style={styles.content}>
          <Card style={styles.scoreCard}>
            <View style={styles.scoreHeader}>
              <Award size={24} color={colors.primary[500]} />
              <Text style={styles.scoreTitle}>Performance Score</Text>
            </View>
            <Text style={styles.scoreValue}>{analysis.score}/100</Text>
            <View style={styles.scoreBar}>
              <View
                style={[
                  styles.scoreProgress,
                  { width: `${analysis.score}%` },
                  analysis.score < 60
                    ? { backgroundColor: colors.error }
                    : analysis.score < 80
                      ? { backgroundColor: colors.warning }
                      : { backgroundColor: colors.success },
                ]}
              />
            </View>
          </Card>

          <Card style={styles.feedbackCard}>
            <View style={styles.feedbackHeader}>
              <MessageCircle size={24} color={colors.primary[500]} />
              <Text style={styles.feedbackTitle}>Form Feedback</Text>
            </View>

            {analysis.feedback.length > 0 ? (
              <View style={styles.feedbackList}>
                {analysis.feedback.map((feedback, index) => (
                  <View
                    key={index}
                    style={[
                      styles.feedbackItem,
                      {
                        borderLeftColor: getFeedbackColor(feedback.severity),
                      },
                    ]}
                  >
                    <Text style={styles.feedbackTimestamp}>{formatTime(feedback.timestamp)}</Text>
                    <Text style={styles.feedbackMessage}>{feedback.message}</Text>
                  </View>
                ))}
              </View>
            ) : (
              <Text style={styles.noFeedbackText}>No feedback available</Text>
            )}
          </Card>

          <Card style={styles.detailsCard}>
            <Text style={styles.detailsTitle}>Exercise Details</Text>
            <View style={styles.detailsRow}>
              <Text style={styles.detailsLabel}>Exercise:</Text>
              <Text style={styles.detailsValue}>{analysis.exercise.name}</Text>
            </View>
            <View style={styles.detailsRow}>
              <Text style={styles.detailsLabel}>Date:</Text>
              <Text style={styles.detailsValue}>
                {new Date(analysis.date).toLocaleDateString()} at{" "}
                {new Date(analysis.date).toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                })}
              </Text>
            </View>
            <View style={styles.detailsRow}>
              <Text style={styles.detailsLabel}>Duration:</Text>
              <Text style={styles.detailsValue}>{analysis.duration} seconds</Text>
            </View>
          </Card>

          <TouchableOpacity style={styles.shareButton}>
            <Share2 size={20} color="white" />
            <Text style={styles.shareButtonText}>Share Results</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </SafeAreaView>
  )
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: "black",
  },
  container: {
    flex: 1,
    backgroundColor: colors.gray[50],
  },
  loadingContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: spacing.lg,
  },
  loadingText: {
    fontSize: fontSizes.md,
    color: colors.gray[600],
    marginTop: spacing.md,
  },
  errorContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: spacing.lg,
  },
  errorText: {
    fontSize: fontSizes.lg,
    color: colors.error,
    textAlign: "center",
  },
  videoContainer: {
    position: "relative",
    width: screenWidth,
    height: videoHeight,
    backgroundColor: "black",
  },
  video: {
    width: "100%",
    height: "100%",
  },
  feedbackOverlay: {
    position: "absolute",
    bottom: spacing.md,
    left: 0,
    right: 0,
    alignItems: "center",
  },
  feedbackBubble: {
    backgroundColor: "rgba(0, 0, 0, 0.7)",
    borderRadius: borderRadius.md,
    padding: spacing.md,
    borderLeftWidth: 4,
    maxWidth: "90%",
  },
  feedbackText: {
    color: "white",
    fontSize: fontSizes.md,
    fontWeight: fontWeights.medium,
  },
  videoControls: {
    backgroundColor: "white",
    padding: spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.gray[200],
  },
  timeContainer: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: spacing.xs,
  },
  timeText: {
    fontSize: fontSizes.sm,
    color: colors.gray[600],
  },
  slider: {
    width: "100%",
    height: 40,
  },
  buttonContainer: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingHorizontal: spacing.md,
  },
  controlButton: {
    width: 44,
    height: 44,
    justifyContent: "center",
    alignItems: "center",
  },
  playButton: {
    backgroundColor: colors.primary[500],
    borderRadius: 22,
  },
  content: {
    padding: spacing.md,
    gap: spacing.md,
  },
  scoreCard: {
    padding: spacing.lg,
  },
  scoreHeader: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: spacing.md,
  },
  scoreTitle: {
    fontSize: fontSizes.lg,
    fontWeight: fontWeights.semibold,
    color: colors.gray[900],
    marginLeft: spacing.sm,
  },
  scoreValue: {
    fontSize: fontSizes.xxxl,
    fontWeight: fontWeights.bold,
    color: colors.gray[900],
    marginBottom: spacing.md,
  },
  scoreBar: {
    height: 12,
    backgroundColor: colors.gray[200],
    borderRadius: borderRadius.full,
    overflow: "hidden",
  },
  scoreProgress: {
    height: "100%",
    backgroundColor: colors.success,
    borderRadius: borderRadius.full,
  },
  feedbackCard: {
    padding: spacing.lg,
  },
  feedbackHeader: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: spacing.md,
  },
  feedbackTitle: {
    fontSize: fontSizes.lg,
    fontWeight: fontWeights.semibold,
    color: colors.gray[900],
    marginLeft: spacing.sm,
  },
  feedbackList: {
    gap: spacing.md,
  },
  feedbackItem: {
    borderLeftWidth: 4,
    paddingLeft: spacing.md,
    paddingVertical: spacing.sm,
  },
  feedbackTimestamp: {
    fontSize: fontSizes.sm,
    color: colors.gray[500],
    marginBottom: spacing.xs,
  },
  feedbackMessage: {
    fontSize: fontSizes.md,
    color: colors.gray[800],
  },
  noFeedbackText: {
    fontSize: fontSizes.md,
    color: colors.gray[600],
    fontStyle: "italic",
    textAlign: "center",
    padding: spacing.md,
  },
  detailsCard: {
    padding: spacing.lg,
  },
  detailsTitle: {
    fontSize: fontSizes.lg,
    fontWeight: fontWeights.semibold,
    color: colors.gray[900],
    marginBottom: spacing.md,
  },
  detailsRow: {
    flexDirection: "row",
    marginBottom: spacing.sm,
  },
  detailsLabel: {
    width: 100,
    fontSize: fontSizes.md,
    fontWeight: fontWeights.medium,
    color: colors.gray[700],
  },
  detailsValue: {
    flex: 1,
    fontSize: fontSizes.md,
    color: colors.gray[900],
  },
  shareButton: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: colors.primary[500],
    padding: spacing.md,
    borderRadius: borderRadius.md,
    marginTop: spacing.sm,
    marginBottom: spacing.xl,
  },
  shareButtonText: {
    color: "white",
    fontSize: fontSizes.md,
    fontWeight: fontWeights.medium,
    marginLeft: spacing.sm,
  },
})

export default AnalysisResultScreen
