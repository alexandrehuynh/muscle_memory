import type React from "react"
import { View, Text, StyleSheet, Image, TouchableOpacity } from "react-native"
import type { Analysis } from "../types"
import { Card } from "./Card"
import { colors, fontSizes, fontWeights, spacing, borderRadius } from "../theme"

interface AnalysisCardProps {
  analysis: Analysis
  onPress?: () => void
}

export const AnalysisCard: React.FC<AnalysisCardProps> = ({ analysis, onPress }) => {
  const { exercise, date, score, duration } = analysis

  // Format date
  const formattedDate = new Date(date).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  })

  // Format time
  const formattedTime = new Date(date).toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
  })

  return (
    <TouchableOpacity onPress={onPress} activeOpacity={0.7}>
      <Card style={styles.card}>
        <View style={styles.header}>
          <View style={styles.scoreContainer}>
            <Text style={styles.scoreValue}>{score}</Text>
            <Text style={styles.scoreLabel}>Score</Text>
          </View>
          <View style={styles.exerciseInfo}>
            <Text style={styles.exerciseName}>{exercise.name}</Text>
            <Text style={styles.dateTime}>
              {formattedDate} at {formattedTime}
            </Text>
            <Text style={styles.duration}>{duration} seconds</Text>
          </View>
        </View>
        <View style={styles.thumbnailContainer}>
          <Image
            source={{ uri: analysis.videoUrl }}
            style={styles.thumbnail as any}
            resizeMode="cover"
          />
          <View style={styles.overlay}>
            <Text style={styles.viewText}>View Analysis</Text>
          </View>
        </View>
        <View style={styles.feedbackPreview}>
          {analysis.feedback.length > 0 ? (
            <Text style={styles.feedbackText} numberOfLines={1}>
              {analysis.feedback[0].message}
            </Text>
          ) : (
            <Text style={styles.noFeedbackText}>No feedback available</Text>
          )}
        </View>
      </Card>
    </TouchableOpacity>
  )
}

const styles = StyleSheet.create({
  card: {
    padding: 0,
    overflow: "hidden",
  },
  header: {
    flexDirection: "row",
    padding: spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.gray[200],
  },
  scoreContainer: {
    width: 60,
    height: 60,
    borderRadius: borderRadius.full,
    backgroundColor: colors.primary[500],
    justifyContent: "center",
    alignItems: "center",
    marginRight: spacing.md,
  },
  scoreValue: {
    color: "white",
    fontSize: fontSizes.xl,
    fontWeight: "700",
  },
  scoreLabel: {
    color: "white",
    fontSize: fontSizes.xs,
    textTransform: "uppercase",
  },
  exerciseInfo: {
    flex: 1,
    justifyContent: "center",
  },
  exerciseName: {
    fontSize: fontSizes.lg,
    fontWeight: "600",
    color: colors.gray[900],
    marginBottom: spacing.xs,
  },
  dateTime: {
    fontSize: fontSizes.sm,
    color: colors.gray[600],
    marginBottom: spacing.xs,
  },
  duration: {
    fontSize: fontSizes.sm,
    color: colors.gray[700],
  },
  thumbnailContainer: {
    position: "relative",
  },
  thumbnail: {
    width: "100%",
    height: 180,
  },
  overlay: {
    position: "absolute",
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: "rgba(0, 0, 0, 0.5)",
    padding: spacing.sm,
    alignItems: "center",
  },
  viewText: {
    color: "white",
    fontSize: fontSizes.sm,
    fontWeight: "500",
  },
  feedbackPreview: {
    padding: spacing.md,
    backgroundColor: colors.gray[50],
  },
  feedbackText: {
    fontSize: fontSizes.sm,
    color: colors.gray[800],
  },
  noFeedbackText: {
    fontSize: fontSizes.sm,
    color: colors.gray[500],
    fontStyle: "italic",
  },
})
