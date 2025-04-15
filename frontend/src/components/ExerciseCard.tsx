import type React from "react"
import { View, Text, StyleSheet, Image, TouchableOpacity } from "react-native"
import type { Exercise } from "../types"
import { Card } from "./Card"
import { colors, fontSizes, fontWeights, spacing, borderRadius } from "../theme"

interface ExerciseCardProps {
  exercise: Exercise
  onPress?: () => void
  compact?: boolean
}

export const ExerciseCard: React.FC<ExerciseCardProps> = ({ exercise, onPress, compact = false }) => {
  const { name, category, difficulty, targetMuscles, thumbnailUrl } = exercise

  const getDifficultyColor = (difficulty: Exercise["difficulty"]) => {
    switch (difficulty) {
      case "beginner":
        return colors.success
      case "intermediate":
        return colors.warning
      case "advanced":
        return colors.error
      default:
        return colors.gray[500]
    }
  }

  if (compact) {
    return (
      <TouchableOpacity onPress={onPress} activeOpacity={0.7}>
        <Card style={styles.compactCard}>
          <Image source={{ uri: thumbnailUrl }} style={styles.compactImage} />
          <View style={styles.compactContent}>
            <Text style={styles.compactTitle} numberOfLines={1}>
              {name}
            </Text>
            <View style={styles.compactMeta}>
              <Text style={[styles.tag, { backgroundColor: getDifficultyColor(difficulty) }]}>{difficulty}</Text>
              <Text style={styles.compactCategory}>{category}</Text>
            </View>
          </View>
        </Card>
      </TouchableOpacity>
    )
  }

  return (
    <TouchableOpacity onPress={onPress} activeOpacity={0.7}>
      <Card style={styles.card}>
        <Image source={{ uri: thumbnailUrl }} style={styles.image} />
        <View style={styles.content}>
          <Text style={styles.title}>{name}</Text>
          <View style={styles.metaContainer}>
            <Text style={[styles.tag, { backgroundColor: getDifficultyColor(difficulty) }]}>{difficulty}</Text>
            <Text style={styles.category}>{category}</Text>
          </View>
          <Text style={styles.muscles} numberOfLines={1}>
            Targets: {targetMuscles.join(", ")}
          </Text>
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
  image: {
    width: "100%",
    height: 150,
    resizeMode: "cover",
  },
  content: {
    padding: spacing.md,
  },
  title: {
    fontSize: fontSizes.lg,
    fontWeight: fontWeights.bold,
    marginBottom: spacing.xs,
    color: colors.gray[900],
  },
  metaContainer: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: spacing.xs,
  },
  tag: {
    fontSize: fontSizes.xs,
    color: "white",
    paddingVertical: 2,
    paddingHorizontal: 8,
    borderRadius: borderRadius.full,
    marginRight: spacing.sm,
    overflow: "hidden",
    fontWeight: fontWeights.medium,
  },
  category: {
    fontSize: fontSizes.sm,
    color: colors.gray[600],
    textTransform: "capitalize",
  },
  muscles: {
    fontSize: fontSizes.sm,
    color: colors.gray[700],
  },
  compactCard: {
    padding: 0,
    flexDirection: "row",
    alignItems: "center",
  },
  compactImage: {
    width: 60,
    height: 60,
    borderRadius: borderRadius.sm,
  },
  compactContent: {
    flex: 1,
    marginLeft: spacing.md,
  },
  compactTitle: {
    fontSize: fontSizes.md,
    fontWeight: fontWeights.semibold,
    color: colors.gray[900],
    marginBottom: spacing.xs,
  },
  compactMeta: {
    flexDirection: "row",
    alignItems: "center",
  },
  compactCategory: {
    fontSize: fontSizes.xs,
    color: colors.gray[600],
    textTransform: "capitalize",
  },
})
