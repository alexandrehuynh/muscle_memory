"use client"

import { useState, useEffect } from "react"
import { View, Text, StyleSheet, ScrollView, Image, SafeAreaView, StatusBar, ActivityIndicator } from "react-native"
import { useNavigation, useRoute, type RouteProp } from "@react-navigation/native"
import type { NativeStackNavigationProp } from "@react-navigation/native-stack"
import { Camera } from "lucide-react-native"

import type { RootStackParamList, Exercise } from "../types"
import { colors, spacing, fontSizes, fontWeights, borderRadius } from "../theme"
import { api } from "../services/api"
import { Button } from "../components/Button"
import { Card } from "../components/Card"

type ExerciseDetailScreenRouteProp = RouteProp<RootStackParamList, "ExerciseDetail">
type ExerciseDetailScreenNavigationProp = NativeStackNavigationProp<RootStackParamList>

const ExerciseDetailScreen = () => {
  const route = useRoute<ExerciseDetailScreenRouteProp>()
  const navigation = useNavigation<ExerciseDetailScreenNavigationProp>()
  const { exerciseId } = route.params

  const [exercise, setExercise] = useState<Exercise | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchExercise = async () => {
      try {
        setLoading(true)
        const data = await api.getExercise(exerciseId)
        if (data) {
          setExercise(data)
          // Update the navigation title
          navigation.setOptions({ title: data.name })
        }
      } catch (error) {
        console.error("Error fetching exercise details:", error)
      } finally {
        setLoading(false)
      }
    }

    fetchExercise()
  }, [exerciseId, navigation])

  const handleRecordVideo = () => {
    navigation.navigate("RecordVideo")
  }

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color={colors.primary[500]} />
        <Text style={styles.loadingText}>Loading exercise details...</Text>
      </View>
    )
  }

  if (!exercise) {
    return (
      <View style={styles.errorContainer}>
        <Text style={styles.errorText}>Exercise not found</Text>
        <Button title="Go Back" onPress={() => navigation.goBack()} style={{ marginTop: spacing.lg }} />
      </View>
    )
  }

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

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar barStyle="dark-content" backgroundColor="white" />
      <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
        <Image source={{ uri: exercise.thumbnailUrl }} style={styles.image} resizeMode="cover" />

        <View style={styles.content}>
          <View style={styles.header}>
            <Text style={styles.title}>{exercise.name}</Text>
            <View style={styles.metaContainer}>
              <Text style={[styles.tag, { backgroundColor: getDifficultyColor(exercise.difficulty) }]}>
                {exercise.difficulty}
              </Text>
              <Text style={styles.category}>{exercise.category}</Text>
            </View>
          </View>

          <Card style={styles.descriptionCard}>
            <Text style={styles.sectionTitle}>Description</Text>
            <Text style={styles.description}>{exercise.description}</Text>
          </Card>

          <Card style={styles.targetsCard}>
            <Text style={styles.sectionTitle}>Target Muscles</Text>
            <View style={styles.musclesList}>
              {exercise.targetMuscles.map((muscle, index) => (
                <View key={index} style={styles.muscleItem}>
                  <View style={styles.muscleDot} />
                  <Text style={styles.muscleName}>{muscle}</Text>
                </View>
              ))}
            </View>
          </Card>

          <Card style={styles.actionCard}>
            <Text style={styles.actionTitle}>Ready to analyze your form?</Text>
            <Text style={styles.actionDescription}>
              Record a video of yourself performing this exercise to get AI-powered feedback
            </Text>
            <Button
              title="Record Exercise"
              leftIcon={<Camera size={20} color="white" />}
              onPress={handleRecordVideo}
              style={styles.actionButton}
            />
          </Card>
        </View>
      </ScrollView>
    </SafeAreaView>
  )
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: "white",
  },
  container: {
    flex: 1,
    backgroundColor: colors.gray[50],
  },
  contentContainer: {
    paddingBottom: spacing.xl,
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
  image: {
    width: "100%",
    height: 250,
  },
  content: {
    padding: spacing.lg,
    gap: spacing.lg,
  },
  header: {
    marginBottom: spacing.md,
  },
  title: {
    fontSize: fontSizes.xxl,
    fontWeight: fontWeights.bold,
    color: colors.gray[900],
    marginBottom: spacing.sm,
  },
  metaContainer: {
    flexDirection: "row",
    alignItems: "center",
  },
  tag: {
    fontSize: fontSizes.sm,
    color: "white",
    paddingVertical: 4,
    paddingHorizontal: 12,
    borderRadius: borderRadius.full,
    marginRight: spacing.md,
    overflow: "hidden",
    fontWeight: fontWeights.medium,
  },
  category: {
    fontSize: fontSizes.md,
    color: colors.gray[600],
    textTransform: "capitalize",
  },
  descriptionCard: {
    padding: spacing.lg,
  },
  sectionTitle: {
    fontSize: fontSizes.lg,
    fontWeight: fontWeights.semibold,
    color: colors.gray[900],
    marginBottom: spacing.md,
  },
  description: {
    fontSize: fontSizes.md,
    color: colors.gray[700],
    lineHeight: 24,
  },
  targetsCard: {
    padding: spacing.lg,
  },
  musclesList: {
    gap: spacing.sm,
  },
  muscleItem: {
    flexDirection: "row",
    alignItems: "center",
  },
  muscleDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: colors.primary[500],
    marginRight: spacing.sm,
  },
  muscleName: {
    fontSize: fontSizes.md,
    color: colors.gray[700],
    textTransform: "capitalize",
  },
  actionCard: {
    padding: spacing.lg,
    alignItems: "center",
    backgroundColor: colors.primary[50],
  },
  actionTitle: {
    fontSize: fontSizes.lg,
    fontWeight: fontWeights.semibold,
    color: colors.primary[700],
    marginBottom: spacing.xs,
    textAlign: "center",
  },
  actionDescription: {
    fontSize: fontSizes.md,
    color: colors.primary[600],
    marginBottom: spacing.lg,
    textAlign: "center",
  },
  actionButton: {
    minWidth: 200,
  },
})

export default ExerciseDetailScreen
