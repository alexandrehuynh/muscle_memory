"use client"

import { useState } from "react"
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  SafeAreaView,
  StatusBar,
  Image,
  Alert,
  ActivityIndicator,
  ScrollView,
  Platform,
} from "react-native"
import { useNavigation } from "@react-navigation/native"
import type { NativeStackNavigationProp } from "@react-navigation/native-stack"
import * as ImagePicker from "expo-image-picker"
import { Upload, X, Check } from "lucide-react-native"

import type { RootStackParamList, Exercise } from "../types"
import { colors, spacing, fontSizes, fontWeights, borderRadius } from "../theme"
import { api } from "../services/api"
import { Button } from "../components/Button"
import { Card } from "../components/Card"
import { ExerciseCard } from "../components/ExerciseCard"

type UploadVideoScreenNavigationProp = NativeStackNavigationProp<RootStackParamList>

const UploadVideoScreen = () => {
  const navigation = useNavigation<UploadVideoScreenNavigationProp>()
  const [videoUri, setVideoUri] = useState<string | null>(null)
  const [selectedExercise, setSelectedExercise] = useState<Exercise | null>(null)
  const [exercises, setExercises] = useState<Exercise[]>([])
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [step, setStep] = useState<"select_video" | "select_exercise" | "confirm">("select_video")

  const pickVideo = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Videos,
        allowsEditing: true,
        aspect: [16, 9],
        quality: 1,
      })

      if (!result.canceled && result.assets && result.assets.length > 0) {
        setVideoUri(result.assets[0].uri)

        // Fetch exercises for the next step
        setLoading(true)
        try {
          const exercisesData = await api.getExercises()
          setExercises(exercisesData)
        } catch (error) {
          console.error("Error fetching exercises:", error)
          Alert.alert("Error", "Failed to load exercises")
        } finally {
          setLoading(false)
        }

        setStep("select_exercise")
      }
    } catch (error) {
      console.error("Error picking video:", error)
      Alert.alert("Error", "Failed to pick video from gallery")
    }
  }

  const handleSelectExercise = (exercise: Exercise) => {
    setSelectedExercise(exercise)
    setStep("confirm")
  }

  const handleUpload = async () => {
    if (!videoUri || !selectedExercise) return

    try {
      setUploading(true)

      // Upload and analyze
      const analysis = await api.uploadVideo(selectedExercise.id, videoUri)

      // Navigate to results
      navigation.replace("AnalysisResult", { analysisId: analysis.id })
    } catch (error) {
      console.error("Error processing video:", error)
      Alert.alert("Error", "Failed to process video")
      setUploading(false)
    }
  }

  const handleBack = () => {
    if (step === "select_exercise") {
      setStep("select_video")
    } else if (step === "confirm") {
      setStep("select_exercise")
    }
  }

  const handleCancel = () => {
    navigation.goBack()
  }

  const renderSelectVideo = () => (
    <View style={styles.centeredContainer}>
      <View style={styles.uploadBox}>
        <Upload size={48} color={colors.primary[500]} />
        <Text style={styles.uploadTitle}>Upload Exercise Video</Text>
        <Text style={styles.uploadDescription}>Select a video from your gallery to analyze your exercise form</Text>
        <Button title="Select Video" onPress={pickVideo} style={styles.uploadButton} />
      </View>
    </View>
  )

  const renderSelectExercise = () => (
    <View style={styles.exerciseContainer}>
      <Text style={styles.stepTitle}>Select Exercise Type</Text>
      <Text style={styles.stepDescription}>Choose the exercise performed in your video</Text>

      {loading ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.primary[500]} />
          <Text style={styles.loadingText}>Loading exercises...</Text>
        </View>
      ) : (
        <ScrollView style={styles.exerciseList}>
          {exercises.map((exercise) => (
            <TouchableOpacity key={exercise.id} onPress={() => handleSelectExercise(exercise)}>
              <ExerciseCard exercise={exercise} compact />
            </TouchableOpacity>
          ))}
        </ScrollView>
      )}

      <View style={styles.actionButtons}>
        <Button title="Back" variant="outline" onPress={handleBack} style={styles.actionButton} />
        <Button title="Cancel" variant="ghost" onPress={handleCancel} style={styles.actionButton} />
      </View>
    </View>
  )

  const renderConfirm = () => (
    <View style={styles.confirmContainer}>
      <Text style={styles.stepTitle}>Confirm Upload</Text>

      <Card style={styles.videoPreview}>
        {videoUri && <Image source={{ uri: videoUri }} style={styles.previewImage} resizeMode="cover" />}
      </Card>

      {selectedExercise && (
        <View style={styles.exerciseInfo}>
          <Text style={styles.exerciseInfoTitle}>Exercise:</Text>
          <Text style={styles.exerciseInfoName}>{selectedExercise.name}</Text>
          <Text style={styles.exerciseInfoDescription}>{selectedExercise.description}</Text>
        </View>
      )}

      <View style={styles.actionButtons}>
        {uploading ? (
          <View style={styles.uploadingContainer}>
            <ActivityIndicator size="large" color={colors.primary[500]} />
            <Text style={styles.uploadingText}>Processing video...</Text>
          </View>
        ) : (
          <>
            <Button
              title="Back"
              variant="outline"
              leftIcon={<X size={20} color={colors.primary[500]} />}
              onPress={handleBack}
              style={styles.actionButton}
            />
            <Button
              title="Upload & Analyze"
              leftIcon={<Check size={20} color="white" />}
              onPress={handleUpload}
              style={styles.actionButton}
            />
          </>
        )}
      </View>
    </View>
  )

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar barStyle="dark-content" backgroundColor="white" />
      <View style={styles.container}>
        {step === "select_video" && renderSelectVideo()}
        {step === "select_exercise" && renderSelectExercise()}
        {step === "confirm" && renderConfirm()}
      </View>
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
  centeredContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: spacing.lg,
  },
  uploadBox: {
    width: "100%",
    padding: spacing.xl,
    backgroundColor: "white",
    borderRadius: borderRadius.md,
    alignItems: "center",
    justifyContent: "center",
    ...Platform.select({
      ios: {
        shadowColor: colors.gray[900],
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 4,
      },
      android: {
        elevation: 3,
      },
    }),
  },
  uploadTitle: {
    fontSize: fontSizes.xl,
    fontWeight: fontWeights.bold,
    color: colors.gray[900],
    marginTop: spacing.md,
    marginBottom: spacing.sm,
  },
  uploadDescription: {
    fontSize: fontSizes.md,
    color: colors.gray[600],
    textAlign: "center",
    marginBottom: spacing.lg,
  },
  uploadButton: {
    minWidth: 200,
  },
  exerciseContainer: {
    flex: 1,
    padding: spacing.lg,
  },
  stepTitle: {
    fontSize: fontSizes.xl,
    fontWeight: fontWeights.bold,
    color: colors.gray[900],
    marginBottom: spacing.sm,
  },
  stepDescription: {
    fontSize: fontSizes.md,
    color: colors.gray[600],
    marginBottom: spacing.lg,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  loadingText: {
    fontSize: fontSizes.md,
    color: colors.gray[600],
    marginTop: spacing.md,
  },
  exerciseList: {
    flex: 1,
    marginBottom: spacing.lg,
  },
  actionButtons: {
    flexDirection: "row",
    justifyContent: "space-between",
    gap: spacing.md,
  },
  actionButton: {
    flex: 1,
  },
  confirmContainer: {
    flex: 1,
    padding: spacing.lg,
  },
  videoPreview: {
    marginVertical: spacing.lg,
    padding: 0,
    overflow: "hidden",
  },
  previewImage: {
    width: "100%",
    height: 200,
  },
  exerciseInfo: {
    backgroundColor: "white",
    borderRadius: borderRadius.md,
    padding: spacing.lg,
    marginBottom: spacing.lg,
  },
  exerciseInfoTitle: {
    fontSize: fontSizes.md,
    fontWeight: fontWeights.medium,
    color: colors.gray[600],
    marginBottom: spacing.xs,
  },
  exerciseInfoName: {
    fontSize: fontSizes.lg,
    fontWeight: fontWeights.bold,
    color: colors.gray[900],
    marginBottom: spacing.sm,
  },
  exerciseInfoDescription: {
    fontSize: fontSizes.md,
    color: colors.gray[700],
  },
  uploadingContainer: {
    flex: 1,
    flexDirection: "row",
    justifyContent: "center",
    alignItems: "center",
    gap: spacing.md,
  },
  uploadingText: {
    fontSize: fontSizes.md,
    color: colors.gray[700],
  },
})

export default UploadVideoScreen
