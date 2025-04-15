"use client"

import { useState, useRef, useEffect } from "react"
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  SafeAreaView,
  StatusBar,
  Alert,
  ActivityIndicator,
  Platform,
} from "react-native"
import { useNavigation } from "@react-navigation/native"
import type { NativeStackNavigationProp } from "@react-navigation/native-stack"
import { Pause, Play, RotateCcw, Check, X } from "lucide-react-native"

import type { RootStackParamList } from "../types"
import { colors, spacing, fontSizes } from "../theme"
import { api } from "../services/api"
import { Button } from "../components/Button"
import { ClientOnly } from "../components/ClientOnly"

// Import the real components only on mobile platforms
let Camera: any
let Video: any
let MediaLibrary: any

// Use mock components on web platform
if (Platform.OS === 'web') {
  const WebCamera = require('../components/WebCamera').default
  const WebVideo = require('../components/WebVideo').default
  const WebMediaLibrary = require('../components/WebMediaLibrary').default
  Camera = WebCamera
  Video = WebVideo
  MediaLibrary = WebMediaLibrary
} else {
  // Use real components on mobile
  Camera = require('expo-camera').Camera
  Video = require('expo-av').Video
  MediaLibrary = require('expo-media-library')
}

type RecordVideoScreenNavigationProp = NativeStackNavigationProp<RootStackParamList>

const RecordVideoScreen = () => {
  const navigation = useNavigation<RecordVideoScreenNavigationProp>()
  const [hasPermission, setHasPermission] = useState<boolean | null>(null)
  const [cameraType, setCameraType] = useState(Camera.Constants.Type.back)
  const [isRecording, setIsRecording] = useState(false)
  const [videoUri, setVideoUri] = useState<string | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const cameraRef = useRef<typeof Camera>(null)
  const videoRef = useRef<typeof Video>(null)

  useEffect(() => {
    ;(async () => {
      const { status: cameraStatus } = await Camera.requestCameraPermissionsAsync()
      const { status: audioStatus } = await Camera.requestMicrophonePermissionsAsync()
      const { status: mediaStatus } = await MediaLibrary.requestPermissionsAsync()

      setHasPermission(cameraStatus === "granted" && audioStatus === "granted" && mediaStatus === "granted")
    })()
  }, [])

  const handleStartRecording = async () => {
    if (cameraRef.current) {
      setIsRecording(true)
      try {
        const videoRecording = await cameraRef.current.recordAsync({
          maxDuration: 30, // 30 seconds max
          quality: Camera.Constants.VideoQuality["720p"],
        })
        setVideoUri(videoRecording.uri)
      } catch (error) {
        console.error("Error recording video:", error)
        Alert.alert("Error", "Failed to record video")
      }
      setIsRecording(false)
    }
  }

  const handleStopRecording = () => {
    if (cameraRef.current && isRecording) {
      cameraRef.current.stopRecording()
    }
  }

  const handleFlipCamera = () => {
    setCameraType(cameraType === Camera.Constants.Type.back 
      ? Camera.Constants.Type.front 
      : Camera.Constants.Type.back)
  }

  const handlePlayPause = async () => {
    if (videoRef.current) {
      if (isPlaying) {
        await videoRef.current.pauseAsync()
      } else {
        await videoRef.current.replayAsync()
      }
      setIsPlaying(!isPlaying)
    }
  }

  const handleReset = () => {
    setVideoUri(null)
    setIsPlaying(false)
  }

  const handleSubmit = async () => {
    if (!videoUri) return

    try {
      setIsProcessing(true)

      // Save to media library
      await MediaLibrary.saveToLibraryAsync(videoUri)

      // Mock exercise ID for demo
      const exerciseId = "ex1" // Squat

      // Upload and analyze
      const analysis = await api.uploadVideo(exerciseId, videoUri)

      // Navigate to results
      navigation.replace("AnalysisResult", { analysisId: analysis.id })
    } catch (error) {
      console.error("Error processing video:", error)
      Alert.alert("Error", "Failed to process video")
      setIsProcessing(false)
    }
  }

  const handleCancel = () => {
    navigation.goBack()
  }

  if (hasPermission === null) {
    return (
      <View style={styles.centeredContainer}>
        <ActivityIndicator size="large" color={colors.primary[500]} />
        <Text style={styles.permissionText}>Requesting permissions...</Text>
      </View>
    )
  }

  if (hasPermission === false) {
    return (
      <View style={styles.centeredContainer}>
        <Text style={styles.permissionText}>Camera and microphone permissions are required to record videos.</Text>
        <Button title="Go Back" onPress={() => navigation.goBack()} style={{ marginTop: spacing.lg }} />
      </View>
    )
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar barStyle="light-content" backgroundColor="black" />
      <View style={styles.container}>
        {videoUri ? (
          // Video preview mode
          <View style={styles.previewContainer}>
            <ClientOnly fallback={<View style={styles.preview} />}>
              <Video
                ref={videoRef}
                source={{ uri: videoUri }}
                style={styles.preview}
                resizeMode={"cover" as any}
                isLooping
                onPlaybackStatusUpdate={(status) => {
                  if (status.isLoaded) {
                    setIsPlaying(status.isPlaying)
                  }
                }}
              />
            </ClientOnly>

            {isProcessing ? (
              <View style={styles.processingOverlay}>
                <ActivityIndicator size="large" color="white" />
                <Text style={styles.processingText}>Processing video...</Text>
              </View>
            ) : (
              <>
                <View style={styles.previewControls}>
                  <TouchableOpacity style={styles.controlButton} onPress={handlePlayPause}>
                    {isPlaying ? <Pause size={24} color="white" /> : <Play size={24} color="white" />}
                  </TouchableOpacity>
                  <TouchableOpacity style={styles.controlButton} onPress={handleReset}>
                    <RotateCcw size={24} color="white" />
                  </TouchableOpacity>
                </View>

                <View style={styles.actionButtons}>
                  <Button
                    title="Cancel"
                    variant="outline"
                    leftIcon={<X size={20} color={colors.primary[500]} />}
                    onPress={handleCancel}
                    style={styles.actionButton}
                    textStyle={{ color: "white" }}
                  />
                  <Button
                    title="Use Video"
                    leftIcon={<Check size={20} color="white" />}
                    onPress={handleSubmit}
                    style={styles.actionButton}
                  />
                </View>
              </>
            )}
          </View>
        ) : (
          // Camera recording mode
          <View style={styles.cameraContainer}>
            <ClientOnly fallback={<View style={styles.camera} />}>
              <Camera ref={cameraRef} style={styles.camera} type={cameraType} ratio="16:9" />
            </ClientOnly>

            <View style={styles.cameraControls}>
              <TouchableOpacity style={styles.flipButton} onPress={handleFlipCamera}>
                <Text style={styles.flipButtonText}>Flip</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[styles.recordButton, isRecording && styles.recordingButton]}
                onPress={isRecording ? handleStopRecording : handleStartRecording}
              >
                <View style={isRecording ? styles.stopIcon : styles.recordIcon} />
              </TouchableOpacity>

              <TouchableOpacity style={styles.cancelButton} onPress={handleCancel}>
                <Text style={styles.cancelButtonText}>Cancel</Text>
              </TouchableOpacity>
            </View>

            {isRecording && (
              <View style={styles.recordingIndicator}>
                <View style={styles.recordingDot} />
                <Text style={styles.recordingText}>Recording</Text>
              </View>
            )}
          </View>
        )}
      </View>
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
    backgroundColor: "black",
  },
  centeredContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: spacing.lg,
  },
  permissionText: {
    fontSize: fontSizes.md,
    color: colors.gray[800],
    textAlign: "center",
    marginTop: spacing.md,
  },
  cameraContainer: {
    flex: 1,
    position: "relative",
  },
  camera: {
    flex: 1,
  },
  cameraControls: {
    position: "absolute",
    bottom: 40,
    left: 0,
    right: 0,
    flexDirection: "row",
    justifyContent: "space-around",
    alignItems: "center",
    paddingHorizontal: spacing.lg,
  },
  flipButton: {
    padding: spacing.md,
  },
  flipButtonText: {
    color: "white",
    fontSize: fontSizes.md,
    fontWeight: "500",
  },
  recordButton: {
    width: 70,
    height: 70,
    borderRadius: 35,
    backgroundColor: "rgba(255, 255, 255, 0.3)",
    justifyContent: "center",
    alignItems: "center",
    borderWidth: 2,
    borderColor: "white",
  },
  recordingButton: {
    backgroundColor: "rgba(255, 0, 0, 0.3)",
  },
  recordIcon: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: colors.error,
  },
  stopIcon: {
    width: 30,
    height: 30,
    backgroundColor: "white",
  },
  cancelButton: {
    padding: spacing.md,
  },
  cancelButtonText: {
    color: "white",
    fontSize: fontSizes.md,
    fontWeight: "500",
  },
  recordingIndicator: {
    position: "absolute",
    top: 40,
    alignSelf: "center",
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "rgba(0, 0, 0, 0.5)",
    paddingVertical: spacing.xs,
    paddingHorizontal: spacing.md,
    borderRadius: 20,
  },
  recordingDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: colors.error,
    marginRight: spacing.xs,
  },
  recordingText: {
    color: "white",
    fontSize: fontSizes.sm,
    fontWeight: "500",
  },
  previewContainer: {
    flex: 1,
    position: "relative",
  },
  preview: {
    flex: 1,
  },
  previewControls: {
    position: "absolute",
    top: 40,
    left: 0,
    right: 0,
    flexDirection: "row",
    justifyContent: "center",
    gap: spacing.lg,
  },
  controlButton: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: "rgba(0, 0, 0, 0.5)",
    justifyContent: "center",
    alignItems: "center",
  },
  actionButtons: {
    position: "absolute",
    bottom: 40,
    left: 0,
    right: 0,
    flexDirection: "row",
    justifyContent: "center",
    gap: spacing.md,
    paddingHorizontal: spacing.lg,
  },
  actionButton: {
    flex: 1,
  },
  processingOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "rgba(0, 0, 0, 0.7)",
    justifyContent: "center",
    alignItems: "center",
  },
  processingText: {
    color: "white",
    fontSize: fontSizes.lg,
    fontWeight: "500",
    marginTop: spacing.md,
  },
})

export default RecordVideoScreen
