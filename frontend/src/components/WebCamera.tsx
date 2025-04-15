"use client"

import { View, Text, StyleSheet } from "react-native"
import { colors } from "../theme"

// Mock Camera component for web to avoid server-side rendering issues
export const WebCamera = ({ style, children }: any) => {
  return (
    <View style={[styles.container, style]}>
      <View style={styles.content}>
        <Text style={styles.text}>Camera Preview</Text>
        <Text style={styles.subtext}>Camera is only available on mobile devices</Text>
      </View>
      {children}
    </View>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "black",
    justifyContent: "center",
    alignItems: "center",
  },
  content: {
    padding: 20,
    backgroundColor: "rgba(0, 0, 0, 0.7)",
    borderRadius: 8,
    alignItems: "center",
  },
  text: {
    color: "white",
    fontSize: 18,
    fontWeight: "600",
    marginBottom: 8,
  },
  subtext: {
    color: colors.gray[400],
    fontSize: 14,
    textAlign: "center",
  },
})

// Mock the Camera API and constants
WebCamera.Constants = {
  Type: {
    back: "back",
    front: "front",
  },
  VideoQuality: {
    "2160p": "2160p",
    "1080p": "1080p",
    "720p": "720p",
    "480p": "480p",
    "4:3": "4:3",
  },
}

// Add mock methods
WebCamera.prototype.recordAsync = async () => {
  console.log("Camera.recordAsync - Mock implementation for web")
  return { uri: "mock-video-uri.mp4" }
}

WebCamera.prototype.stopRecording = () => {
  console.log("Camera.stopRecording - Mock implementation for web")
}

WebCamera.requestCameraPermissionsAsync = async () => {
  console.log("Camera.requestCameraPermissionsAsync - Mock implementation for web")
  return { status: "granted" }
}

WebCamera.requestMicrophonePermissionsAsync = async () => {
  console.log("Camera.requestMicrophonePermissionsAsync - Mock implementation for web")
  return { status: "granted" }
}

export default WebCamera 