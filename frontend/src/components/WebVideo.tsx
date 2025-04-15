"use client"

import React from "react"
import { View, Text, StyleSheet, Image } from "react-native"
import { colors } from "../theme"

// Mock Video component for web to avoid server-side rendering issues
export const WebVideo = React.forwardRef(({ style, source, resizeMode, isLooping, onPlaybackStatusUpdate }: any, ref) => {
  // Simulate status update on mount
  React.useEffect(() => {
    if (onPlaybackStatusUpdate) {
      onPlaybackStatusUpdate({
        isLoaded: true,
        isPlaying: false,
      })
    }
  }, [onPlaybackStatusUpdate])

  // Add methods to the ref
  React.useImperativeHandle(ref, () => ({
    pauseAsync: async () => {
      console.log("Video.pauseAsync - Mock implementation for web")
      if (onPlaybackStatusUpdate) {
        onPlaybackStatusUpdate({
          isLoaded: true,
          isPlaying: false,
        })
      }
      return {}
    },
    playAsync: async () => {
      console.log("Video.playAsync - Mock implementation for web")
      if (onPlaybackStatusUpdate) {
        onPlaybackStatusUpdate({
          isLoaded: true,
          isPlaying: true,
        })
      }
      return {}
    },
    replayAsync: async () => {
      console.log("Video.replayAsync - Mock implementation for web")
      if (onPlaybackStatusUpdate) {
        onPlaybackStatusUpdate({
          isLoaded: true,
          isPlaying: true,
        })
      }
      return {}
    },
  }))

  return (
    <View style={[styles.container, style]}>
      <View style={styles.content}>
        <Text style={styles.text}>Video Preview</Text>
        <Text style={styles.subtext}>Video preview is only available on mobile devices</Text>
      </View>
    </View>
  )
})

WebVideo.displayName = "WebVideo"

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

export default WebVideo 