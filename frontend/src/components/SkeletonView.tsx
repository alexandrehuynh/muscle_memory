"use client"

import type React from "react"
import { useEffect, useRef } from "react"
import { StyleSheet, Animated, Platform, type ViewStyle } from "react-native"
import { colors } from "../theme"

interface SkeletonViewProps {
  width?: number | string
  height?: number | string
  borderRadius?: number
  style?: ViewStyle
}

export const SkeletonView: React.FC<SkeletonViewProps> = ({ width = "100%", height = 20, borderRadius = 4, style }) => {
  const opacity = useRef(new Animated.Value(0.3)).current

  useEffect(() => {
    const animation = Animated.loop(
      Animated.sequence([
        Animated.timing(opacity, {
          toValue: 0.6,
          duration: 800,
          useNativeDriver: Platform.OS !== 'web',
        }),
        Animated.timing(opacity, {
          toValue: 0.3,
          duration: 800,
          useNativeDriver: Platform.OS !== 'web',
        }),
      ]),
    )

    animation.start()

    return () => {
      animation.stop()
    }
  }, [opacity])

  return (
    <Animated.View
      style={[
        styles.skeleton,
        {
          width: typeof width === 'number' ? width : width,
          height: typeof height === 'number' ? height : height,
          borderRadius,
          opacity,
        } as any,
        style,
      ]}
    />
  )
}

const styles = StyleSheet.create({
  skeleton: {
    backgroundColor: colors.gray[300],
  },
})
