import type React from "react"
import { View, StyleSheet, Text, type ViewStyle } from "react-native"
import { colors, borderRadius, fontSizes } from "../theme"

interface ProgressBarProps {
  progress: number // 0 to 100
  height?: number
  showLabel?: boolean
  color?: string
  style?: ViewStyle
}

export const ProgressBar: React.FC<ProgressBarProps> = ({
  progress,
  height = 8,
  showLabel = false,
  color = colors.primary[500],
  style,
}) => {
  // Ensure progress is between 0 and 100
  const clampedProgress = Math.min(Math.max(progress, 0), 100)

  return (
    <View style={[styles.container, style]}>
      {showLabel && <Text style={styles.label}>{Math.round(clampedProgress)}%</Text>}
      <View style={[styles.track, { height }]}>
        <View
          style={[
            styles.progress,
            {
              width: `${clampedProgress}%`,
              height,
              backgroundColor: color,
            },
          ]}
        />
      </View>
    </View>
  )
}

const styles = StyleSheet.create({
  container: {
    width: "100%",
  },
  label: {
    fontSize: fontSizes.sm,
    marginBottom: 4,
    color: colors.gray[700],
  },
  track: {
    width: "100%",
    backgroundColor: colors.gray[200],
    borderRadius: borderRadius.full,
    overflow: "hidden",
  },
  progress: {
    borderRadius: borderRadius.full,
  },
})
