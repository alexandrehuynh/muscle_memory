import type React from "react"
import { View, StyleSheet } from "react-native"
import Svg, { Line, Circle } from "react-native-svg"
import type { SkeletonFrame, Keypoint } from "../types"
import { colors } from "../theme"

interface SkeletonRendererProps {
  frame: SkeletonFrame
  width: number
  height: number
  highlightJoints?: string[]
}

// Define the connections between keypoints to draw lines
const connections = [
  ["nose", "left_shoulder"],
  ["nose", "right_shoulder"],
  ["left_shoulder", "right_shoulder"],
  ["left_shoulder", "left_elbow"],
  ["right_shoulder", "right_elbow"],
  ["left_elbow", "left_wrist"],
  ["right_elbow", "right_wrist"],
  ["left_shoulder", "left_hip"],
  ["right_shoulder", "right_hip"],
  ["left_hip", "right_hip"],
  ["left_hip", "left_knee"],
  ["right_hip", "right_knee"],
  ["left_knee", "left_ankle"],
  ["right_knee", "right_ankle"],
]

export const SkeletonRenderer: React.FC<SkeletonRendererProps> = ({ frame, width, height, highlightJoints = [] }) => {
  // Function to find a keypoint by name
  const findKeypoint = (name: string): Keypoint | undefined => {
    return frame.keypoints.find((kp) => kp.name === name)
  }

  // Convert normalized coordinates to pixel values
  const toPixelX = (x: number) => x * width
  const toPixelY = (y: number) => y * height

  return (
    <View style={[styles.container, { width, height }]}>
      <Svg width={width} height={height}>
        {/* Draw connections (lines) */}
        {connections.map(([start, end], index) => {
          const startPoint = findKeypoint(start)
          const endPoint = findKeypoint(end)

          if (!startPoint || !endPoint || startPoint.confidence < 0.5 || endPoint.confidence < 0.5) {
            return null
          }

          const isHighlighted = highlightJoints.includes(start) || highlightJoints.includes(end)

          return (
            <Line
              key={`line-${index}`}
              x1={toPixelX(startPoint.x)}
              y1={toPixelY(startPoint.y)}
              x2={toPixelX(endPoint.x)}
              y2={toPixelY(endPoint.y)}
              stroke={isHighlighted ? colors.error : colors.primary[500]}
              strokeWidth={isHighlighted ? 3 : 2}
            />
          )
        })}

        {/* Draw keypoints (circles) */}
        {frame.keypoints.map((keypoint, index) => {
          if (keypoint.confidence < 0.5) return null

          const isHighlighted = highlightJoints.includes(keypoint.name)

          return (
            <Circle
              key={`keypoint-${index}`}
              cx={toPixelX(keypoint.x)}
              cy={toPixelY(keypoint.y)}
              r={isHighlighted ? 6 : 4}
              fill={isHighlighted ? colors.error : colors.primary[500]}
            />
          )
        })}
      </Svg>
    </View>
  )
}

const styles = StyleSheet.create({
  container: {
    position: "absolute",
    top: 0,
    left: 0,
  },
})
