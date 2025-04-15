import type React from "react"
import { View, StyleSheet, Platform, type ViewProps, type ViewStyle } from "react-native"
import { colors, borderRadius, shadows } from "../theme"

interface CardProps extends ViewProps {
  variant?: "elevated" | "outlined" | "filled"
  style?: ViewStyle
}

export const Card: React.FC<CardProps> = ({ variant = "elevated", style, children, ...rest }) => {
  const getVariantStyle = (): any => {
    switch (variant) {
      case "elevated":
        return Platform.OS === 'web'
          ? {
              backgroundColor: "white",
              boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)',
            } 
          : {
              backgroundColor: "white",
              ...shadows.md,
            }
      case "outlined":
        return {
          backgroundColor: "white",
          borderWidth: 1,
          borderColor: colors.gray[200],
        }
      case "filled":
        return {
          backgroundColor: colors.gray[100],
        }
      default:
        return {}
    }
  }

  return (
    <View style={[styles.card, getVariantStyle(), style]} {...rest}>
      {children}
    </View>
  )
}

const styles = StyleSheet.create({
  card: {
    borderRadius: borderRadius.md,
    padding: 16,
    overflow: "hidden",
  },
})
