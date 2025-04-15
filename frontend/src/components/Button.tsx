import type React from "react"
import {
  TouchableOpacity,
  Text,
  StyleSheet,
  ActivityIndicator,
  type TouchableOpacityProps,
  type ViewStyle,
  type TextStyle,
} from "react-native"
import { colors, fontSizes, fontWeights, borderRadius, spacing } from "../theme"

interface ButtonProps extends TouchableOpacityProps {
  title: string
  variant?: "primary" | "secondary" | "outline" | "ghost"
  size?: "sm" | "md" | "lg"
  isLoading?: boolean
  leftIcon?: React.ReactNode
  rightIcon?: React.ReactNode
  style?: ViewStyle
  textStyle?: TextStyle
}

export const Button: React.FC<ButtonProps> = ({
  title,
  variant = "primary",
  size = "md",
  isLoading = false,
  leftIcon,
  rightIcon,
  style,
  textStyle,
  disabled,
  ...rest
}) => {
  const getVariantStyle = (): ViewStyle => {
    switch (variant) {
      case "primary":
        return {
          backgroundColor: colors.primary[500],
          borderWidth: 0,
        }
      case "secondary":
        return {
          backgroundColor: colors.secondary[500],
          borderWidth: 0,
        }
      case "outline":
        return {
          backgroundColor: "transparent",
          borderWidth: 1,
          borderColor: colors.primary[500],
        }
      case "ghost":
        return {
          backgroundColor: "transparent",
          borderWidth: 0,
        }
      default:
        return {}
    }
  }

  const getTextColor = (): string => {
    switch (variant) {
      case "primary":
      case "secondary":
        return "white"
      case "outline":
      case "ghost":
        return colors.primary[500]
      default:
        return "white"
    }
  }

  const getSizeStyle = (): ViewStyle => {
    switch (size) {
      case "sm":
        return {
          paddingVertical: spacing.xs,
          paddingHorizontal: spacing.md,
        }
      case "md":
        return {
          paddingVertical: spacing.sm,
          paddingHorizontal: spacing.lg,
        }
      case "lg":
        return {
          paddingVertical: spacing.md,
          paddingHorizontal: spacing.xl,
        }
      default:
        return {}
    }
  }

  const getTextSize = (): number => {
    switch (size) {
      case "sm":
        return fontSizes.sm
      case "md":
        return fontSizes.md
      case "lg":
        return fontSizes.lg
      default:
        return fontSizes.md
    }
  }

  return (
    <TouchableOpacity
      style={[styles.button, getVariantStyle(), getSizeStyle(), disabled && styles.disabled, style]}
      disabled={disabled || isLoading}
      {...rest}
    >
      {isLoading ? (
        <ActivityIndicator color={getTextColor()} size="small" />
      ) : (
        <>
          {leftIcon && <>{leftIcon}</>}
          <Text style={[styles.text, { color: getTextColor(), fontSize: getTextSize() }, textStyle]}>{title}</Text>
          {rightIcon && <>{rightIcon}</>}
        </>
      )}
    </TouchableOpacity>
  )
}

const styles = StyleSheet.create({
  button: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    borderRadius: borderRadius.md,
    gap: spacing.sm,
  },
  text: {
    fontWeight: fontWeights.medium,
    textAlign: "center",
  },
  disabled: {
    opacity: 0.5,
  },
})
