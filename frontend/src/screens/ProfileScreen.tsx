"use client"

import { useState, useEffect } from "react"
import { View, Text, StyleSheet, ScrollView, Image, TouchableOpacity, SafeAreaView, StatusBar } from "react-native"
import { useNavigation } from "@react-navigation/native"
import type { NativeStackNavigationProp } from "@react-navigation/native-stack"
import { Settings, Calendar, Award, TrendingUp } from "lucide-react-native"

import type { RootStackParamList, User, Analysis } from "../types"
import { colors, spacing, fontSizes, fontWeights, borderRadius } from "../theme"
import { api } from "../services/api"
import { Card } from "../components/Card"
import { ProgressBar } from "../components/ProgressBar"
import { AnalysisCard } from "../components/AnalysisCard"
import { SkeletonView } from "../components/SkeletonView"

type ProfileScreenNavigationProp = NativeStackNavigationProp<RootStackParamList>

const ProfileScreen = () => {
  const navigation = useNavigation<ProfileScreenNavigationProp>()
  const [user, setUser] = useState<User | null>(null)
  const [analyses, setAnalyses] = useState<Analysis[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        const userData = await api.getUser()
        setUser(userData)

        const analysesData = await api.getAnalyses(userData.id)
        // Sort by date (newest first)
        const sortedAnalyses = [...analysesData].sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())
        setAnalyses(sortedAnalyses)
      } catch (error) {
        console.error("Error fetching profile data:", error)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [])

  const handleViewAnalysis = (analysisId: string) => {
    navigation.navigate("AnalysisResult", { analysisId })
  }

  const handleSettingsPress = () => {
    navigation.navigate("Settings")
  }

  // Calculate average score
  const averageScore = analyses.length ? Math.round(analyses.reduce((sum, a) => sum + a.score, 0) / analyses.length) : 0

  // Get most recent analysis
  const latestAnalysis = analyses.length > 0 ? analyses[0] : null

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar barStyle="dark-content" backgroundColor="white" />
      <View style={styles.container}>
        <View style={styles.header}>
          <Text style={styles.title}>Profile</Text>
          <TouchableOpacity onPress={handleSettingsPress} style={styles.settingsButton}>
            <Settings size={24} color={colors.gray[700]} />
          </TouchableOpacity>
        </View>

        <ScrollView contentContainerStyle={styles.scrollContent}>
          {loading ? (
            <View style={styles.loadingContainer}>
              <SkeletonView height={120} style={styles.skeleton} />
              <SkeletonView height={200} style={styles.skeleton} />
              <SkeletonView height={200} style={styles.skeleton} />
            </View>
          ) : (
            <>
              <View style={styles.profileSection}>
                <View style={styles.profileHeader}>
                  {user?.profileImage ? (
                    <Image source={{ uri: user.profileImage }} style={styles.profileImage} />
                  ) : (
                    <View style={styles.profileImagePlaceholder}>
                      <Text style={styles.profileImageText}>
                        {user?.name
                          .split(" ")
                          .map((n) => n[0])
                          .join("")}
                      </Text>
                    </View>
                  )}
                  <View style={styles.profileInfo}>
                    <Text style={styles.profileName}>{user?.name}</Text>
                    <Text style={styles.profileEmail}>{user?.email}</Text>
                    <Text style={styles.profileJoined}>
                      Joined {new Date(user?.joinedDate || "").toLocaleDateString()}
                    </Text>
                  </View>
                </View>
              </View>

              <View style={styles.statsSection}>
                <Text style={styles.sectionTitle}>Your Stats</Text>
                <View style={styles.statsGrid}>
                  <Card style={styles.statCard}>
                    <Calendar size={24} color={colors.primary[500]} />
                    <Text style={styles.statValue}>{user?.stats.totalWorkouts || 0}</Text>
                    <Text style={styles.statLabel}>Workouts</Text>
                  </Card>
                  <Card style={styles.statCard}>
                    <TrendingUp size={24} color={colors.primary[500]} />
                    <Text style={styles.statValue}>{user?.stats.totalMinutes || 0}</Text>
                    <Text style={styles.statLabel}>Minutes</Text>
                  </Card>
                  <Card style={styles.statCard}>
                    <Award size={24} color={colors.primary[500]} />
                    <Text style={styles.statValue}>{averageScore}</Text>
                    <Text style={styles.statLabel}>Avg. Score</Text>
                  </Card>
                </View>
              </View>

              {latestAnalysis && (
                <View style={styles.progressSection}>
                  <Text style={styles.sectionTitle}>Latest Performance</Text>
                  <Card style={styles.progressCard}>
                    <Text style={styles.progressTitle}>{latestAnalysis.exercise.name}</Text>
                    <Text style={styles.progressDate}>
                      {new Date(latestAnalysis.date).toLocaleDateString()} at{" "}
                      {new Date(latestAnalysis.date).toLocaleTimeString([], {
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </Text>
                    <View style={styles.scoreContainer}>
                      <Text style={styles.scoreLabel}>Score</Text>
                      <ProgressBar progress={latestAnalysis.score} height={12} showLabel style={styles.progressBar} />
                    </View>
                    <TouchableOpacity style={styles.viewButton} onPress={() => handleViewAnalysis(latestAnalysis.id)}>
                      <Text style={styles.viewButtonText}>View Details</Text>
                    </TouchableOpacity>
                  </Card>
                </View>
              )}

              <View style={styles.historySection}>
                <Text style={styles.sectionTitle}>Exercise History</Text>
                {analyses.length > 0 ? (
                  analyses.map((analysis) => (
                    <AnalysisCard
                      key={analysis.id}
                      analysis={analysis}
                      onPress={() => handleViewAnalysis(analysis.id)}
                    />
                  ))
                ) : (
                  <Card style={styles.emptyCard}>
                    <Text style={styles.emptyText}>No exercise history yet</Text>
                  </Card>
                )}
              </View>
            </>
          )}
        </ScrollView>
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
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    backgroundColor: "white",
    borderBottomWidth: 1,
    borderBottomColor: colors.gray[200],
  },
  title: {
    fontSize: fontSizes.xxl,
    fontWeight: fontWeights.bold,
    color: colors.gray[900],
  },
  settingsButton: {
    padding: spacing.xs,
  },
  scrollContent: {
    padding: spacing.md,
    gap: spacing.lg,
  },
  loadingContainer: {
    gap: spacing.md,
  },
  skeleton: {
    borderRadius: borderRadius.md,
    marginBottom: spacing.md,
  },
  profileSection: {
    backgroundColor: "white",
    borderRadius: borderRadius.md,
    padding: spacing.md,
    marginBottom: spacing.md,
  },
  profileHeader: {
    flexDirection: "row",
    alignItems: "center",
  },
  profileImage: {
    width: 80,
    height: 80,
    borderRadius: 40,
    marginRight: spacing.md,
  },
  profileImagePlaceholder: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: colors.primary[500],
    justifyContent: "center",
    alignItems: "center",
    marginRight: spacing.md,
  },
  profileImageText: {
    color: "white",
    fontSize: fontSizes.xl,
    fontWeight: fontWeights.bold,
  },
  profileInfo: {
    flex: 1,
  },
  profileName: {
    fontSize: fontSizes.xl,
    fontWeight: fontWeights.bold,
    color: colors.gray[900],
    marginBottom: spacing.xs,
  },
  profileEmail: {
    fontSize: fontSizes.md,
    color: colors.gray[600],
    marginBottom: spacing.xs,
  },
  profileJoined: {
    fontSize: fontSizes.sm,
    color: colors.gray[500],
  },
  sectionTitle: {
    fontSize: fontSizes.lg,
    fontWeight: fontWeights.semibold,
    color: colors.gray[900],
    marginBottom: spacing.md,
  },
  statsSection: {
    marginBottom: spacing.lg,
  },
  statsGrid: {
    flexDirection: "row",
    justifyContent: "space-between",
    gap: spacing.sm,
  },
  statCard: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    padding: spacing.md,
    gap: spacing.xs,
  },
  statValue: {
    fontSize: fontSizes.xl,
    fontWeight: fontWeights.bold,
    color: colors.gray[900],
  },
  statLabel: {
    fontSize: fontSizes.sm,
    color: colors.gray[600],
  },
  progressSection: {
    marginBottom: spacing.lg,
  },
  progressCard: {
    padding: spacing.lg,
  },
  progressTitle: {
    fontSize: fontSizes.lg,
    fontWeight: fontWeights.semibold,
    color: colors.gray[900],
    marginBottom: spacing.xs,
  },
  progressDate: {
    fontSize: fontSizes.sm,
    color: colors.gray[600],
    marginBottom: spacing.md,
  },
  scoreContainer: {
    marginBottom: spacing.md,
  },
  scoreLabel: {
    fontSize: fontSizes.md,
    fontWeight: fontWeights.medium,
    color: colors.gray[800],
    marginBottom: spacing.xs,
  },
  progressBar: {
    marginTop: spacing.xs,
  },
  viewButton: {
    alignSelf: "flex-start",
    paddingVertical: spacing.xs,
    paddingHorizontal: spacing.md,
    backgroundColor: colors.primary[500],
    borderRadius: borderRadius.md,
  },
  viewButtonText: {
    color: "white",
    fontSize: fontSizes.sm,
    fontWeight: fontWeights.medium,
  },
  historySection: {
    gap: spacing.md,
  },
  emptyCard: {
    padding: spacing.lg,
    alignItems: "center",
    justifyContent: "center",
  },
  emptyText: {
    fontSize: fontSizes.md,
    color: colors.gray[600],
    textAlign: "center",
  },
})

export default ProfileScreen
