"use client"

import { useState, useEffect } from "react"
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  RefreshControl,
  SafeAreaView,
  StatusBar,
} from "react-native"
import { useNavigation } from "@react-navigation/native"
import type { NativeStackNavigationProp } from "@react-navigation/native-stack"
import { Camera, Plus, Upload } from "lucide-react-native"

import type { RootStackParamList, Analysis, User } from "../types"
import { colors, spacing, fontSizes, fontWeights, borderRadius } from "../theme"
import { api } from "../services/api"
import { Card } from "../components/Card"
import { Button } from "../components/Button"
import { AnalysisCard } from "../components/AnalysisCard"
import { SkeletonView } from "../components/SkeletonView"

type HomeScreenNavigationProp = NativeStackNavigationProp<RootStackParamList>

const HomeScreen = () => {
  const navigation = useNavigation<HomeScreenNavigationProp>()
  const [user, setUser] = useState<User | null>(null)
  const [recentAnalyses, setRecentAnalyses] = useState<Analysis[]>([])
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)

  const fetchData = async () => {
    try {
      setLoading(true)
      const userData = await api.getUser()
      setUser(userData)

      const analyses = await api.getAnalyses(userData.id)
      // Sort by date (newest first)
      const sortedAnalyses = [...analyses].sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())
      setRecentAnalyses(sortedAnalyses)
    } catch (error) {
      console.error("Error fetching data:", error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
  }, [])

  const onRefresh = async () => {
    setRefreshing(true)
    await fetchData()
    setRefreshing(false)
  }

  const handleRecordVideo = () => {
    navigation.navigate("RecordVideo")
  }

  const handleUploadVideo = () => {
    navigation.navigate("UploadVideo")
  }

  const handleViewAnalysis = (analysisId: string) => {
    navigation.navigate("AnalysisResult", { analysisId })
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar barStyle="dark-content" backgroundColor="white" />
      <View style={styles.container}>
        <View style={styles.header}>
          <Text style={styles.title}>Muscle Memory</Text>
          {user && (
            <TouchableOpacity onPress={() => navigation.navigate("Settings")}>
              <View style={styles.avatar}>
                <Text style={styles.avatarText}>
                  {user.name
                    .split(" ")
                    .map((n) => n[0])
                    .join("")}
                </Text>
              </View>
            </TouchableOpacity>
          )}
        </View>

        <ScrollView
          contentContainerStyle={styles.scrollContent}
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
        >
          <Card style={styles.actionsCard}>
            <Text style={styles.sectionTitle}>Record New Exercise</Text>
            <View style={styles.actionButtons}>
              <Button
                title="Record Video"
                leftIcon={<Camera size={20} color="white" />}
                onPress={handleRecordVideo}
                style={styles.actionButton}
              />
              <Button
                title="Upload Video"
                variant="outline"
                leftIcon={<Upload size={20} color={colors.primary[500]} />}
                onPress={handleUploadVideo}
                style={styles.actionButton}
              />
            </View>
          </Card>

          <View style={styles.statsSection}>
            <Text style={styles.sectionTitle}>Your Stats</Text>
            {loading ? (
              <View style={styles.statsCards}>
                <SkeletonView height={100} style={styles.statCardSkeleton} />
                <SkeletonView height={100} style={styles.statCardSkeleton} />
                <SkeletonView height={100} style={styles.statCardSkeleton} />
              </View>
            ) : (
              <View style={styles.statsCards}>
                <Card style={styles.statCard}>
                  <Text style={styles.statValue}>{user?.stats.totalWorkouts || 0}</Text>
                  <Text style={styles.statLabel}>Workouts</Text>
                </Card>
                <Card style={styles.statCard}>
                  <Text style={styles.statValue}>{user?.stats.totalMinutes || 0}</Text>
                  <Text style={styles.statLabel}>Minutes</Text>
                </Card>
                <Card style={styles.statCard}>
                  <Text style={styles.statValue}>{user?.stats.streak || 0}</Text>
                  <Text style={styles.statLabel}>Day Streak</Text>
                </Card>
              </View>
            )}
          </View>

          <View style={styles.recentSection}>
            <Text style={styles.sectionTitle}>Recent Analyses</Text>
            {loading ? (
              <>
                <SkeletonView height={200} style={styles.analysisSkeleton} />
                <SkeletonView height={200} style={styles.analysisSkeleton} />
              </>
            ) : recentAnalyses.length > 0 ? (
              recentAnalyses.map((analysis) => (
                <AnalysisCard key={analysis.id} analysis={analysis} onPress={() => handleViewAnalysis(analysis.id)} />
              ))
            ) : (
              <Card style={styles.emptyCard}>
                <Text style={styles.emptyText}>No analyses yet</Text>
                <Button
                  title="Record Your First Exercise"
                  leftIcon={<Plus size={20} color="white" />}
                  onPress={handleRecordVideo}
                  style={styles.emptyButton}
                />
              </Card>
            )}
          </View>
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
  avatar: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: colors.primary[500],
    justifyContent: "center",
    alignItems: "center",
  },
  avatarText: {
    color: "white",
    fontSize: fontSizes.md,
    fontWeight: fontWeights.bold,
  },
  scrollContent: {
    padding: spacing.md,
    gap: spacing.lg,
  },
  actionsCard: {
    marginBottom: spacing.md,
  },
  sectionTitle: {
    fontSize: fontSizes.lg,
    fontWeight: fontWeights.semibold,
    color: colors.gray[900],
    marginBottom: spacing.md,
  },
  actionButtons: {
    flexDirection: "row",
    gap: spacing.md,
  },
  actionButton: {
    flex: 1,
  },
  statsSection: {
    marginBottom: spacing.lg,
  },
  statsCards: {
    flexDirection: "row",
    justifyContent: "space-between",
    gap: spacing.sm,
  },
  statCard: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    padding: spacing.md,
  },
  statCardSkeleton: {
    flex: 1,
    marginHorizontal: 4,
    borderRadius: borderRadius.md,
  },
  statValue: {
    fontSize: fontSizes.xxl,
    fontWeight: fontWeights.bold,
    color: colors.primary[500],
    marginBottom: spacing.xs,
  },
  statLabel: {
    fontSize: fontSizes.sm,
    color: colors.gray[600],
  },
  recentSection: {
    gap: spacing.md,
  },
  analysisSkeleton: {
    marginBottom: spacing.md,
    borderRadius: borderRadius.md,
  },
  emptyCard: {
    alignItems: "center",
    justifyContent: "center",
    padding: spacing.xl,
  },
  emptyText: {
    fontSize: fontSizes.md,
    color: colors.gray[600],
    marginBottom: spacing.md,
  },
  emptyButton: {
    minWidth: 200,
  },
})

export default HomeScreen
