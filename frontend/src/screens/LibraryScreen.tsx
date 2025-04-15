"use client"

import type React from "react"
import { useState, useEffect } from "react"
import { View, Text, StyleSheet, FlatList, TextInput, TouchableOpacity, SafeAreaView, StatusBar } from "react-native"
import { useNavigation } from "@react-navigation/native"
import type { NativeStackNavigationProp } from "@react-navigation/native-stack"
import { Search, X } from "lucide-react-native"

import type { RootStackParamList, Exercise } from "../types"
import { colors, spacing, fontSizes, fontWeights, borderRadius } from "../theme"
import { api } from "../services/api"
import { ExerciseCard } from "../components/ExerciseCard"
import { SkeletonView } from "../components/SkeletonView"

type LibraryScreenNavigationProp = NativeStackNavigationProp<RootStackParamList>

const CATEGORIES = ["all", "strength", "cardio", "flexibility", "balance"]
const DIFFICULTIES = ["all", "beginner", "intermediate", "advanced"]

const LibraryScreen = () => {
  const navigation = useNavigation<LibraryScreenNavigationProp>()
  const [exercises, setExercises] = useState<Exercise[]>([])
  const [filteredExercises, setFilteredExercises] = useState<Exercise[]>([])
  const [searchQuery, setSearchQuery] = useState("")
  const [selectedCategory, setSelectedCategory] = useState("all")
  const [selectedDifficulty, setSelectedDifficulty] = useState("all")
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchExercises = async () => {
      try {
        setLoading(true)
        const data = await api.getExercises()
        setExercises(data)
        setFilteredExercises(data)
      } catch (error) {
        console.error("Error fetching exercises:", error)
      } finally {
        setLoading(false)
      }
    }

    fetchExercises()
  }, [])

  useEffect(() => {
    filterExercises()
  }, [searchQuery, selectedCategory, selectedDifficulty, exercises])

  const filterExercises = () => {
    let filtered = [...exercises]

    // Filter by search query
    if (searchQuery) {
      const query = searchQuery.toLowerCase()
      filtered = filtered.filter(
        (exercise) =>
          exercise.name.toLowerCase().includes(query) ||
          exercise.description.toLowerCase().includes(query) ||
          exercise.targetMuscles.some((muscle) => muscle.toLowerCase().includes(query)),
      )
    }

    // Filter by category
    if (selectedCategory !== "all") {
      filtered = filtered.filter((exercise) => exercise.category === selectedCategory)
    }

    // Filter by difficulty
    if (selectedDifficulty !== "all") {
      filtered = filtered.filter((exercise) => exercise.difficulty === selectedDifficulty)
    }

    setFilteredExercises(filtered)
  }

  const handleExercisePress = (exerciseId: string) => {
    navigation.navigate("ExerciseDetail", { exerciseId })
  }

  const clearSearch = () => {
    setSearchQuery("")
  }

  const renderExerciseItem = ({ item }: { item: Exercise }) => (
    <ExerciseCard exercise={item} onPress={() => handleExercisePress(item.id)} />
  )

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar barStyle="dark-content" backgroundColor="white" />
      <View style={styles.container}>
        <View style={styles.header}>
          <Text style={styles.title}>Exercise Library</Text>
        </View>

        <View style={styles.searchContainer}>
          <View style={styles.searchInputContainer}>
            <Search size={20} color={colors.gray[500]} style={{marginRight: spacing.sm}} />
            <TextInput
              style={styles.searchInput}
              placeholder="Search exercises..."
              value={searchQuery}
              onChangeText={setSearchQuery}
              placeholderTextColor={colors.gray[400]}
              id="exercise-search"
              accessibilityLabel="Search exercises"
            />
            {searchQuery.length > 0 && (
              <TouchableOpacity onPress={clearSearch} style={styles.clearButton}>
                <X size={18} color={colors.gray[500]} />
              </TouchableOpacity>
            )}
          </View>
        </View>

        <View style={styles.filtersContainer}>
          <ScrollableFilter
            items={CATEGORIES}
            selectedItem={selectedCategory}
            onSelectItem={setSelectedCategory}
            labelTransform="capitalize"
          />
          <ScrollableFilter
            items={DIFFICULTIES}
            selectedItem={selectedDifficulty}
            onSelectItem={setSelectedDifficulty}
            labelTransform="capitalize"
          />
        </View>

        {loading ? (
          <View style={styles.loadingContainer}>
            <SkeletonView height={200} style={styles.skeleton} />
            <SkeletonView height={200} style={styles.skeleton} />
            <SkeletonView height={200} style={styles.skeleton} />
          </View>
        ) : (
          <FlatList
            data={filteredExercises}
            renderItem={renderExerciseItem}
            keyExtractor={(item) => item.id}
            contentContainerStyle={styles.listContent}
            showsVerticalScrollIndicator={false}
            ListEmptyComponent={
              <View style={styles.emptyContainer}>
                <Text style={styles.emptyText}>No exercises found</Text>
              </View>
            }
          />
        )}
      </View>
    </SafeAreaView>
  )
}

interface ScrollableFilterProps {
  items: string[]
  selectedItem: string
  onSelectItem: (item: string) => void
  labelTransform?: "capitalize" | "uppercase" | "lowercase" | "none"
}

const ScrollableFilter: React.FC<ScrollableFilterProps> = ({
  items,
  selectedItem,
  onSelectItem,
  labelTransform = "none",
}) => {
  return (
    <View style={styles.filterScrollContainer}>
      <FlatList
        data={items}
        horizontal
        showsHorizontalScrollIndicator={false}
        keyExtractor={(item) => item}
        renderItem={({ item }) => (
          <TouchableOpacity
            style={[styles.filterItem, selectedItem === item && styles.filterItemSelected]}
            onPress={() => onSelectItem(item)}
          >
            <Text
              style={[
                styles.filterItemText,
                selectedItem === item && styles.filterItemTextSelected,
                labelTransform === "capitalize" && styles.capitalize,
                labelTransform === "uppercase" && styles.uppercase,
                labelTransform === "lowercase" && styles.lowercase,
              ]}
            >
              {item}
            </Text>
          </TouchableOpacity>
        )}
        contentContainerStyle={styles.filterScrollContent}
      />
    </View>
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
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    backgroundColor: "white",
    borderBottomWidth: 1,
    borderBottomColor: colors.gray[200],
  },
  title: {
    fontSize: fontSizes.xxl,
    fontWeight: "700",
    color: colors.gray[900],
  },
  searchContainer: {
    padding: spacing.md,
    backgroundColor: "white",
  },
  searchInputContainer: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: colors.gray[100],
    borderRadius: borderRadius.md,
    paddingHorizontal: spacing.md,
  },
  searchIcon: {
    marginRight: spacing.sm,
  },
  searchInput: {
    flex: 1,
    height: 44,
    color: colors.gray[900],
    fontSize: fontSizes.md,
  },
  clearButton: {
    padding: spacing.xs,
  },
  filtersContainer: {
    backgroundColor: "white",
    paddingBottom: spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: colors.gray[200],
  },
  filterScrollContainer: {
    marginVertical: spacing.xs,
  },
  filterScrollContent: {
    paddingHorizontal: spacing.md,
  },
  filterItem: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
    borderRadius: borderRadius.full,
    marginRight: spacing.sm,
    backgroundColor: colors.gray[100],
  },
  filterItemSelected: {
    backgroundColor: colors.primary[500],
  },
  filterItemText: {
    fontSize: fontSizes.sm,
    color: colors.gray[700],
  },
  filterItemTextSelected: {
    color: "white",
    fontWeight: "500",
  },
  capitalize: {
    textTransform: "capitalize",
  },
  uppercase: {
    textTransform: "uppercase",
  },
  lowercase: {
    textTransform: "lowercase",
  },
  listContent: {
    padding: spacing.md,
    gap: spacing.md,
  },
  loadingContainer: {
    padding: spacing.md,
    gap: spacing.md,
  },
  skeleton: {
    borderRadius: borderRadius.md,
    marginBottom: spacing.md,
  },
  emptyContainer: {
    alignItems: "center",
    justifyContent: "center",
    padding: spacing.xl,
  },
  emptyText: {
    fontSize: fontSizes.md,
    color: colors.gray[600],
  },
})

export default LibraryScreen
