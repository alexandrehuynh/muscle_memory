import React from 'react';
import { NavigationContainer } from "@react-navigation/native"
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs"
import { createNativeStackNavigator } from "@react-navigation/native-stack"
import { Home, Dumbbell, User } from "lucide-react-native"
import { View, Text } from 'react-native';

import type { RootStackParamList, MainTabParamList } from "../types"
import { colors } from "../theme"

// Screens
import HomeScreen from "../screens/HomeScreen"
import LibraryScreen from "../screens/LibraryScreen"
import ProfileScreen from "../screens/ProfileScreen"
import RecordVideoScreen from "../screens/RecordVideoScreen"
import UploadVideoScreen from "../screens/UploadVideoScreen"
import ExerciseDetailScreen from "../screens/ExerciseDetailScreen"
import AnalysisResultScreen from "../screens/AnalysisResultScreen"
import SettingsScreen from "../screens/SettingsScreen"

const Stack = createNativeStackNavigator<RootStackParamList>()
const Tab = createBottomTabNavigator<MainTabParamList>()

const MainTabs = () => {
  return (
    <Tab.Navigator
      screenOptions={{
        tabBarActiveTintColor: colors.primary[500],
        tabBarInactiveTintColor: colors.gray[400],
        tabBarStyle: {
          borderTopWidth: 1,
          borderTopColor: colors.gray[200],
          paddingTop: 5,
          paddingBottom: 5,
          height: 60,
        },
        tabBarLabelStyle: {
          fontSize: 12,
          fontWeight: "500",
        },
        headerShown: false,
      }}
    >
      <Tab.Screen
        name="Home"
        component={HomeScreen}
        options={{
          tabBarIcon: ({ color, size }) => <Home color={color} size={size} />,
        }}
      />
      <Tab.Screen
        name="Library"
        component={LibraryScreen}
        options={{
          tabBarIcon: ({ color, size }) => <Dumbbell color={color} size={size} />,
        }}
      />
      <Tab.Screen
        name="Profile"
        component={ProfileScreen}
        options={{
          tabBarIcon: ({ color, size }) => <User color={color} size={size} />,
        }}
      />
    </Tab.Navigator>
  )
}

export const AppNavigator = () => {
  return (
    <NavigationContainer>
      <Stack.Navigator
        screenOptions={{
          headerStyle: {
            backgroundColor: colors.primary[500],
          },
          headerTintColor: "white",
          headerTitleStyle: {
            fontWeight: "600",
          },
        }}
      >
        <Stack.Screen name="Main" component={MainTabs} options={{ headerShown: false }} />
        <Stack.Screen name="RecordVideo" component={RecordVideoScreen} options={{ title: "Record Exercise" }} />
        <Stack.Screen name="UploadVideo" component={UploadVideoScreen} options={{ title: "Upload Exercise Video" }} />
        <Stack.Screen
          name="ExerciseDetail"
          component={ExerciseDetailScreen}
          options={({ route }) => ({ title: "Exercise Details" })}
        />
        <Stack.Screen name="AnalysisResult" component={AnalysisResultScreen} options={{ title: "Analysis Results" }} />
        <Stack.Screen name="Settings" component={SettingsScreen} options={{ title: "Settings" }} />
      </Stack.Navigator>
    </NavigationContainer>
  )
}
