"use client"

import { useState } from "react"
import {
  View,
  Text,
  StyleSheet,
  Switch,
  ScrollView,
  TouchableOpacity,
  SafeAreaView,
  StatusBar,
  Alert,
} from "react-native"
import { Bell, Moon, Smartphone, Lock, HelpCircle, LogOut, ChevronRight } from "lucide-react-native"

import { colors, spacing, fontSizes, fontWeights } from "../theme"
import { Card } from "../components/Card"

const SettingsScreen = () => {
  const [notifications, setNotifications] = useState(true)
  const [darkMode, setDarkMode] = useState(false)
  const [saveVideos, setSaveVideos] = useState(true)
  const [highQuality, setHighQuality] = useState(false)

  const handleLogout = () => {
    Alert.alert(
      "Logout",
      "Are you sure you want to logout?",
      [
        {
          text: "Cancel",
          style: "cancel",
        },
        {
          text: "Logout",
          style: "destructive",
          onPress: () => {
            // Handle logout logic here
            Alert.alert("Logged out successfully")
          },
        },
      ],
      { cancelable: true },
    )
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar barStyle="dark-content" backgroundColor="white" />
      <ScrollView style={styles.container}>
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Preferences</Text>
          <Card style={styles.card}>
            <View style={styles.settingItem}>
              <View style={styles.settingInfo}>
                <View style={styles.iconContainer}>
                  <Bell size={20} color={colors.primary[500]} />
                </View>
                <Text style={styles.settingLabel}>Notifications</Text>
              </View>
              <Switch
                value={notifications}
                onValueChange={setNotifications}
                trackColor={{ false: colors.gray[300], true: colors.primary[500] }}
                thumbColor="white"
              />
            </View>

            <View style={styles.divider} />

            <View style={styles.settingItem}>
              <View style={styles.settingInfo}>
                <View style={styles.iconContainer}>
                  <Moon size={20} color={colors.primary[500]} />
                </View>
                <Text style={styles.settingLabel}>Dark Mode</Text>
              </View>
              <Switch
                value={darkMode}
                onValueChange={setDarkMode}
                trackColor={{ false: colors.gray[300], true: colors.primary[500] }}
                thumbColor="white"
              />
            </View>
          </Card>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Video Settings</Text>
          <Card style={styles.card}>
            <View style={styles.settingItem}>
              <View style={styles.settingInfo}>
                <View style={styles.iconContainer}>
                  <Smartphone size={20} color={colors.primary[500]} />
                </View>
                <Text style={styles.settingLabel}>Save Videos to Device</Text>
              </View>
              <Switch
                value={saveVideos}
                onValueChange={setSaveVideos}
                trackColor={{ false: colors.gray[300], true: colors.primary[500] }}
                thumbColor="white"
              />
            </View>

            <View style={styles.divider} />

            <View style={styles.settingItem}>
              <View style={styles.settingInfo}>
                <View style={styles.iconContainer}>
                  <Smartphone size={20} color={colors.primary[500]} />
                </View>
                <Text style={styles.settingLabel}>High Quality Recording</Text>
              </View>
              <Switch
                value={highQuality}
                onValueChange={setHighQuality}
                trackColor={{ false: colors.gray[300], true: colors.primary[500] }}
                thumbColor="white"
              />
            </View>
          </Card>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Account</Text>
          <Card style={styles.card}>
            <TouchableOpacity style={styles.linkItem}>
              <View style={styles.settingInfo}>
                <View style={styles.iconContainer}>
                  <Lock size={20} color={colors.primary[500]} />
                </View>
                <Text style={styles.settingLabel}>Privacy Settings</Text>
              </View>
              <ChevronRight size={20} color={colors.gray[400]} />
            </TouchableOpacity>

            <View style={styles.divider} />

            <TouchableOpacity style={styles.linkItem}>
              <View style={styles.settingInfo}>
                <View style={styles.iconContainer}>
                  <HelpCircle size={20} color={colors.primary[500]} />
                </View>
                <Text style={styles.settingLabel}>Help & Support</Text>
              </View>
              <ChevronRight size={20} color={colors.gray[400]} />
            </TouchableOpacity>

            <View style={styles.divider} />

            <TouchableOpacity style={styles.linkItem} onPress={handleLogout}>
              <View style={styles.settingInfo}>
                <View style={styles.iconContainer}>
                  <LogOut size={20} color={colors.error} />
                </View>
                <Text style={[styles.settingLabel, { color: colors.error }]}>Logout</Text>
              </View>
              <ChevronRight size={20} color={colors.gray[400]} />
            </TouchableOpacity>
          </Card>
        </View>

        <View style={styles.footer}>
          <Text style={styles.version}>Muscle Memory v1.0.0</Text>
          <TouchableOpacity>
            <Text style={styles.termsLink}>Terms of Service</Text>
          </TouchableOpacity>
          <TouchableOpacity>
            <Text style={styles.termsLink}>Privacy Policy</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
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
  section: {
    padding: spacing.md,
  },
  sectionTitle: {
    fontSize: fontSizes.md,
    fontWeight: fontWeights.semibold,
    color: colors.gray[700],
    marginBottom: spacing.sm,
    marginLeft: spacing.xs,
  },
  card: {
    padding: 0,
  },
  settingItem: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    padding: spacing.md,
  },
  linkItem: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    padding: spacing.md,
  },
  settingInfo: {
    flexDirection: "row",
    alignItems: "center",
  },
  iconContainer: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: colors.primary[50],
    justifyContent: "center",
    alignItems: "center",
    marginRight: spacing.md,
  },
  settingLabel: {
    fontSize: fontSizes.md,
    color: colors.gray[900],
  },
  divider: {
    height: 1,
    backgroundColor: colors.gray[200],
    marginHorizontal: spacing.md,
  },
  footer: {
    padding: spacing.xl,
    alignItems: "center",
  },
  version: {
    fontSize: fontSizes.sm,
    color: colors.gray[500],
    marginBottom: spacing.md,
  },
  termsLink: {
    fontSize: fontSizes.sm,
    color: colors.primary[500],
    marginBottom: spacing.sm,
  },
})

export default SettingsScreen
