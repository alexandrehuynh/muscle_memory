// import { SafeAreaProvider } from "react-native-safe-area-context"
import { SafeAreaProvider } from "react-native-safe-area-context"
import { AppNavigator } from "./navigation"
import { ClientOnly } from "./components/ClientOnly"
import { View } from "react-native"
import { colors } from "./theme"

export default function App() {
  return (
    <SafeAreaProvider>
      <ClientOnly
        fallback={
          <View style={{ flex: 1, backgroundColor: colors.gray[50] }} />
        }
      >
        <AppNavigator />
      </ClientOnly>
    </SafeAreaProvider>
  )
}
