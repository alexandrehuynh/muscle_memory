"use client"

// Mock MediaLibrary for web to avoid server-side rendering issues
const WebMediaLibrary = {
  requestPermissionsAsync: async () => {
    console.log("MediaLibrary.requestPermissionsAsync - Mock implementation for web")
    return { status: "granted" }
  },
  saveToLibraryAsync: async (uri: string) => {
    console.log("MediaLibrary.saveToLibraryAsync - Mock implementation for web", { uri })
    return true
  },
}

export default WebMediaLibrary 