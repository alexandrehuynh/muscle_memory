let userConfig = undefined
try {
  // try to import ESM first
  userConfig = await import('./v0-user-next.config.mjs')
} catch (e) {
  try {
    // fallback to CJS import
    userConfig = await import("./v0-user-next.config");
  } catch (innerError) {
    // ignore error
  }
}

/** @type {import('next').NextConfig} */
const nextConfig = {
  transpilePackages: [
    'expo-av',
    'expo-camera',
    'expo-media-library',
    'expo-image-picker',
    'react-native',
    'react-native-web',
    'expo',
    'expo-modules-core',
    '@react-native-community/slider',
    '@react-navigation/native',
    '@react-navigation/bottom-tabs',
    '@react-navigation/native-stack',
    'lucide-react-native',
    'react-native-safe-area-context',
    'react-native-screens',
    'react-native-svg',
    'expo-asset',
    '@react-native/assets-registry',
  ],
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  experimental: {
    webpackBuildWorker: true,
    parallelServerBuildTraces: true,
    parallelServerCompiles: true,
  },
  webpack: (config, { isServer }) => {
    config.resolve.alias = {
      ...(config.resolve.alias || {}),
      // Alias react-native to react-native-web
      'react-native$': 'react-native-web',
    }
    config.resolve.extensions = [
      '.web.js',
      '.web.jsx',
      '.web.ts',
      '.web.tsx',
      ...config.resolve.extensions,
    ]

    // Prevent "Worker is not defined" error with expo-camera
    if (isServer) {
      config.externals = [
        ...config.externals,
        'expo-camera',
        'react-native-web',
      ]
    }

    // Add fallbacks for common Node.js modules
    config.resolve.fallback = {
      ...config.resolve.fallback,
      'process': false,
      'stream': false,
      'zlib': false,
      'util': false,
      'buffer': false,
      'asset': false,
    }

    return config
  },
}

if (userConfig) {
  // ESM imports will have a "default" property
  const config = userConfig.default || userConfig

  for (const key in config) {
    if (
      typeof nextConfig[key] === 'object' &&
      !Array.isArray(nextConfig[key])
    ) {
      nextConfig[key] = {
        ...nextConfig[key],
        ...config[key],
      }
    } else {
      nextConfig[key] = config[key]
    }
  }
}

export default nextConfig
