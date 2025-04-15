import type { NextConfig } from "next"
import { PHASE_DEVELOPMENT_SERVER } from "next/constants"

const baseConfig: NextConfig = {
  // For simpler sandbox hosting, just use static export build
  output: "export",
}

const config = (phase: string) => {
  if (phase === PHASE_DEVELOPMENT_SERVER) {
    const devConfig: NextConfig = {
      ...baseConfig,
      rewrites: async () => {
        return [
          {
            source: "/openedx_proxy/:path*",
            destination: "http://localhost:8004/:path*",
          },
        ]
      },
    }
    return devConfig
  }
  return baseConfig
}

export default config
