import { Configuration } from "@api/v0"
import axios from "axios"

const instance = axios.create()

const getConfig = () => {
  return new Configuration({
    basePath: process.env.NEXT_PUBLIC_MIT_LEARN_API_BASE_URL,
  })
}

/**
 * Settings for generated learn-ai API Client.
 * Spread the result into any client, e.g., LlmModelsApi
 */
const getAPISettings = () => [getConfig(), undefined, instance] as const

export { getAPISettings }
