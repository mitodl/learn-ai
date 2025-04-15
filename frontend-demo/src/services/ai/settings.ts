import { Configuration } from "@api/v0"
import axios from "axios"

/**
 * Our axios instance with default baseURL, headers, etc.
 */
const instance = axios.create({
  xsrfCookieName: process.env.NEXT_PUBLIC_AI_CSRF_COOKIE_NAME,
  xsrfHeaderName: "X-CSRFToken",
  withXSRFToken: true,
  withCredentials: true,
})

const getConfig = () => {
  return new Configuration({
    basePath: process.env.NEXT_PUBLIC_MITOL_API_BASE_URL,
  })
}

/**
 * Settings for generated learn-ai API Client.
 * Spread the result into any client, e.g., LlmModelsApi
 */
const getAPISettings = () => [getConfig(), undefined, instance] as const

export { getAPISettings }
