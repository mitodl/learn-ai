import { Configuration } from "@api/v0"
import axios from "axios"
import invariant from "tiny-invariant"

const AI_API_BASE_URL = process.env.NEXT_PUBLIC_MITOL_API_BASE_URL
invariant(AI_API_BASE_URL, "NEXT_PUBLIC_MITOL_API_BASE_URL is required")

/**
 * Our axios instance with default baseURL, headers, etc.
 */
const axiosInstance = axios.create({
  xsrfCookieName: process.env.NEXT_PUBLIC_AI_CSRF_COOKIE_NAME,
  xsrfHeaderName: "X-CSRFToken",
  withXSRFToken: true,
  withCredentials: true,
})

const getConfig = () => {
  return new Configuration({ basePath: AI_API_BASE_URL })
}

/**
 * Settings for generated learn-ai API Client.
 * Spread the result into any client, e.g., LlmModelsApi
 */
const getAPISettings = () => [getConfig(), undefined, axiosInstance] as const

export { getAPISettings, axiosInstance, AI_API_BASE_URL }
