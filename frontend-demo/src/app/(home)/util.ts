import type { AiChatProps } from "@mitodl/smoot-design/ai"
import { useSearchParams } from "next/navigation"
import { useMemo, useRef, useState } from "react"

/**
 * Convenience for AiChat component to compute requestOpts settings
 * for a particular apiUrl / set of extra body params.
 */
const getRequestOpts = <Body extends Record<string, unknown>>({
  extraBody,
  apiUrl,
}: {
  apiUrl: string
  extraBody: Body
}): AiChatProps["requestOpts"] => {
  return {
    apiUrl,
    fetchOpts: { credentials: "include" },
    transformBody: (messages) => {
      return {
        message: messages[messages.length - 1].content,
        ...extraBody,
        model: extraBody.model ? extraBody.model : undefined,
        search_url: extraBody.search_url ? extraBody.search_url : undefined,
      }
    },
  }
}

/**
 * This hook is intended to:
 * 1. Help create requestOpts for use with AiChat component
 * 2. Facilitate clearing the chat history (starting a new thread)
 *
 * When the requestNewThread function is called, it will:
 * 1. configure the next message sent to chatbot to clear history
 * 2. increment the threadCount state variable
 *
 * Incrementing the threadCount variable is to facilitate resetting the state
 * of the AiChat component.
 */
const useRequestOpts = <Body extends Record<string, unknown>>({
  extraBody,
  apiUrl,
}: {
  apiUrl: string
  extraBody: Body
}): {
  requestOpts: AiChatProps["requestOpts"]
  requestNewThread: () => void
  threadCount: number
} => {
  const clearHistoryRef = useRef(false)
  const [threadCount, setThreadCount] = useState(0)
  const requestOpts = useMemo(() => {
    const opts: AiChatProps["requestOpts"] = {
      apiUrl,
      fetchOpts: { credentials: "include" },
      transformBody: (messages) => {
        const clearHistory = clearHistoryRef.current
        clearHistoryRef.current = false
        return {
          message: messages[messages.length - 1].content,
          ...extraBody,
          clear_history: clearHistory ? true : undefined,
          model: extraBody.model ? extraBody.model : undefined,
          search_url: extraBody.search_url ? extraBody.search_url : undefined,
        }
      },
    }
    return opts
  }, [apiUrl, extraBody])
  const requestNewThread = () => {
    clearHistoryRef.current = true
    setThreadCount((count) => count + 1)
  }
  return { requestOpts, requestNewThread, threadCount }
}

/**
 * Patch the current URL with new search params.
 */
const patchSearchParams = <Settings extends Record<string, string>>(
  patch: Partial<Record<keyof Settings, string | null>>,
) => {
  const url = new URL(window.location.href)
  Object.entries(patch).forEach(([key, value]) => {
    if (value !== null && value !== undefined) {
      url.searchParams.set(key, value)
    } else {
      url.searchParams.delete(key)
    }
  })
  window.history.pushState({}, "", url.toString())
}
const useSearchParamSettings = <Settings extends Record<string, string>>(
  defaultSettings: Settings,
) => {
  const searchParams = useSearchParams()
  const settings = useMemo(() => {
    return {
      ...defaultSettings,
      ...Object.fromEntries(
        searchParams
          .entries()
          .filter(([key]) => Object.keys(defaultSettings).includes(key)),
      ),
    } as Settings
  }, [searchParams, defaultSettings])

  return [
    settings,
    /**
     * This is a constnat function, but pass it here to:
     * 1. mimic useState API
     * 2. provide a type-definition that matches `settings`
     */
    patchSearchParams<Settings>,
  ] as const
}

export { useRequestOpts, getRequestOpts, useSearchParamSettings }
