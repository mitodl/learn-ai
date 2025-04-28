import type { AiChatProps } from "@mitodl/smoot-design/ai"
import { useSearchParams } from "next/navigation"
import { useMemo } from "react"

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

export { getRequestOpts, useSearchParamSettings }
