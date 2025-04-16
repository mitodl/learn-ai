import type { AiChatProps } from "@mitodl/smoot-design/ai"
import { useSearchParams } from "next/navigation"
import { useMemo } from "react"

/**
 * Convenience for AiChat component to compute requestOpts settings
 * for a particular apiUrl / set of extra body params.
 */
const getRequestOpts = <B extends Record<string, unknown>>({
  extraBody,
  apiUrl,
}: {
  apiUrl: string
  extraBody: B
}): AiChatProps["requestOpts"] => {
  return {
    apiUrl,
    fetchOpts: { credentials: "include" },
    transformBody: (messages) => {
      return {
        message: messages[messages.length - 1].content,
        ...extraBody,
        model: extraBody.model ? extraBody.model : undefined,
      }
    },
  }
}

/**
 * Patch the current URL with new search params.
 */
const patchSearchParams = <S extends Record<string, string>>(
  patch: Partial<Record<keyof S, string | null>>,
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
const useSearchParamSettings = <S extends Record<string, string>>(
  defaultSettings: S,
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
    } as S
  }, [searchParams, defaultSettings])

  return [
    settings,
    /**
     * This is a constnat function, but pass it here to:
     * 1. mimic useState API
     * 2. provide a type-definition that matches `settings`
     */
    patchSearchParams<S>,
  ] as const
}

export { getRequestOpts, useSearchParamSettings }
