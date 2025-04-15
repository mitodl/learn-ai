import type { AiChatProps } from "@mitodl/smoot-design/ai"

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
      }
    },
  }
}

export { getRequestOpts }
