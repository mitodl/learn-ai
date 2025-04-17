import { ContentFile } from "@mitodl/open-api-axios/v1"
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

const domParser = new DOMParser()
/**
 * Given a video contentfile object, attempt to extract the transcript block ID.
 *
 * This relies on the contentfile's properties:
 *  - `edx_module_id`: the block ID of the video
 *  - `content`: the XML content of the video block
 *
 * The edx_module_id for the video looks like:
 *  block-v1:COURSE_ID+type@video+block@@VIDEO_FILENAME
 *
 * The desired transcript block ID looks like:
 *  asset-v1:COURSE_ID+type@asset+block@@TRANSCRIPT_FILENAME
 *
 * We extract the TRANSCRIPT_FILENAME from the video block XML, then do some
 * surgery on the video's edx_module_id.
 *
 * This is all a bit hacky, but neither Learn nor OpenEdx expose a
 * video <-->transcript relationship between the global( block-v1..., asset-v1...)
 * ids as far as I can tell.
 *
 * Using OpenEdx staff APIs, we could get the transcript filename without XML
 * parsing via v2/block_metadata/<block_id>.
 */
const getTranscriptBlockId = (contentfile: ContentFile) => {
  const videoBlockId = contentfile.edx_module_id
  if (!videoBlockId) {
    throw new Error("No video block ID found.")
  }
  if (!contentfile.content) {
    throw new Error("Contentfile has no content.")
  }

  const xml = domParser.parseFromString(contentfile.content, "text/xml")
  const videoTag = xml.querySelector("video")
  if (!videoTag) {
    throw new Error("No video tag found in content.")
  }
  const transcriptsAttr = videoTag.getAttribute("transcripts")
  if (!transcriptsAttr) {
    throw new Error(
      "No transcripts found in video tag. Video may not have transcript or may use YouTube transcripts.",
    )
  }
  const transcripts = JSON.parse(transcriptsAttr)
  const englishTranscriptId = transcripts["en"]
  if (!englishTranscriptId) {
    throw new Error("No English transcript found.")
  }

  /**
   * videoBlockId       = block-v1:MITxT+3.012Sx+3T2024+type@video+block@....
   * transcriptIdPrefix = asset-v1:MITxT+3.012Sx+3T2024+type@asset+block
   */
  const transcriptIdPrefix = videoBlockId
    .replace("block-v1:", "asset-v1")
    .replace("video+block", "asset+block")
    .split("@")
    .slice(0, 2)
    .join("@")

  return `${transcriptIdPrefix}@${englishTranscriptId}`
}

export { getRequestOpts, useSearchParamSettings, getTranscriptBlockId }
