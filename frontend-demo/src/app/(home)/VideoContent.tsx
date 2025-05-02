import { VIDEO_GPT_URL } from "@/services/ai/urls"
import AiChatDisplay from "./StyledAiChatDisplay"
import { AiChatProvider, type AiChatProps } from "@mitodl/smoot-design/ai"
import Typography from "@mui/material/Typography"
import Grid from "@mui/material/Grid2"
import SelectModel from "./SelectModel"
import { getRequestOpts, useSearchParamSettings } from "./util"

import { useQuery } from "@tanstack/react-query"
import OpenEdxLoginAlert from "./OpenedxLoginAlert"
import { contenfilesQueries } from "@/services/learn"

import Alert from "@mui/lab/Alert"
import { useMemo } from "react"

import OpenedxUnitSelectionForm from "./OpenedxUnitSelectionForm"
import { ContentFile } from "@mitodl/open-api-axios/v1"
import { CircularProgress } from "@mui/material"
import MetadataDisplay from "./MetadataDisplay"

const CONVERSATION_STARTERS: AiChatProps["conversationStarters"] = []
const INITIAL_MESSAGES: AiChatProps["initialMessages"] = [
  { role: "assistant", content: "What do you want to know about this video?" },
]

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
  /**
   * NOTE: DOMParser is not available on the NextJS server.
   * This doesn't really matter for us since out build is static.
   * However, it does cause a warning in dev mode (which does use the server)
   * if DOMParser is accessed on first-render. Here, it won't be.
   */
  const domParser = new DOMParser()
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
    .replace("block-v1:", "asset-v1:")
    .replace("video+block", "asset+block")
    .split("@")
    .slice(0, 2)
    .join("@")

  return `${transcriptIdPrefix}@${englishTranscriptId}`
}

// https://learn.mit.edu/?resource=2812
const DEFAULT_VERTICAL =
  "block-v1:MITxT+3.012Sx+3T2024+type@vertical+block@2e6efaa3135d49d29b6464d24b398fda"
const DEFAULT_VIDEO =
  "block-v1:MITxT+3.012Sx+3T2024+type@video+block@ab8c1a02e9804e75aff98835dd03c28d"

const VideoCntent = () => {
  const [settings, setSettings] = useSearchParamSettings({
    video_model: "",
    video_vertical: DEFAULT_VERTICAL,
    video_unit: DEFAULT_VIDEO,
  })

  const videoContenfiles = useQuery({
    ...contenfilesQueries.listing({ edxModuleIds: [settings.video_unit] }),
    enabled: !!settings.video_unit,
  })
  const { id: transcriptBlockId, error: transcriptError } = useMemo(() => {
    const results = videoContenfiles.data?.results ?? []
    if (videoContenfiles.isLoading) return { id: null, error: null }
    if (videoContenfiles.isError) {
      return { id: null, error: "Error loading video content." }
    }
    if (results.length !== 1) {
      return { id: null, error: "Expected exactly 1 contentfile match." }
    }
    try {
      const id = getTranscriptBlockId(results[0])
      return { id, error: null }
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : "Unknown Error"
      return { id: null, error: errMsg }
    }
  }, [videoContenfiles])

  const requestOpts = getRequestOpts({
    apiUrl: VIDEO_GPT_URL,
    extraBody: {
      model: settings.video_model,
      transcript_asset_id: transcriptBlockId,
      edx_module_id: settings.video_unit,
    },
  })
  const isReady = !!transcriptBlockId
  return (
    <>
      <Typography variant="h3">VideoGPT</Typography>
      <AiChatProvider
        chatId="video-gpt"
        initialMessages={INITIAL_MESSAGES}
        requestOpts={requestOpts}
      >
        <Grid container spacing={2} sx={{ padding: 2 }}>
          <Grid
            size={{ xs: 12, md: 8 }}
            sx={{ position: "relative", minHeight: "600px" }}
            inert={!isReady}
          >
            <AiChatDisplay
              entryScreenEnabled={false}
              conversationStarters={CONVERSATION_STARTERS}
            />
            {!isReady && (
              <CircularProgress
                color="primary"
                sx={{
                  position: "absolute",
                  zIndex: 1000,
                  top: "50%",
                  left: "50%",
                  transform: "translate(-50%, -50%)",
                }}
              />
            )}
          </Grid>
          <Grid
            size={{ xs: 12, md: 4 }}
            sx={{
              display: "flex",
              flexDirection: "column",
              gap: "16px",
            }}
          >
            <OpenEdxLoginAlert />
            <SelectModel
              value={settings.video_model}
              onChange={(e) => setSettings({ video_model: e.target.value })}
            />
            <OpenedxUnitSelectionForm
              selectedVertical={settings.video_vertical}
              selectedUnit={settings.video_unit}
              defaultUnit={settings.video_unit}
              defaultVertical={settings.video_vertical}
              onSubmit={(values) => {
                setSettings({
                  video_unit: values.unit,
                  video_vertical: values.vertical,
                })
              }}
              onReset={() => {
                setSettings({
                  video_unit: null,
                  video_vertical: null,
                })
              }}
              unitFilterType="video"
              unitLabel="Video"
            />
            {videoContenfiles.isLoading && <Typography>Loading...</Typography>}
            {transcriptError && (
              <Alert severity="error">{transcriptError}</Alert>
            )}
          </Grid>
          <Grid size={{ xs: 12 }}>
            <MetadataDisplay />
          </Grid>
        </Grid>
      </AiChatProvider>
    </>
  )
}

export default VideoCntent
