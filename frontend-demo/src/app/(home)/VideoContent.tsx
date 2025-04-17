import { ASSESSMENT_GPT_URL } from "@/services/ai/urls"
import { AiChat } from "@mitodl/smoot-design/ai"
import type { AiChatProps } from "@mitodl/smoot-design/ai"
import Typography from "@mui/material/Typography"
import Grid from "@mui/material/Grid2"
import SelectModel from "./SelectModel"
import {
  getRequestOpts,
  getTranscriptBlockId,
  useSearchParamSettings,
} from "./util"
import VerticalAndUnitSelector, {
  isVerticalBlockId,
} from "./VerticalAndUnitSelector"
import { openEdxQueries } from "@/services/openedx"
import { useQuery } from "@tanstack/react-query"
import OpenEdxLoginAlert from "./OpenedxLoginAlert"
import { contenfilesQueries } from "@/services/learn"

import Alert from "@mui/lab/Alert"
import { useMemo } from "react"
import ChatContainer from "./ChatContainer"

const CONVERSATION_STARTERS: AiChatProps["conversationStarters"] = []
const INITIAL_MESSAGES: AiChatProps["initialMessages"] = [
  { role: "assistant", content: "What do you want to know about this video?" },
]

// https://learn.mit.edu/?resource=2812
const DEFAULT_RESOURCE =
  "block-v1:MITxT+3.012Sx+3T2024+type@vertical+block@7fe893afc4044b18abef3ee484118f30"

const VideoCntent = () => {
  const [settings, setSettings] = useSearchParamSettings({
    video_model: "",
    video_vertical: DEFAULT_RESOURCE,
    video_unit: "",
  })

  console.log(settings)

  const userMe = useQuery(openEdxQueries.userMe())
  const username = userMe.data?.username ?? ""

  const vertical = useQuery({
    ...openEdxQueries.coursesV2Blocks({
      blockUsageKey: settings.video_vertical,
      username,
    }),
    // Don't use error to infer enabled; error might not be set on first render
    enabled: !!username && isVerticalBlockId(settings.video_vertical),
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
      console.log(results)
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
    apiUrl: ASSESSMENT_GPT_URL,
    extraBody: {
      model: settings.video_model,
      transcript_asset_id: transcriptBlockId,
      edx_module_id: settings.video_unit,
    },
  })
  const isReady = Boolean(transcriptBlockId && settings.video_unit)
  return (
    <>
      <Typography variant="h3">VideoGPT</Typography>
      <Grid container spacing={2} sx={{ padding: 2 }}>
        <Grid size={{ xs: 12, md: 8 }}>
          <ChatContainer enabled={isReady}>
            <AiChat
              chatId="syllabus-gpt"
              entryScreenEnabled={false}
              initialMessages={INITIAL_MESSAGES}
              conversationStarters={CONVERSATION_STARTERS}
              requestOpts={requestOpts}
            />
          </ChatContainer>
        </Grid>
        <Grid size={{ xs: 12, md: 4 }}>
          <OpenEdxLoginAlert />
          <VerticalAndUnitSelector
            verticalSettingsName="video_vertical"
            unitSettingsName="video_unit"
            settings={settings}
            setSettings={setSettings}
            unitFieldLabel="Video"
            unitFilterType="video"
            vertical={vertical}
          />
          {transcriptError && <Alert severity="error">{transcriptError}</Alert>}
          <SelectModel
            value={settings.video_model}
            onChange={(e) => setSettings({ video_model: e.target.value })}
          />
        </Grid>
      </Grid>
    </>
  )
}

export default VideoCntent
