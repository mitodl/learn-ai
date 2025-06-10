import { VIDEO_GPT_URL } from "@/services/ai/urls"
import AiChatDisplay from "./StyledAiChatDisplay"
import { AiChatProvider, type AiChatProps } from "@mitodl/smoot-design/ai"
import Typography from "@mui/material/Typography"
import Grid from "@mui/material/Grid2"
import SelectModel from "./SelectModel"
import { useRequestOpts, useSearchParamSettings } from "./util"

import { useQuery } from "@tanstack/react-query"
import OpenEdxLoginAlert from "./OpenedxLoginAlert"

import Alert from "@mui/lab/Alert"
import { useEffect, useMemo, useState } from "react"

import OpenedxUnitSelectionForm from "./OpenedxUnitSelectionForm"
import { CircularProgress, FormLabel, TextareaAutosize } from "@mui/material"
import MetadataDisplay from "./MetadataDisplay"
import { Button } from "@mitodl/smoot-design"
import { promptQueries, userQueries } from "@/services/ai"

const CONVERSATION_STARTERS: AiChatProps["conversationStarters"] = []
const INITIAL_MESSAGES: AiChatProps["initialMessages"] = [
  { role: "assistant", content: "What do you want to know about this video?" },
]

// https://learn.mit.edu/?resource=2812
const DEFAULT_VERTICAL =
  "block-v1:MITxT+3.012Sx+3T2024+type@vertical+block@2e6efaa3135d49d29b6464d24b398fda"
const DEFAULT_VIDEO =
  "block-v1:MITxT+3.012Sx+3T2024+type@video+block@ab8c1a02e9804e75aff98835dd03c28d"

const VideoContent = () => {
  const [settings, setSettings] = useSearchParamSettings({
    video_model: "",
    video_vertical: DEFAULT_VERTICAL,
    video_unit: DEFAULT_VIDEO,
    video_prompt: "",
  })

  const me = useQuery(userQueries.me())
  const promptResult = useQuery(promptQueries.get("video_gpt"))
  const [promptText, setPromptText] = useState(settings.video_prompt)
  useEffect(() => {
    const nextValue = promptText || promptResult.data?.prompt_value || ""
    if (nextValue === promptResult.data?.prompt_value) {
      // If the prompt is identical to the default, don't send it in the request.
      setSettings({ video_prompt: "" })
    } else if (settings.video_prompt !== nextValue) {
      setSettings({ video_prompt: nextValue })
    }
  }, [
    promptText,
    setSettings,
    settings.video_prompt,
    promptResult.data?.prompt_value,
  ])

  const getTranscriptBlockId = async (edxModuleId: string) => {
    const response = await fetch(
      `${process.env.NEXT_PUBLIC_MITOL_API_BASE_URL}/api/v0/get_transcript_edx_module_id/?edx_module_id=${encodeURIComponent(edxModuleId)}`,
    )
    return response.json()
  }

  const transcriptIdQueryResult = useQuery({
    queryKey: ["transcriptBlockId", settings.video_unit],
    queryFn: () => getTranscriptBlockId(settings.video_unit),
    enabled: !!settings.video_unit,
  })
  const { id: transcriptBlockId, error: transcriptError } = useMemo(() => {
    if (transcriptIdQueryResult.isLoading) return { id: null, error: null }
    if (transcriptIdQueryResult.data.error) {
      return { id: null, error: transcriptIdQueryResult.data.error }
    }

    return { id: transcriptIdQueryResult.data.transcript_block_id, error: null }
  }, [transcriptIdQueryResult])

  const { requestOpts, chatSuffix, requestNewThread } = useRequestOpts({
    apiUrl: VIDEO_GPT_URL,
    extraBody: {
      model: settings.video_model,
      transcript_asset_id: transcriptBlockId,
      edx_module_id: settings.video_unit,
      instructions: settings.video_prompt,
    },
  })
  const isReady = !!transcriptBlockId
  const chatId = `video-gpt-${chatSuffix}`
  return (
    <>
      <Typography variant="h3">VideoGPT</Typography>
      <AiChatProvider
        chatId={chatId}
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
              onChange={(e) => {
                setSettings({ video_model: e.target.value })
                requestNewThread()
              }}
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
            {transcriptIdQueryResult.isLoading && (
              <Typography>Loading...</Typography>
            )}
            {transcriptError && (
              <Alert severity="error">{transcriptError}</Alert>
            )}
            {me.data?.is_staff ? (
              <>
                <FormLabel htmlFor="video-prompt-ta">System Prompt</FormLabel>
                <TextareaAutosize
                  id="video-prompt-ta"
                  minRows={5}
                  maxRows={15}
                  value={promptText || promptResult.data?.prompt_value || ""}
                  onChange={(e) => setPromptText(e.target.value)}
                />
              </>
            ) : (
              <></>
            )}
            <Button variant="secondary" onClick={requestNewThread}>
              Start new thread
            </Button>
          </Grid>
          <Grid size={{ xs: 12 }}>
            <MetadataDisplay />
          </Grid>
        </Grid>
      </AiChatProvider>
    </>
  )
}

export default VideoContent
