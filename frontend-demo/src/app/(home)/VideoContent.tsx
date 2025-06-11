import { useMemo } from "react"
import { VIDEO_GPT_URL } from "@/services/ai/urls"
import { type AiChatProps } from "@mitodl/smoot-design/ai"
import { useSearchParamSettings } from "./util"
import { useQuery } from "@tanstack/react-query"
import OpenEdxLoginAlert from "./OpenedxLoginAlert"
import Alert from "@mui/lab/Alert"
import OpenedxUnitSelectionForm from "./OpenedxUnitSelectionForm"
import Typography from "@mui/material/Typography"
import BaseChatContent from "./BaseChatContent"

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

  const isReady = !!transcriptBlockId

  return (
    <BaseChatContent
      title="VideoGPT"
      apiUrl={VIDEO_GPT_URL}
      chatIdPrefix="video-gpt"
      conversationStarters={CONVERSATION_STARTERS}
      initialMessages={INITIAL_MESSAGES}
      promptQueryKey="video_gpt"
      promptSettingKey="video_prompt"
      modelSettingKey="video_model"
      isReady={isReady}
      extraBody={{
        model: settings.video_model,
        transcript_asset_id: transcriptBlockId,
        edx_module_id: settings.video_unit,
        instructions: settings.video_prompt,
      }}
      settings={settings}
      setSettings={setSettings}
      sidebarContent={
        <>
          <OpenEdxLoginAlert />
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
          {transcriptError && <Alert severity="error">{transcriptError}</Alert>}
        </>
      }
    />
  )
}

export default VideoContent
