import { ASSESSMENT_GPT_URL } from "@/services/ai/urls"
import AiChat from "./StyledAiChat"
import type { AiChatProps } from "@mitodl/smoot-design/ai"
import Typography from "@mui/material/Typography"
import Grid from "@mui/material/Grid2"
import SelectModel from "./SelectModel"
import { getRequestOpts, useSearchParamSettings } from "./util"
import { useV2Block } from "@/services/openedx"
import OpenEdxLoginAlert from "./OpenedxLoginAlert"
import OpenedxUnitSelectionForm from "./OpenedxUnitSelectionForm"
import CircularProgress from "@mui/material/CircularProgress"

const CONVERSATION_STARTERS: AiChatProps["conversationStarters"] = []
const INITIAL_MESSAGES: AiChatProps["initialMessages"] = [
  { role: "assistant", content: "Hi, do you need any help?" },
]

// https://courses-qa.mitxonline.mit.edu/learn/course/course-v1:MITxT+3.012Sx+3T2024/block-v1:MITxT+3.012Sx+3T2024+type@sequential+block@74c793b8e88e41b8820760bcc7ef2bdb/block-v1:MITxT+3.012Sx+3T2024+type@vertical+block@fbcef83f1e154419810f04e7db686de5
const DEFAULT_VERTICAL =
  "block-v1:MITxT+3.012Sx+3T2024+type@vertical+block@fbcef83f1e154419810f04e7db686de5"
const DEFAULT_UNIT =
  "block-v1:MITxT+3.012Sx+3T2024+type@problem+block@318c3d44596649c39fb10c25aa847862"

const AssessmentContent = () => {
  const [settings, setSettings] = useSearchParamSettings({
    tutor_model: "",
    tutor_vertical: DEFAULT_VERTICAL,
    tutor_sibling_context: "true",
    tutor_unit: DEFAULT_UNIT,
  })

  const vertical = useV2Block({
    blockUsageKey: settings.tutor_vertical,
  })

  const siblings = Object.keys(vertical.data?.blocks ?? {}).filter(
    (key) => key !== vertical.data?.root,
  )

  const requestOpts = getRequestOpts({
    apiUrl: ASSESSMENT_GPT_URL,
    extraBody: {
      model: settings.tutor_model,
      block_siblings: siblings,
      edx_module_id: settings.tutor_unit,
    },
  })
  const isReady = vertical.isSuccess

  return (
    <>
      <Typography variant="h3">AssessmentGPT</Typography>
      <Grid container spacing={2} sx={{ padding: 2 }}>
        <Grid
          size={{ xs: 12, md: 8 }}
          sx={{ position: "relative", minHeight: "600px" }}
          inert={!isReady}
        >
          <AiChat
            chatId="syllabus-gpt"
            entryScreenEnabled={false}
            initialMessages={INITIAL_MESSAGES}
            conversationStarters={CONVERSATION_STARTERS}
            requestOpts={requestOpts}
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
        <Grid size={{ xs: 12, md: 4 }}>
          <OpenEdxLoginAlert />
          <SelectModel
            value={settings.tutor_model}
            onChange={(e) => setSettings({ tutor_model: e.target.value })}
          />
          <OpenedxUnitSelectionForm
            selectedVertical={settings.tutor_vertical}
            selectedUnit={settings.tutor_unit}
            defaultUnit={settings.tutor_unit}
            defaultVertical={settings.tutor_vertical}
            onSubmit={(values) => {
              setSettings({
                tutor_unit: values.unit,
                tutor_vertical: values.vertical,
              })
            }}
            onReset={() => {
              setSettings({
                tutor_unit: null,
                tutor_vertical: null,
              })
            }}
            unitFilterType="problem"
            unitLabel="Problem"
          />
        </Grid>
      </Grid>
    </>
  )
}

export default AssessmentContent
