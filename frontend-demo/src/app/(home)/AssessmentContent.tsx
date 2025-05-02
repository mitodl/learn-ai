import { ASSESSMENT_GPT_URL } from "@/services/ai/urls"
import AiChatDisplay from "./StyledAiChatDisplay"
import { AiChatProvider, type AiChatProps } from "@mitodl/smoot-design/ai"
import Typography from "@mui/material/Typography"
import Grid from "@mui/material/Grid2"
import SelectModel from "./SelectModel"
import { useRequestOpts, useSearchParamSettings } from "./util"
import { useV2Block } from "@/services/openedx"
import OpenEdxLoginAlert from "./OpenedxLoginAlert"
import OpenedxUnitSelectionForm from "./OpenedxUnitSelectionForm"
import CircularProgress from "@mui/material/CircularProgress"
import MetadataDisplay from "./MetadataDisplay"
import { MathJaxContext } from "better-react-mathjax"
import { Button } from "@mitodl/smoot-design"

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

  const { requestOpts, requestNewThread, threadCount } = useRequestOpts({
    apiUrl: ASSESSMENT_GPT_URL,
    extraBody: {
      model: settings.tutor_model,
      block_siblings: siblings,
      edx_module_id: settings.tutor_unit,
    },
  })
  const isReady = vertical.isSuccess

  const chatId = `assessment-gpt-${threadCount}`
  return (
    <>
      <Typography variant="h3">AssessmentGPT</Typography>
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
            <MathJaxContext>
              <AiChatDisplay
                entryScreenEnabled={false}
                conversationStarters={CONVERSATION_STARTERS}
                useMathJax={true}
              />
            </MathJaxContext>
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
              value={settings.tutor_model}
              onChange={(e) => {
                setSettings({ tutor_model: e.target.value })
                requestNewThread()
              }}
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

export default AssessmentContent
