import { ASSESSMENT_GPT_URL } from "@/services/ai/urls"
import { AiChat } from "@mitodl/smoot-design/ai"
import type { AiChatProps } from "@mitodl/smoot-design/ai"
import Typography from "@mui/material/Typography"
import Grid from "@mui/material/Grid2"
import SelectModel from "./SelectModel"
import { getRequestOpts, useSearchParamSettings } from "./util"
import VerticalAndUnitSelector, {
  isVerticalBlockId,
} from "./VerticalAndUnitSelector"
import FormControlLabel from "@mui/material/FormControlLabel"
import Checkbox from "@mui/material/Checkbox"
import { openEdxQueries } from "@/services/openedx"
import { useQuery } from "@tanstack/react-query"
import OpenEdxLoginAlert from "./OpenedxLoginAlert"

const CONVERSATION_STARTERS: AiChatProps["conversationStarters"] = []
const INITIAL_MESSAGES: AiChatProps["initialMessages"] = [
  { role: "assistant", content: "Hi, do you need any help?" },
]

// https://learn.mit.edu/?resource=2812
const DEFAULT_RESOURCE =
  "block-v1:MITxT+3.012Sx+3T2024+type@vertical+block@fbcef83f1e154419810f04e7db686de5"

const AssessmentContent = () => {
  const [settings, setSettings] = useSearchParamSettings({
    tutor_model: "",
    tutor_vertical: DEFAULT_RESOURCE,
    tutor_sibling_context: "true",
    tutor_unit: "",
  })
  const includeBlockSiblings = settings.tutor_sibling_context !== "false"

  const userMe = useQuery(openEdxQueries.userMe())
  const username = userMe.data?.username ?? ""

  const vertical = useQuery({
    ...openEdxQueries.coursesV2Blocks({
      blockUsageKey: settings.tutor_vertical,
      username,
    }),
    // Don't use error to infer enabled; error might not be set on first render
    enabled: !!username && isVerticalBlockId(settings.tutor_vertical),
  })

  const siblings = includeBlockSiblings
    ? Object.keys(vertical.data?.blocks ?? {}).filter(
        (key) => key !== vertical.data?.root,
      )
    : []

  const requestOpts = getRequestOpts({
    apiUrl: ASSESSMENT_GPT_URL,
    extraBody: {
      model: settings.tutor_model,
      block_siblings: siblings,
      edx_module_id: settings.tutor_unit,
    },
  })

  return (
    <>
      <Typography variant="h3">AssessmentGPT</Typography>
      <Grid container spacing={2} sx={{ padding: 2 }}>
        <Grid
          size={{ xs: 12, md: 8 }}
          sx={{ position: "relative", minHeight: "600px" }}
        >
          <AiChat
            chatId="syllabus-gpt"
            entryScreenEnabled={false}
            initialMessages={INITIAL_MESSAGES}
            conversationStarters={CONVERSATION_STARTERS}
            requestOpts={requestOpts}
          />
        </Grid>
        <Grid size={{ xs: 12, md: 4 }}>
          <OpenEdxLoginAlert />
          <VerticalAndUnitSelector
            verticalSettingsName="tutor_vertical"
            unitSettingsName="tutor_unit"
            settings={settings}
            setSettings={setSettings}
            unitFieldLabel="Problem"
            unitFilterType="problem"
            vertical={vertical}
          />
          <FormControlLabel
            control={
              <Checkbox
                checked={includeBlockSiblings}
                onChange={(e) => {
                  setSettings({
                    tutor_sibling_context: e.target.checked ? "true" : "false",
                  })
                }}
              />
            }
            label="Include Sibling Context"
          />
          <SelectModel
            value={settings.tutor_model}
            onChange={(e) => setSettings({ tutor_model: e.target.value })}
          />
        </Grid>
      </Grid>
    </>
  )
}

export default AssessmentContent
