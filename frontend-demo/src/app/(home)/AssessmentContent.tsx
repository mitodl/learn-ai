import { ASSESSMENT_GPT_URL } from "@/services/ai/urls"
import { type AiChatProps } from "@mitodl/smoot-design/ai"
import { useSearchParamSettings } from "./util"
import { useV2Block } from "@/services/openedx"
import OpenEdxLoginAlert from "./OpenedxLoginAlert"
import OpenedxUnitSelectionForm from "./OpenedxUnitSelectionForm"
import { MathJaxContext } from "better-react-mathjax"
import AiChatDisplay from "./StyledAiChatDisplay"
import BaseChatContent from "./BaseChatContent"

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
    tutor_prompt: "",
  })

  const vertical = useV2Block({
    blockUsageKey: settings.tutor_vertical,
  })

  const siblings = Object.keys(vertical.data?.blocks ?? {}).filter(
    (key) => key !== vertical.data?.root,
  )

  const isReady = vertical.isSuccess

  return (
    <BaseChatContent
      title="AssessmentGPT"
      apiUrl={ASSESSMENT_GPT_URL}
      chatIdPrefix="assessment-gpt"
      conversationStarters={CONVERSATION_STARTERS}
      initialMessages={INITIAL_MESSAGES}
      promptQueryKey=""
      promptSettingKey=""
      modelSettingKey="tutor_model"
      isReady={isReady}
      extraBody={{
        model: settings.tutor_model,
        block_siblings: siblings,
        edx_module_id: settings.tutor_unit,
      }}
      settings={settings}
      setSettings={setSettings}
      showSystemPrompt={false}
      sidebarContent={
        <>
          <OpenEdxLoginAlert />
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
        </>
      }
      chatDisplayProps={{
        useMathJax: true,
      }}
    >
      <MathJaxContext>
        <AiChatDisplay
          entryScreenEnabled={false}
          conversationStarters={CONVERSATION_STARTERS}
          useMathJax={true}
        />
      </MathJaxContext>
    </BaseChatContent>
  )
}

export default AssessmentContent
