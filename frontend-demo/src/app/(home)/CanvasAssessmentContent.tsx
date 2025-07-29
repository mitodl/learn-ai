import { ASSESSMENT_GPT_URL } from "@/services/ai/urls"
import { type AiChatProps } from "@mitodl/smoot-design/ai"
import { useSearchParamSettings } from "./util"
import { MathJaxContext } from "better-react-mathjax"
import AiChatDisplay from "./StyledAiChatDisplay"
import BaseChatContent from "./BaseChatContent"
import CanvasProblemSelectionForm from "./CanvasProblemSelectionForm"

const CONVERSATION_STARTERS: AiChatProps["conversationStarters"] = []
const INITIAL_MESSAGES: AiChatProps["initialMessages"] = [
  { role: "assistant", content: "Hi, do you need any help?" },
]
const DEFAULT_COURSE_RUN = "14566-kaleba:20211202+canvas"
const DEFAULT_PROBLEM_SET = "Problem Set 1"


const CanvasAssessmentContent = () => {
  const [settings, setSettings] = useSearchParamSettings({
    tutor_model: "",
    run: DEFAULT_COURSE_RUN,
    problem_set: DEFAULT_PROBLEM_SET,
    tutor_prompt: "",
  })

  return (
    <CanvasProblemSelectionForm
      selectedRun={settings.run}
      selectedProblemSet={settings.problem_set}
      defaultRun={DEFAULT_COURSE_RUN}
      defaultProblemSet={DEFAULT_PROBLEM_SET}
      onSubmit={(values) => {
        setSettings({
          tutor_model: values.model,
          run: values.run,
          tutor_prompt: values.prompt,
        })
      }}
      onReset={() => {
        setSettings({
          tutor_model: null,
          run: null,
          problem_set: null,
        })
      }}
    />
  )
}
  
/*
  return (
    <BaseChatContent
      title="CanvasAssessmentGPT"
      apiUrl={ASSESSMENT_GPT_URL}
      chatIdPrefix="canvas-assessment-gpt"
      conversationStarters={CONVERSATION_STARTERS}
      initialMessages={INITIAL_MESSAGES}
      promptQueryKey=""
      promptSettingKey=""
      modelSettingKey="canvas_tutor_model"
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
          <CanvasProblemSelectionForm
            selectedRun={settings.tutor_course_run}
            selectedProblemSet={settings.tutor_problem_set}
            defaultRun={DEFAULT_COURSE_RUN}
            defaultProblemSet={settings.tutor_problem_set}
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
?*/

export default CanvasAssessmentContent
