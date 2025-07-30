import { ASSESSMENT_GPT_URL } from "@/services/ai/urls"
import { type AiChatProps } from "@mitodl/smoot-design/ai"
import { useSearchParamSettings } from "./util"
import { MathJaxContext } from "better-react-mathjax"
import AiChatDisplay from "./StyledAiChatDisplay"
import BaseChatContent from "./BaseChatContent"
import { useQuery } from "@tanstack/react-query"
import { useMemo } from "react"
import React from "react"
import TextField from "@mui/material/TextField"
import { useFormik, Field } from "formik"
import MenuItem from "@mui/material/MenuItem"

const CONVERSATION_STARTERS: AiChatProps["conversationStarters"] = []
const INITIAL_MESSAGES: AiChatProps["initialMessages"] = [
  { role: "assistant", content: "Hi, do you need any help?" },
]
const DEFAULT_COURSE_RUN = "14566-kaleba:20211202+canvas"
type CanvasProblemSelectionFormProps = {
  selectedRun: string
  selectedProblemSet: string
  defaultRun: string
  defaultProblemSet: string
  problemSetList: string[] | []
  onChange: (values: { run: string; problem_set: string }) => void
  onReset: () => void
}

const CanvasProblemSelectionForm: React.FC<CanvasProblemSelectionFormProps> = ({
  selectedRun,
  selectedProblemSet,
  defaultRun,
  defaultProblemSet,
  problemSetList,
  onChange,
  onReset,
}) => {
  const [editing, setEditing] = React.useState(false)

  const formik = useFormik({
    enableReinitialize: true,
    initialValues: {
      run: defaultRun,
      problem_set: defaultProblemSet,
    },
    onSubmit: (values) => {
      setEditing(false)
      onChange(values)
    },
    validateOnChange: false,
  })

  return (
    <form onSubmit={formik.handleSubmit}>
      <TextField
        size="small"
        label="Course Run Readable ID"
        fullWidth
        autoCapitalize="off"
        spellCheck={false}
        margin="normal"
        name="run"
        value={formik.values.run}
        onChange={(e) => {
          console.log("run changed", e.target.value)
          formik.handleChange(e)
          // Call onChange with updated values so parent can update settings and trigger useQuery
          onChange({
            run: e.target.value,
            problem_set: formik.values.problem_set,
          })
        }}
      />
      <TextField
        label="Problem Set"
        size="small"
        margin="normal"
        name="problem_set"
        fullWidth
        select
        onChange={formik.handleChange}
        value={formik.values.problem_set}
      >
        {problemSetList.map((problemSet) => (
          <MenuItem key={problemSet} value={problemSet}>
            {problemSet}
          </MenuItem>
        ))}
      </TextField>
    </form>
  )
}

const CanvasAssessmentContent = () => {
  const [settings, setSettings] = useSearchParamSettings({
    tutor_model: "",
    run: DEFAULT_COURSE_RUN,
    problem_set: "",
    tutor_prompt: "",
  })

  const getProblemSetList = async (runReadableId: string) => {
    const response = await fetch(
      `${process.env.NEXT_PUBLIC_MITOL_API_BASE_URL}/api/v0/get_problem_set_list/?run_readable_id=${encodeURIComponent(runReadableId)}`,
    )
    return response.json()
  }

  const problemSetListResult = useQuery({
    queryKey: ["problemSetList", settings.run],
    queryFn: () => getProblemSetList(settings.run),
    enabled: !!settings.run,
  })

  const { problemSetList: problemSetList, error: problemSetListError } =
    useMemo(() => {
      if (problemSetListResult.isLoading)
        return { problemSetList: [], error: null }
      if (problemSetListResult.data.error) {
        return { problemSetList: [], error: problemSetListResult.data.error }
      }

      return {
        problemSetList: problemSetListResult.data.problem_set_titles,
        error: null,
      }
    }, [settings.run, problemSetListResult])

  const isReady = !!problemSetList
  console.log("problemSetList", problemSetList)

  return (
    <div>
      <CanvasProblemSelectionForm
        selectedRun={settings.run}
        selectedProblemSet={settings.problem_set}
        defaultRun={DEFAULT_COURSE_RUN}
        defaultProblemSet={problemSetList.length > 0 ? problemSetList[0] : ""}
        problemSetList={problemSetList || []}
        onChange={(values) => {
          setSettings({
            problem_set: values.problem_set,
            run: values.run,
          })
        }}
        onReset={() => {
          setSettings({
            run: null,
            problem_set: null,
          })
        }}
      />
    </div>
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
