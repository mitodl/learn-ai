import { CANVAS_ASSESSMENT_GPT_URL } from "@/services/ai/urls"
import { type AiChatProps } from "@mitodl/smoot-design/ai"
import { useSearchParamSettings } from "./util"
import { MathJaxContext } from "better-react-mathjax"
import AiChatDisplay from "./StyledAiChatDisplay"
import BaseChatContent from "./BaseChatContent"
import { useQuery } from "@tanstack/react-query"
import { useMemo, useEffect } from "react"
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
  problemSetList: string[] | []
  onChange: (values: { run: string; problem_set_title: string }) => void
  onReset: () => void
}

const CanvasProblemSelectionForm: React.FC<CanvasProblemSelectionFormProps> = ({
  selectedRun,
  selectedProblemSet,
  defaultRun,
  problemSetList,
  onChange,
  onReset,
}) => {
  const [editing, setEditing] = React.useState(false)

  const formik = useFormik({
    enableReinitialize: true,
    initialValues: {
      run: defaultRun,
      problem_set_title: problemSetList[0] || "",
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
          formik.handleChange(e)
          // Call onChange with updated values so parent can update settings and trigger useQuery
          onChange({
            run: e.target.value,
            problem_set_title: formik.values.problem_set_title,
          })
        }}
      />
      <TextField
        label="Problem Set"
        size="small"
        margin="normal"
        name="problem_set_title"
        fullWidth
        select
        onChange={(e) => {
          formik.handleChange(e)
          // Call onChange with updated values so parent can update settings and trigger useQuery
          onChange({
            problem_set_title: e.target.value,
            run: formik.values.run,
          })
        }}
        value={formik.values.problem_set_title}
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
    problem_set_title: "",
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

  useEffect(() => {
    // Set problem_set_title to first item or "" whenever problemSetList changes
    if (problemSetList && problemSetList.length > 0) {
      if (settings.problem_set_title !== problemSetList[0]) {
        setSettings({
          problem_set_title: problemSetList[0],
        })
      }
    } else if (settings.problem_set_title !== "") {
      setSettings({
        problem_set_title: "",
      })
    }
  }, [problemSetList])

  const isReady = !!problemSetList

  return (
    <BaseChatContent
      title="CanvasAssessmentGPT"
      apiUrl={CANVAS_ASSESSMENT_GPT_URL}
      chatIdPrefix="canvas-assessment-gpt"
      conversationStarters={CONVERSATION_STARTERS}
      initialMessages={INITIAL_MESSAGES}
      promptQueryKey=""
      promptSettingKey=""
      modelSettingKey="tutor_model"
      isReady={isReady}
      extraBody={{
        model: settings.tutor_model,
        run_readable_id: settings.run,
        problem_set_title: settings.problem_set_title,
      }}
      settings={settings}
      setSettings={setSettings}
      showSystemPrompt={false}
      sidebarContent={
        <>
          <CanvasProblemSelectionForm
            selectedRun={settings.run}
            selectedProblemSet={settings.problem_set_title}
            defaultRun={DEFAULT_COURSE_RUN}
            problemSetList={problemSetList || []}
            onChange={(values) => {
              setSettings({
                problem_set_title: values.problem_set_title,
                run: values.run,
              })
            }}
            onReset={() => {
              setSettings({
                run: null,
                problem_set_title: null,
              })
            }}
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

export default CanvasAssessmentContent
