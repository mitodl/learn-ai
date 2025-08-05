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
import { useFormik } from "formik"
import MenuItem from "@mui/material/MenuItem"
import Button from "@mui/material/Button"

const CONVERSATION_STARTERS: AiChatProps["conversationStarters"] = []
const INITIAL_MESSAGES: AiChatProps["initialMessages"] = []
const DEFAULT_COURSE_RUN = "14566-kaleba:20211202+canvas"
type CanvasProblemSelectionFormProps = {
  defaultRun: string
  problemSetList: string[] | []
  onChange: (values: { run: string; problem_set_title: string }) => void
}

const CanvasProblemSelectionForm: React.FC<CanvasProblemSelectionFormProps> = ({
  defaultRun,
  problemSetList,
  onChange,
}) => {
  const formik = useFormik({
    enableReinitialize: true,
    initialValues: {
      run: defaultRun,
      problem_set_title:
        problemSetList && problemSetList.length > 0 ? problemSetList[0] : "",
    },
    onSubmit: (values) => {
      onChange({
        run: values.run,
        problem_set_title: "",
      })
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
        onChange={formik.handleChange}
      />
      <Button
        type="submit"
        variant="contained"
        color="primary"
        sx={{ mt: 1, mb: 2 }}
      >
        Update Course Run
      </Button>
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
      `${process.env.NEXT_PUBLIC_MITOL_API_BASE_URL}/api/v0/problem_set_list/?run_readable_id=${encodeURIComponent(runReadableId)}`,
    )
    return response.json()
  }

  const problemSetListResult = useQuery({
    queryKey: ["problemSetList", settings.run],
    queryFn: () => getProblemSetList(settings.run),
    enabled: !!settings.run,
  })

  const { problemSetList: problemSetList } = useMemo(() => {
    if (problemSetListResult.isLoading) return { problemSetList: [] }
    if (problemSetListResult.data.error) {
      return { problemSetList: [] }
    }

    return {
      problemSetList: problemSetListResult.data.problem_set_titles,
      error: null,
    }
  }, [problemSetListResult, settings.run])

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
            defaultRun={settings.run}
            problemSetList={problemSetList || []}
            onChange={(values) => {
              setSettings({
                problem_set_title: values.problem_set_title,
                run: values.run,
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
