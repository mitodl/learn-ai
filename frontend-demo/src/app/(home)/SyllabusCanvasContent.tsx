import { useEffect, useMemo, useState } from "react"
import { SYLLABUS_CANVAS_GPT_URL } from "@/services/ai/urls"
import { type AiChatProps } from "@mitodl/smoot-design/ai"
import TextField from "@mui/material/TextField"
import { useSearchParamSettings } from "./util"
import BaseChatContent from "./BaseChatContent"

const CONVERSATION_STARTERS: AiChatProps["conversationStarters"] = [
  {
    content: "What topics are covered in this course?",
  },
]

const getReadableId = (readableId: string) => {
  if (!readableId) {
    return { id: null, errMsg: "Resource readable_id is required." }
  }
  return { id: readableId, errMsg: null }
}

// https://learn.mit.edu/?resource=16875
const DEFAULT_READABLE_ID = "14566-kaleba:20211202"

const SyllabusCanvasContent = () => {
  const [settings, setSettings] = useSearchParamSettings({
    syllabus_model: "",
    syllabus_readable_id: DEFAULT_READABLE_ID,
    syllabus_prompt: "",
  })
  const [readableIdText, setReadableIdText] = useState(
    settings.syllabus_readable_id,
  )

  // Compute derived state during render instead of in effect
  const { id: parsedId, errMsg: resourceParseError } = useMemo(
    () => getReadableId(readableIdText),
    [readableIdText],
  )

  // Update settings when input changes and is valid
  useEffect(() => {
    if (resourceParseError === null && parsedId !== null) {
      if (settings.syllabus_readable_id !== parsedId) {
        setSettings({ syllabus_readable_id: parsedId })
      }
    }
  }, [parsedId, resourceParseError, setSettings, settings.syllabus_readable_id])

  // Sync readableIdText with settings when settings change (e.g., when navigating back to tab)
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional sync from URL params to local state
    setReadableIdText(settings.syllabus_readable_id)
  }, [settings.syllabus_readable_id])

  const readableId = settings.syllabus_readable_id || DEFAULT_READABLE_ID

  const isReady = !!readableId

  return (
    <BaseChatContent
      title="SyllabusCanvasGPT"
      apiUrl={SYLLABUS_CANVAS_GPT_URL}
      chatIdPrefix="syllabus-canvas-gpt"
      conversationStarters={CONVERSATION_STARTERS}
      promptQueryKey="syllabus_canvas"
      promptSettingKey="syllabus_canvas_prompt"
      modelSettingKey="syllabus_model"
      isReady={isReady}
      extraBody={{
        model: settings.syllabus_model,
        course_id: readableId,
        related_resources: undefined,
        instructions: settings.syllabus_prompt,
      }}
      settings={settings}
      setSettings={setSettings}
      sidebarContent={
        <TextField
          size="small"
          margin="normal"
          label="Resource Readable ID"
          fullWidth
          value={readableIdText}
          onChange={(e) => setReadableIdText(e.target.value)}
          error={!!resourceParseError}
          helperText=""
        />
      }
    />
  )
}

export default SyllabusCanvasContent
