import { useEffect, useMemo, useState } from "react"
import { SYLLABUS_GPT_URL } from "@/services/ai/urls"
import { type AiChatProps } from "@mitodl/smoot-design/ai"
import Link from "@mui/material/Link"
import TextField from "@mui/material/TextField"
import { useSearchParamSettings } from "./util"
import { useQuery, UseQueryResult } from "@tanstack/react-query"
import { learningResourcesQueries } from "@/services/learn"
import { LearningResource } from "@mitodl/mit-learn-api-axios/v1"
import BaseChatContent from "./BaseChatContent"

const CONVERSATION_STARTERS: AiChatProps["conversationStarters"] = [
  {
    content: "What topics are covered in this course?",
  },
]

const getResourceId = (resource: string) => {
  if (!resource) {
    return { id: null, errMsg: "Resource is required." }
  }
  if (Number.isFinite(+resource)) {
    return { id: +resource, errMsg: null }
  }

  const errMsg =
    "Resource must be a numeric id or URL with `resource=<id>` param."
  try {
    const url = new URL(resource)
    const resourceId = url.searchParams.get("resource") ?? ""
    const id = +resourceId
    if (Number.isFinite(id)) {
      return { id, errMsg: null }
    }
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
  } catch (_err) {
    // pass, return below
  }
  return { id: null, errMsg }
}

const getResourceHelpText = (
  overridingErrorMessage: string | null,
  resource: UseQueryResult<LearningResource>,
) => {
  if (overridingErrorMessage) {
    return overridingErrorMessage
  }
  if (resource.isLoading) {
    return "Loading resource..."
  }
  if (resource.isError) {
    return "Resource not found."
  }
  if (!resource.data) {
    return "No resource found."
  }
  return (
    <span>
      Found{" "}
      <Link
        target="_blank"
        href={`${process.env.NEXT_PUBLIC_MIT_LEARN_APP_BASE_URL}?resource=${resource.data.id}`}
      >
        {resource.data.title}
      </Link>
    </span>
  )
}

// https://learn.mit.edu/?resource=2812
const DEFAULT_RESOURCE = "2812"

const SyllabusContent = () => {
  const [settings, setSettings] = useSearchParamSettings({
    syllabus_model: "",
    syllabus_resource: DEFAULT_RESOURCE,
    syllabus_prompt: "",
  })
  const [resourceText, setResourceText] = useState(settings.syllabus_resource)

  // Compute derived state during render instead of in effect
  const { id: parsedId, errMsg: resourceParseError } = useMemo(
    () => getResourceId(resourceText),
    [resourceText],
  )

  // Update settings when input changes and is valid
  useEffect(() => {
    const nextValue = String(parsedId) ?? resourceText
    if (settings.syllabus_resource !== nextValue) {
      setSettings({ syllabus_resource: nextValue })
    }
  }, [parsedId, resourceText, setSettings, settings.syllabus_resource])

  const resourceId = Number.isFinite(+settings.syllabus_resource)
    ? +settings.syllabus_resource
    : -1
  const resource = useQuery({
    ...learningResourcesQueries.retrieve({ id: resourceId }),
    enabled: !!resourceId,
  })

  const isReady = resource.isSuccess

  return (
    <BaseChatContent
      title="SyllabusGPT"
      apiUrl={SYLLABUS_GPT_URL}
      chatIdPrefix="syllabus-gpt"
      conversationStarters={CONVERSATION_STARTERS}
      promptQueryKey="syllabus"
      promptSettingKey="syllabus_prompt"
      modelSettingKey="syllabus_model"
      isReady={isReady}
      extraBody={{
        model: settings.syllabus_model,
        course_id: resource.data?.readable_id,
        related_resources: Array.isArray(resource.data?.children)
          ? resource.data.children.map((child) => child.readable_id)
          : undefined,
        instructions: settings.syllabus_prompt,
      }}
      settings={settings}
      setSettings={setSettings}
      sidebarContent={
        <TextField
          size="small"
          margin="normal"
          label="Resource ID or Learn Resource URL"
          fullWidth
          /**
           * don't use settings.syllabus_resource directly here to avoid
           * so that we can keep the updates syncrhonous to avoid
           * https://stackoverflow.com/questions/46000544/react-controlled-input-cursor-jumps
           */
          value={resourceText}
          onChange={(e) => setResourceText(e.target.value)}
          error={!!resourceParseError || resource.isError}
          helperText={getResourceHelpText(resourceParseError, resource)}
        />
      }
    />
  )
}

export default SyllabusContent
