import { SYLLABUS_GPT_URL } from "@/services/ai/urls"
import { AiChat } from "@mitodl/smoot-design/ai"
import type { AiChatProps } from "@mitodl/smoot-design/ai"
import Link from "@mui/material/Link"
import Typography from "@mui/material/Typography"
import Grid from "@mui/material/Grid2"
import TextField from "@mui/material/TextField"
import SelectModel from "./SelectModel"
import { getRequestOpts, useSearchParamSettings } from "./util"
import { useQuery, UseQueryResult } from "@tanstack/react-query"
import { learningResourcesQueries } from "@/services/learn"
import { LearningResource } from "@mitodl/open-api-axios/v1"
import { useEffect, useState } from "react"

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
    console.log({ id })
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
  })
  const [resourceParseError, setResourceParseError] = useState<string | null>(
    null,
  )
  useEffect(() => {
    /**
     * This changes what the user entered to extract just the resource ID.
     * Delay it so they see the URL they copy-pasted.
     */
    setTimeout(() => {
      const { id, errMsg } = getResourceId(settings.syllabus_resource)
      setResourceParseError(errMsg)
      if (id) {
        setSettings({ syllabus_resource: id.toString() })
      }
    }, 150)
  }, [settings.syllabus_resource, setSettings])
  const resourceId = Number.isFinite(+settings.syllabus_resource)
    ? +settings.syllabus_resource
    : -1
  const resource = useQuery({
    ...learningResourcesQueries.retrieve({
      id: resourceId,
    }),
    enabled: !!resourceId,
  })
  const requestOpts = getRequestOpts({
    apiUrl: SYLLABUS_GPT_URL,
    extraBody: {
      model: settings.syllabus_model,
      course_id: resource.data?.readable_id,
    },
  })

  return (
    <>
      <Typography variant="h3">SyllabusGPT</Typography>
      <Grid container spacing={2} sx={{ padding: 2 }}>
        <Grid
          size={{ xs: 12, md: 8 }}
          sx={{ position: "relative", minHeight: "600px" }}
        >
          <AiChat
            chatId="syllabus-gpt"
            entryScreenEnabled={false}
            conversationStarters={CONVERSATION_STARTERS}
            requestOpts={requestOpts}
          />
        </Grid>
        <Grid size={{ xs: 12, md: 4 }}>
          <TextField
            size="small"
            label="Resource ID or Learn Resource URL"
            fullWidth
            value={settings.syllabus_resource}
            onChange={(e) => {
              setSettings({ syllabus_resource: e.target.value })
            }}
            sx={{ marginBottom: 2 }}
            error={!!resourceParseError || resource.isError}
            helperText={getResourceHelpText(resourceParseError, resource)}
          />
          <SelectModel
            value={settings.syllabus_model}
            onChange={(e) => setSettings({ syllabus_model: e.target.value })}
          />
        </Grid>
      </Grid>
    </>
  )
}

export default SyllabusContent
