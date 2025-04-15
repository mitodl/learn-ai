import { SYLLABUS_GPT_URL } from "@/services/ai/urls"
import { AiChat } from "@mitodl/smoot-design/ai"
import type { AiChatProps } from "@mitodl/smoot-design/ai"
import Typography from "@mui/material/Typography"
import Grid from "@mui/material/Grid2"
import TextField from "@mui/material/TextField"
import { useState } from "react"
import SelectModel from "./SelectModel"
import { getRequestOpts } from "./util"
import { useQuery, UseQueryResult } from "@tanstack/react-query"
import { learningResourcesQueries } from "@/services/learn"
import { LearningResource } from "@mitodl/open-api-axios/v1"

const CONVERSATION_STARTERS: AiChatProps["conversationStarters"] = [
  {
    content: "What topics are covered in this course?",
  },
]

const getResourceId = (url: string) => {
  if (!url) {
    return { id: null, errMsg: "URL is required." }
  }
  try {
    const urlObj = new URL(url)
    const resource = urlObj.searchParams.get("resource")
    if (!resource) {
      return { id: null, errMsg: "URL did not include 'resource' param." }
    }
    const id = +resource
    if (!Number.isFinite(id)) {
      return { id: null, errMsg: "URL's 'resource' param is not a number" }
    }
    return { id, errMsg: null }
  } catch (err) {
    if (err instanceof Error) {
      return { id: null, errMsg: err.message }
    }
    return { id: null, errMsg: "Unknown Error" }
  }
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
      Found <strong>{resource.data.title}</strong>
    </span>
  )
}

const DEFAULT_RESOURCE_URL = "https://learn.mit.edu/search?resource=2812"

const SyllabusContent = () => {
  const [model, setModel] = useState<string | undefined>(undefined)
  const [resourceUrl, setResourceUrl] = useState<string>(DEFAULT_RESOURCE_URL)
  const { id: resourceId, errMsg } = getResourceId(resourceUrl || "")
  const resource = useQuery({
    ...learningResourcesQueries.retrieve({ id: resourceId ?? -1 }),
    enabled: !!resourceId,
  })
  const requestOpts = getRequestOpts({
    apiUrl: SYLLABUS_GPT_URL,
    extraBody: { model, course_id: resource.data?.readable_id },
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
            label="Resource URL"
            fullWidth
            value={resourceUrl || ""}
            onChange={(e) => setResourceUrl(e.target.value)}
            sx={{ marginBottom: 2 }}
            error={!!errMsg || resource.isError}
            helperText={getResourceHelpText(errMsg, resource)}
          />
          <SelectModel onChange={setModel} />
        </Grid>
      </Grid>
    </>
  )
}

export default SyllabusContent
