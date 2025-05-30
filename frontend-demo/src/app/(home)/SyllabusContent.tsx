import { SYLLABUS_GPT_URL } from "@/services/ai/urls"
import AiChatDisplay from "./StyledAiChatDisplay"
import { AiChatProvider, type AiChatProps } from "@mitodl/smoot-design/ai"
import Link from "@mui/material/Link"
import Typography from "@mui/material/Typography"
import Grid from "@mui/material/Grid2"
import TextField from "@mui/material/TextField"
import SelectModel from "./SelectModel"
import { useRequestOpts, useSearchParamSettings } from "./util"
import { useQuery, UseQueryResult } from "@tanstack/react-query"
import { learningResourcesQueries } from "@/services/learn"
import { LearningResource } from "@mitodl/open-api-axios/v1"
import { useEffect, useState } from "react"
import CircularProgress from "@mui/material/CircularProgress"
import MetadataDisplay from "./MetadataDisplay"
import { Button } from "@mitodl/smoot-design"

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
  })
  const [resourceParseError, setResourceParseError] = useState<string | null>(
    null,
  )
  const [resourceText, setResourceText] = useState(settings.syllabus_resource)
  useEffect(() => {
    const { id, errMsg } = getResourceId(resourceText)
    setResourceParseError(errMsg)

    const nextValue = String(id) ?? resourceText
    if (settings.syllabus_resource !== nextValue) {
      setSettings({ syllabus_resource: nextValue })
    }
  }, [resourceText, setSettings, settings.syllabus_resource])
  const resourceId = Number.isFinite(+settings.syllabus_resource)
    ? +settings.syllabus_resource
    : -1
  const resource = useQuery({
    ...learningResourcesQueries.retrieve({ id: resourceId }),
    enabled: !!resourceId,
  })
  const { requestOpts, chatSuffix, requestNewThread } = useRequestOpts({
    apiUrl: SYLLABUS_GPT_URL,
    extraBody: {
      model: settings.syllabus_model,
      course_id: resource.data?.readable_id,
      related_resources: resource.data?.children.map(
        (child) => child.readable_id,
      ),
    },
  })

  const isReady = resource.isSuccess
  const chatId = `syllabus-gpt-${chatSuffix}`
  return (
    <>
      <Typography variant="h3">SyllabusGPT</Typography>
      <AiChatProvider chatId={chatId} requestOpts={requestOpts}>
        <Grid container spacing={2} sx={{ padding: 2 }}>
          <Grid
            size={{ xs: 12, md: 8 }}
            sx={{ position: "relative", minHeight: "600px" }}
            inert={!isReady}
          >
            <AiChatDisplay
              entryScreenEnabled={false}
              conversationStarters={CONVERSATION_STARTERS}
            />
            {!isReady && (
              <CircularProgress
                color="primary"
                sx={{
                  position: "absolute",
                  zIndex: 1000,
                  top: "50%",
                  left: "50%",
                  transform: "translate(-50%, -50%)",
                }}
              />
            )}
          </Grid>
          <Grid
            size={{ xs: 12, md: 4 }}
            sx={{
              display: "flex",
              flexDirection: "column",
              gap: "16px",
            }}
          >
            <SelectModel
              value={settings.syllabus_model}
              onChange={(e) => {
                setSettings({ syllabus_model: e.target.value })
                requestNewThread()
              }}
            />
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
            <Button variant="secondary" onClick={requestNewThread}>
              Start new thread
            </Button>
          </Grid>
          <Grid size={{ xs: 12 }}>
            <MetadataDisplay />
          </Grid>
        </Grid>
      </AiChatProvider>
    </>
  )
}

export default SyllabusContent
