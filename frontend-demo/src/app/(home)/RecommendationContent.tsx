import { useEffect, useState } from "react"
import { RECOMMENDATION_GPT_URL } from "@/services/ai/urls"
import AiChatDisplay from "./StyledAiChatDisplay"
import { AiChatProvider, type AiChatProps } from "@mitodl/smoot-design/ai"
import Typography from "@mui/material/Typography"
import Grid from "@mui/material/Grid2"
import TextareaAutosize from "@mui/material/TextareaAutosize"

import SelectModel from "./SelectModel"
import React from "react"
import { useRequestOpts, useSearchParamSettings } from "./util"
import SelectSearchURL from "./SelectSearchUrl"
import MetadataDisplay from "./MetadataDisplay"
import { Button } from "@mitodl/smoot-design"
import { promptQueries } from "@/services/ai"
import { useQuery } from "@tanstack/react-query"

const CONVERSATION_STARTERS: AiChatProps["conversationStarters"] = [
  {
    content: "What are some good courses to take for a career in data science?",
  },
]

const RecommendationContent: React.FC = () => {
  const [settings, setSettings] = useSearchParamSettings({
    rec_model: "",
    search_url: "",
    systemPrompt: "initial",
  })

  const promptResult = useQuery(promptQueries.get("recommendation"))
  const [promptText, setPromptText] = useState(settings.systemPrompt)
  useEffect(() => {
    const nextValue =
      promptText || promptText || promptResult.data?.prompt_value || ""
    if (settings.systemPrompt !== nextValue) {
      setSettings({ systemPrompt: nextValue })
    }
  }, [promptText, setSettings, settings.systemPrompt])

  const { requestOpts, requestNewThread, chatSuffix } = useRequestOpts({
    apiUrl: RECOMMENDATION_GPT_URL,
    extraBody: {
      model: settings.rec_model,
      search_url: settings.search_url,
      instructions: settings.systemPrompt,
    },
  })

  const chatId = `recommendation-gpt-${chatSuffix}`

  return (
    <>
      <Typography
        variant="h3"
        sx={(theme) => ({
          color: theme.custom.colors.darkGray2,
        })}
      >
        RecommendationGPT
      </Typography>
      <AiChatProvider chatId={chatId} requestOpts={requestOpts}>
        <Grid container spacing={2} sx={{ padding: 2 }}>
          <Grid
            size={{ xs: 12, md: 8 }}
            sx={{ position: "relative", minHeight: "600px" }}
          >
            <AiChatDisplay
              entryScreenEnabled={false}
              conversationStarters={CONVERSATION_STARTERS}
            />
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
              value={settings.rec_model}
              onChange={(e) => {
                setSettings({ rec_model: e.target.value })
                requestNewThread()
              }}
            />
            <SelectSearchURL
              value={settings.search_url}
              onChange={(e) => setSettings({ search_url: e.target.value })}
            />
            <TextareaAutosize
              aria-label="System Prompt"
              minRows={6}
              maxRows={10}
              value={
                promptText === "initial"
                  ? promptResult.data?.prompt_value
                  : promptText || ""
              }
              onChange={(e) => setPromptText(e.target.value)}
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

export default RecommendationContent
