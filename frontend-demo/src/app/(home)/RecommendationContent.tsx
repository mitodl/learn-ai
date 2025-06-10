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
import { promptQueries, userQueries } from "@/services/ai"
import { useQuery } from "@tanstack/react-query"
import { FormLabel } from "@mui/material"

const CONVERSATION_STARTERS: AiChatProps["conversationStarters"] = [
  {
    content: "What are some good courses to take for a career in data science?",
  },
]

const RecommendationContent: React.FC = () => {
  const [settings, setSettings] = useSearchParamSettings({
    rec_model: "",
    search_url: "",
    rec_prompt: "",
  })

  const me = useQuery(userQueries.me())
  const promptResult = useQuery(promptQueries.get("recommendation"))
  const [promptText, setPromptText] = useState(settings.rec_prompt)
  useEffect(() => {
    const nextValue = promptText || promptResult.data?.prompt_value || ""
    if (nextValue === promptResult.data?.prompt_value) {
      // If the prompt is identical to the default, don't send it in the request.
      setSettings({ rec_prompt: "" })
    } else if (settings.rec_prompt !== promptText) {
      setSettings({ rec_prompt: nextValue })
    }
  }, [
    promptText,
    setSettings,
    settings.rec_prompt,
    promptResult.data?.prompt_value,
  ])

  const { requestOpts, requestNewThread, chatSuffix } = useRequestOpts({
    apiUrl: RECOMMENDATION_GPT_URL,
    extraBody: {
      model: settings.rec_model,
      search_url: settings.search_url,
      instructions: settings.rec_prompt,
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
            {me.data?.is_staff ? (
              <>
                <FormLabel htmlFor="rec-prompt-ta">System Prompt</FormLabel>
                <TextareaAutosize
                  id="rec-prompt-ta"
                  minRows={5}
                  maxRows={15}
                  value={promptText || promptResult.data?.prompt_value || ""}
                  onChange={(e) => setPromptText(e.target.value)}
                />
              </>
            ) : (
              <></>
            )}
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
