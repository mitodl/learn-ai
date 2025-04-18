import { RECOMMENDATION_GPT_URL } from "@/services/ai/urls"
import AiChat from "./StyledAiChat"
import type { AiChatProps } from "@mitodl/smoot-design/ai"
import Typography from "@mui/material/Typography"
import Grid from "@mui/material/Grid2"
import SelectModel from "./SelectModel"
import React, { useMemo } from "react"
import { getRequestOpts, useSearchParamSettings } from "./util"

const CONVERSATION_STARTERS: AiChatProps["conversationStarters"] = [
  {
    content: "What are some good courses to take for a career in data science?",
  },
]

const RecommendationContent: React.FC = () => {
  const [settings, setSettings] = useSearchParamSettings({
    rec_model: "",
  })

  const requestOpts = useMemo(
    () =>
      getRequestOpts({
        apiUrl: RECOMMENDATION_GPT_URL,
        extraBody: { model: settings.rec_model },
      }),
    [settings.rec_model],
  )

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
      <Grid container spacing={2} sx={{ padding: 2 }}>
        <Grid
          size={{ xs: 12, md: 8 }}
          sx={{ position: "relative", minHeight: "600px" }}
        >
          <AiChat
            chatId="recommendation-gpt"
            entryScreenEnabled={false}
            conversationStarters={CONVERSATION_STARTERS}
            requestOpts={requestOpts}
          />
        </Grid>
        <Grid
          size={{ xs: 12, md: 4 }}
          sx={{ display: "flex", flexDirection: "column", gap: "16px" }}
        >
          <SelectModel
            value={settings.rec_model}
            onChange={(e) => setSettings({ rec_model: e.target.value })}
          />
        </Grid>
      </Grid>
    </>
  )
}

export default RecommendationContent
