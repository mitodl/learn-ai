import { RECOMMENDATION_GPT_URL } from "@/services/ai/urls"
import { AiChat } from "@mitodl/smoot-design/ai"
import type { AiChatProps } from "@mitodl/smoot-design/ai"
import Typography from "@mui/material/Typography"
import Grid from "@mui/material/Grid2"
import SelectModel from "./SelectModel"
import React, { useMemo, useState } from "react"
import { getRequestOpts } from "./util"

const CONVERSATION_STARTERS: AiChatProps["conversationStarters"] = [
  {
    content: "What are some good courses to take for a career in data science?",
  },
]

const RecommendationContent: React.FC = () => {
  const [model, setModel] = useState<string | undefined>(undefined)

  const requestOpts = useMemo(
    () =>
      getRequestOpts({ apiUrl: RECOMMENDATION_GPT_URL, extraBody: { model } }),
    [model],
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
          <Typography
            variant="h5"
            sx={(theme) => ({
              color: theme.custom.colors.silverGrayDark,
            })}
          >
            Settings
          </Typography>
          <SelectModel onChange={setModel} />
        </Grid>
      </Grid>
    </>
  )
}

export default RecommendationContent
