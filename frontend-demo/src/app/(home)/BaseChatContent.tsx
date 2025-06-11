import React, { useEffect, useState } from "react"
import { AiChatProvider, type AiChatProps } from "@mitodl/smoot-design/ai"
import Typography from "@mui/material/Typography"
import Grid from "@mui/material/Grid2"
import TextareaAutosize from "@mui/material/TextareaAutosize"
import { FormLabel } from "@mui/material"
import CircularProgress from "@mui/material/CircularProgress"
import { Button } from "@mitodl/smoot-design"
import { useQuery } from "@tanstack/react-query"
import { promptQueries, userQueries } from "@/services/ai"
import { useRequestOpts } from "./util"
import AiChatDisplay from "./StyledAiChatDisplay"
import SelectModel from "./SelectModel"
import MetadataDisplay from "./MetadataDisplay"

export interface BaseChatContentProps {
  title: string
  apiUrl: string
  chatIdPrefix: string
  conversationStarters?: AiChatProps["conversationStarters"]
  initialMessages?: AiChatProps["initialMessages"]
  promptQueryKey: string
  promptSettingKey: string
  modelSettingKey: string
  isReady: boolean
  extraBody: Record<string, unknown>
  settings: Record<string, unknown>
  setSettings: (patch: Record<string, unknown>) => void
  sidebarContent?: React.ReactNode
  children?: React.ReactNode
  chatDisplayProps?: Partial<React.ComponentProps<typeof AiChatDisplay>>
  showSystemPrompt?: boolean
}

const BaseChatContent: React.FC<BaseChatContentProps> = ({
  title,
  apiUrl,
  chatIdPrefix,
  conversationStarters = [],
  initialMessages,
  promptQueryKey,
  promptSettingKey,
  modelSettingKey,
  isReady,
  extraBody,
  settings,
  setSettings,
  sidebarContent,
  children,
  chatDisplayProps = {},
  showSystemPrompt = true,
}) => {
  const me = useQuery(userQueries.me())
  const promptResult = useQuery({
    ...promptQueries.get(promptQueryKey),
    enabled: !!promptQueryKey,
  })
  const [promptText, setPromptText] = useState(
    settings[promptSettingKey] as string,
  )

  useEffect(() => {
    const nextValue = promptText || promptResult.data?.prompt_value || ""
    if (nextValue === promptResult.data?.prompt_value || !me.data?.is_staff) {
      // If the prompt is identical to the default, or user is not staff,
      // don't send it in the request.
      setSettings({ [promptSettingKey]: "" })
    } else if (settings[promptSettingKey] !== promptText) {
      setSettings({ [promptSettingKey]: nextValue })
    }
  }, [
    promptText,
    setSettings,
    settings,
    promptSettingKey,
    promptResult.data?.prompt_value,
  ])

  const { requestOpts, requestNewThread, chatSuffix } = useRequestOpts({
    apiUrl,
    extraBody,
  })

  const chatId = `${chatIdPrefix}-${chatSuffix}`

  return (
    <>
      <Typography
        variant="h3"
        sx={(theme) => ({
          color: theme.custom?.colors?.darkGray2,
        })}
      >
        {title}
      </Typography>
      <AiChatProvider
        chatId={chatId}
        requestOpts={requestOpts}
        initialMessages={initialMessages}
      >
        <Grid container spacing={2} sx={{ padding: 2 }}>
          <Grid
            size={{ xs: 12, md: 8 }}
            sx={{ position: "relative", minHeight: "600px" }}
            inert={!isReady}
          >
            {children || (
              <AiChatDisplay
                entryScreenEnabled={false}
                conversationStarters={conversationStarters}
                {...chatDisplayProps}
              />
            )}
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
            {sidebarContent}
            <SelectModel
              value={settings[modelSettingKey] as string}
              onChange={(e) => {
                setSettings({ [modelSettingKey]: e.target.value })
                requestNewThread()
              }}
            />
            {showSystemPrompt && me.data?.is_staff ? (
              <>
                <FormLabel htmlFor={`${promptSettingKey}-ta`}>
                  System Prompt
                </FormLabel>
                <TextareaAutosize
                  id={`${promptSettingKey}-ta`}
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

export default BaseChatContent
