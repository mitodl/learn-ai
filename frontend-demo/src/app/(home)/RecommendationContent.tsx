import React from "react"
import { RECOMMENDATION_GPT_URL } from "@/services/ai/urls"
import { type AiChatProps } from "@mitodl/smoot-design/ai"
import { useSearchParamSettings } from "./util"
import BaseChatContent from "./BaseChatContent"

const CONVERSATION_STARTERS: AiChatProps["conversationStarters"] = [
  {
    content: "What are some good courses to take for a career in data science?",
  },
]

const RecommendationContent: React.FC = () => {
  const [settings, setSettings] = useSearchParamSettings({
    rec_model: "",
    rec_prompt: "",
  })

  return (
    <BaseChatContent
      title="RecommendationGPT"
      apiUrl={RECOMMENDATION_GPT_URL}
      chatIdPrefix="recommendation-gpt"
      conversationStarters={CONVERSATION_STARTERS}
      promptQueryKey="recommendation"
      promptSettingKey="rec_prompt"
      modelSettingKey="rec_model"
      isReady={true}
      extraBody={{
        model: settings.rec_model,
        instructions: settings.rec_prompt,
      }}
      settings={settings}
      setSettings={setSettings}
    />
  )
}

export default RecommendationContent
