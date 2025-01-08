"use client"
import React from "react"
import { AiChatWs, AiChatProps } from "@mitodl/smoot-design/ai"
import styled from "@emotion/styled"

const INITIAL_MESSAGES: AiChatProps["initialMessages"] = [
  {
    content: "Hi! What are you interested in learning about?",
    role: "assistant",
  },
]

const STARTERS = [
  { content: "I'm interested in quantum computing" },
  { content: "I want to understand global warming. " },
  { content: "I am curious about AI applications for business" },
]

const REQUEST_OPTS: AiChatProps["requestOpts"] = {
  apiUrl: `${process.env.NEXT_PUBLIC_MITOL_WS_API_BASE_URL}/ws/recommendation_agent/`,
  transformBody(messages) {
    const message = messages[messages.length - 1].content
    return { message }
  },
}

const StyledChat = styled(AiChatWs)({
  height: "calc(80vh - 72px)",
  marginTop: "16px",
})

const HomePage: React.FC = () => {
  return (
    <StyledChat
      initialMessages={INITIAL_MESSAGES}
      requestOpts={REQUEST_OPTS}
      conversationStarters={STARTERS}
    />
  )
}

export default HomePage
