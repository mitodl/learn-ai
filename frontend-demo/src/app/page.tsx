"use client"
import React from "react"
import { AiChat, AiChatProps } from "@mitodl/smoot-design/ai"
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
  apiUrl: `${process.env.NEXT_PUBLIC_MITOL_API_BASE_URL}/http/recommendation_agent/`,
  transformBody(messages) {
    const message = messages[messages.length - 1].content
    return { message }
  },
  fetchOpts: {
    credentials: "include",
  },
}

const StyledChat = styled(AiChat)({
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
