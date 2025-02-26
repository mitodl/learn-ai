"use client"
import React from "react"
import { AiChat, AiChatProps } from "@mitodl/smoot-design/ai"
import styled from "@emotion/styled"

/**
 * Extracts a JSON object from a comment string
 * @param comment the comment string
 * @returns the JSON object
 */
const extractJSONFromComment = (comment: string) => {
  const jsonStr = comment.toString().match(/<!-{2}(.*)-{2}>/)?.[1] || "{}"
  try {
    return JSON.parse(jsonStr)
  } catch (e) {
    console.error("error parsing JSON from comment", comment, e)
    return {}
  }
}

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

const Container = styled.div({
  height: "calc(80vh - 72px)",
  marginTop: "16px",
})

const HomePage: React.FC = () => {
  const parseContent = (content: string | unknown) => {
    if (typeof content !== "string") {
      return ""
    }
    const contentParts = content.split("<!--")
    if (contentParts.length > 1) {
      // to do: show debug info if enabled here
      extractJSONFromComment(contentParts[1])
    }
    return contentParts[0]
  }

  return (
    <Container>
      <AiChat
        initialMessages={INITIAL_MESSAGES}
        requestOpts={REQUEST_OPTS}
        conversationStarters={STARTERS}
        parseContent={parseContent}
      />
    </Container>
  )
}

export default HomePage
