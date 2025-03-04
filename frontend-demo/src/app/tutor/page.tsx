"use client"
import React from "react"
import { AiChat, AiChatProps } from "@mitodl/smoot-design/ai"
import styled from "@emotion/styled"
import ReactMarkdown from "react-markdown"
import { extractJSONFromComment } from "../utils"

/**
 * Makes a request to the tutor problem api view to get the problem and solution text
 * @param problemCode the id of the problem
 * @returns the problem and solution text as a JSON object  {problem: string, solution: string}
 */
const getProblemAndSolution = async (problemCode: string) => {
  const response = await fetch(
    `${process.env.NEXT_PUBLIC_MITOL_API_BASE_URL}/api/v0/tutor_problem/?problem_code=${problemCode}`,
  )
  return response.json()
}

const INITIAL_MESSAGES: AiChatProps["initialMessages"] = [
  {
    content: "Hi! Do you need any help solving the problem?",
    role: "assistant",
  },
]

const REQUEST_OPTS: AiChatProps["requestOpts"] = {
  apiUrl: `${process.env.NEXT_PUBLIC_MITOL_API_BASE_URL}/http/tutor_agent/`,
  transformBody(messages) {
    const message = messages[messages.length - 1].content
    const problemCode = "A1P1"
    return { message: message, problem_code: problemCode }
  },
  fetchOpts: {
    credentials: "include",
  },
}

const StyledChat = styled(AiChat)({
  marginTop: "16px",
})

const StyledProblem = styled.div({
  lineHeight: "1.4",
  fontSize: "0.875rem",
  ol: {
    paddingInlineStart: "0px",
  },
  p: {
    marginTop: "10px",
  },
  pre: {
    marginTop: "10px",
    backgroundColor: "#f0f0f0",
    padding: "15px",
    borderRadius: "5px",
  },
})

const TutorPage: React.FC = () => {
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

  const [problemAndSolution, setProblemAndSolution] = React.useState<{
    problem: string
    solution: string
  } | null>(null)

  React.useEffect(() => {
    ;(async () => {
      try {
        const data = await getProblemAndSolution("A1P1")
        setProblemAndSolution(data)
      } catch (error) {
        console.error("Failed to fetch problem and solution", error)
      }
    })()
  }, [])

  return (
    <div>
      <h3> Problem </h3>

      <StyledProblem>
        {problemAndSolution ? (
          <ReactMarkdown>{problemAndSolution["problem"]}</ReactMarkdown>
        ) : null}
      </StyledProblem>
      <StyledChat
        initialMessages={INITIAL_MESSAGES}
        requestOpts={REQUEST_OPTS}
        parseContent={parseContent}
      />
    </div>
  )
}

export default TutorPage
