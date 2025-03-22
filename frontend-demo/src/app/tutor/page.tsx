"use client"
import React from "react"
import { AiChat, AiChatProps } from "@mitodl/smoot-design/ai"
import styled from "@emotion/styled"
import { extractJSONFromComment } from "../utils"

const INITIAL_MESSAGES: AiChatProps["initialMessages"] = [
  {
    content: "Hi! Do you need any help solving the problem?",
    role: "assistant",
  },
]

const EDX_MODULE_ID =
  "block-v1:MITx+15.071x+2T2020+type@problem+block@6f74a8e1904349c8b38903f26037e7ff"

const BLOCK_SIBLINGS = [
  "block-v1:MITx+15.071x+2T2020+type@html+block@2156d1b330cc456299590d1a1b2d47ef",
  "block-v1:MITx+15.071x+2T2020+type@problem+block@6f74a8e1904349c8b38903f26037e7ff",
  "block-v1:MITx+15.071x+2T2020+type@problem+block@b2d7e452656c417998f96adc91433c5b",
  "block-v1:MITx+15.071x+2T2020+type@problem+block@809b22e95b564420b57d74c5a2ececad",
  "block-v1:MITx+15.071x+2T2020+type@problem+block@d8df9513dd714f08b607480525e873d7",
  "block-v1:MITx+15.071x+2T2020+type@problem+block@a2c5712365164050b4bfb908c876b476",
  "block-v1:MITx+15.071x+2T2020+type@problem+block@2180716bc04b42868fcd6209e135026d",
  "block-v1:MITx+15.071x+2T2020+type@problem+block@683b3b0ab7ac4110bbb57ef69f84f178",
  "block-v1:MITx+15.071x+2T2020+type@problem+block@8c24dd36fcd7435390105d794a21c7b6",
  "block-v1:MITx+15.071x+2T2020+type@problem+block@c3c2c089da5247acaedca13c55b02050",
  "block-v1:MITx+15.071x+2T2020+type@problem+block@b9a4c0bab3904dd695cc91108f9af5ff",
  "block-v1:MITx+15.071x+2T2020+type@problem+block@4d17f111c6d0481fbf115bd984ca6e45",
  "block-v1:MITx+15.071x+2T2020+type@problem+block@3840cd113b26457584e0f16bad4eef76",
  "block-v1:MITx+15.071x+2T2020+type@problem+block@09e29feea45c4eb3bd4d83dacaa35870",
  "block-v1:MITx+15.071x+2T2020+type@html+block@496c1aed6e3d40a08a749c2882fcf61e",
  "block-v1:MITx+15.071x+2T2020+type@discussion+block@e25d79a549874401a0ea770589fc9825",
]

const REQUEST_OPTS: AiChatProps["requestOpts"] = {
  apiUrl: `${process.env.NEXT_PUBLIC_MITOL_API_BASE_URL}/http/tutor_agent/`,
  transformBody(messages) {
    const message = messages[messages.length - 1].content
    return {
      message: message,
      edx_module_id: EDX_MODULE_ID,
      block_siblings: BLOCK_SIBLINGS,
    }
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
  marginTop: "16px",
  marginBottom: "16px",
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

  return (
    <div>
      <StyledProblem>
        The tutor is here to help with Problem 1.1 from
        <a
          href={
            "https://courses-qa.mitxonline.mit.edu/learn/course/course-v1:MITx+15.071x+2T2020/block-v1:MITx+15.071x+2T2020+type@sequential+block@60d93a44280348d7a0a16663f92af0f7/block-v1:MITx+15.071x+2T2020+type@vertical+block@8f18405a93f245b183429f6c8ac07f64"
          }
        >
          {" "}
          this problem set
        </a>
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
