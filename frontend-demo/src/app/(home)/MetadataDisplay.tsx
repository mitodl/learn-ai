import * as React from "react"
import { useAiChat } from "@mitodl/smoot-design/ai"
import { Button, styled } from "@mitodl/smoot-design"
import Typography from "@mui/material/Typography"
import Alert from "@mui/material/Alert"
import Stack from "@mui/material/Stack"

const MetadataContainer = styled.div({
  maxWidth: "100%",
  overflow: "auto",
  display: "flex",
  flexDirection: "column",
  gap: "8px",
})

function extractMetadata(input: string): Record<string, unknown> | null {
  const match = input.match(/<!--([\s\S]*?)-->/)
  if (!match) return null

  try {
    return JSON.parse(match[1].trim())
  } catch (error) {
    console.error("Failed to parse metadata as JSON:", error)
    return null
  }
}

const MetadataDisplay: React.FC = () => {
  const [showMeta, setShowMeta] = React.useState(false)
  const { messages } = useAiChat()
  const assistantMessages = messages.filter(
    (message) => message.role === "assistant",
  )
  const lastMessage = assistantMessages[assistantMessages.length - 1]
  const metadata = lastMessage ? extractMetadata(lastMessage?.content) : null
  if (!metadata) {
    return (
      <MetadataContainer>
        <Typography variant="body3">
          <em>No metadata available</em>
        </Typography>
      </MetadataContainer>
    )
  }

  const error = metadata.error

  return (
    <MetadataContainer>
      <Stack direction="row" spacing={2} alignItems="center">
        <Button
          variant="text"
          onClick={() => setShowMeta((current) => !current)}
        >
          {showMeta ? "Hide Metadata" : "Show Metadata"}
        </Button>
        {error ? (
          <Alert severity="error">
            <pre>
              <code>{JSON.stringify(error, null)}</code>
            </pre>
          </Alert>
        ) : null}
      </Stack>

      {showMeta && <pre>{JSON.stringify(metadata, null, 2)}</pre>}
    </MetadataContainer>
  )
}

export default MetadataDisplay
